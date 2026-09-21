"""The three compression routes at matched footprint, run for one held-out group.

Comparing distillation with pruning says which of the two is better. Comparing both
with a network of the same widths trained from hard labels says whether either did
anything beyond making the network smaller, which is the question a practitioner
has. Without that third arm the benefit of a route and the cost of shrinking are
not separately identifiable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models as M
import train as TR

PRUNE_LR = 1e-3


def prune_epochs(epochs):
    """Fine-tuning epochs for the pruned network: half the full schedule, at least 8."""
    return max(8, epochs // 2)


def distillation_loss(logits, teacher_logits, target, T=4.0, alpha=0.5):
    """alpha * T^2 * KL(teacher_T || small_T) + (1 - alpha) * CE(small, target).

    Softening by T shrinks the soft term's gradients by about 1/T^2, so the T^2 factor
    is what keeps high temperatures from silently switching the soft term off. A
    temperature sweep without it looks like evidence against distillation.
    """
    hard = F.cross_entropy(logits, target)
    soft = F.kl_div(F.log_softmax(logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    return alpha * soft + (1.0 - alpha) * hard


def channel_l1_scores(model):
    """L1 norm of each output channel's kernel, one tensor per convolution."""
    return [m.weight.detach().abs().sum(dim=(1, 2))
            for m in model.features if isinstance(m, nn.Conv1d)]


def transfer_surviving_channels(src, dst):
    """Copy the highest-L1 channels of every block of `src` into the narrower `dst`.

    Each convolution keeps its top-scoring output channels (as many as `dst` has, in
    their original order), restricted on the input side to the channels the previous
    block kept. Batch norm parameters and running statistics follow their channels,
    and the head keeps the columns of the last block's survivors.

    This copy is the whole of the pruning route. A "pruned" network that is
    re-initialised before fine-tuning is just the same-size network with a different
    training schedule, and comparing it with the third arm compares a thing with
    itself.
    """
    scores = channel_l1_scores(src)
    s_convs = [m for m in src.features if isinstance(m, nn.Conv1d)]
    d_convs = [m for m in dst.features if isinstance(m, nn.Conv1d)]
    s_bns = [m for m in src.features if isinstance(m, nn.BatchNorm1d)]
    d_bns = [m for m in dst.features if isinstance(m, nn.BatchNorm1d)]
    prev = torch.arange(M.CH)
    with torch.no_grad():
        for i, (sc, dc) in enumerate(zip(s_convs, d_convs)):
            keep = torch.argsort(scores[i], descending=True)[:dc.out_channels]
            keep, _ = torch.sort(keep)
            dc.weight.copy_(sc.weight[keep][:, prev])
            dc.bias.copy_(sc.bias[keep])
            d_bns[i].weight.copy_(s_bns[i].weight[keep])
            d_bns[i].bias.copy_(s_bns[i].bias[keep])
            d_bns[i].running_mean.copy_(s_bns[i].running_mean[keep])
            d_bns[i].running_var.copy_(s_bns[i].running_var[keep])
            prev = keep
        dst.head.weight.copy_(src.head.weight[:, prev])
        dst.head.bias.copy_(src.head.bias)
    return dst


def run_fold(X, y, a, hold, widths_list, n_classes, epochs=TR.EPOCHS, T=4.0,
             alpha=0.5, seed=0, grid=None):
    """Teacher plus, at every width, distillation, pruning and the same-size control.

    Returns one row per trained model, every model scored after int8 weight
    quantisation. `grid`, if given, is a list of (T, alpha) pairs and replaces the
    single distillation setting.

    What each route gets:
      distillation  `epochs` at TR.LR, teacher logits, no class weights
      pruning       teacher's surviving channels, then prune_epochs(epochs) epochs at
                    PRUNE_LR, class weights
      same size     fresh initialisation, `epochs` at TR.LR, class weights

    Every network is built in a fixed order, and building one advances the random
    generator, including the unused `stud` built only to read a footprint. Reordering
    or removing a construction changes every later initialisation.
    """
    fd = TR.fold_data(X, y, a, hold, n_classes)
    if fd is None:
        return []
    Xtr, ytr, Xte, yte = fd
    cw = TR.class_weights(ytr, n_classes)
    teacher = TR.train_teacher(Xtr, ytr, n_classes, epochs=epochs, seed=seed)
    rows = [TR.row("teacher", TR.TEACHER_WIDTHS, hold, teacher.footprint_bytes(),
                   TR.evaluate(M.quantise_int8(teacher), Xte, yte, n_classes), seed=seed)]

    for w in widths_list:
        stud = M.AccNet(w, n_classes)
        fpw = stud.footprint_bytes()

        for (Tv, av) in (grid if grid else [(T, alpha)]):
            s = M.AccNet(w, n_classes)
            TR.fit(s, Xtr, ytr, epochs=epochs, teacher=teacher, T=Tv, alpha=av, seed=seed)
            rows.append(TR.row("distil", w, hold, fpw,
                               TR.evaluate(M.quantise_int8(s), Xte, yte, n_classes),
                               T=float(Tv), alpha=float(av), seed=seed))

        p = M.AccNet(w, n_classes)
        transfer_surviving_channels(teacher, p)
        TR.fit(p, Xtr, ytr, epochs=prune_epochs(epochs), lr=PRUNE_LR, seed=seed, class_weight=cw)
        rows.append(TR.row("prune", w, hold, fpw,
                           TR.evaluate(M.quantise_int8(p), Xte, yte, n_classes), seed=seed))

        b = M.AccNet(w, n_classes)
        TR.fit(b, Xtr, ytr, epochs=epochs, seed=seed, class_weight=cw)
        rows.append(TR.row("scratch", w, hold, fpw,
                           TR.evaluate(M.quantise_int8(b), Xte, yte, n_classes), seed=seed))
    return rows
