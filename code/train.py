"""Standardisation, the training loop, scoring, and the teacher under
leave-one-animal-out.

Every route in the study goes through `fit` and `evaluate`, so anything that differs
between routes is visible in the arguments they pass (see routes.run_fold).
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
import models as M

TEACHER_WIDTHS = (64, 128, 128)
EPOCHS = 30
LR = 3e-3


def standardise(Xtr, Xte):
    """Per-channel z-score using statistics of the training windows only.

    Fitting the mean and standard deviation on all windows lets the held-out animal
    shape its own inputs, which is a leak even though no label is involved.
    """
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def class_weights(y, n_classes):
    """Inverse-frequency weights, n / (k * count), from the training labels only."""
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    return cnt.sum() / (n_classes * np.maximum(cnt, 1))


def fit(model, Xtr, ytr, epochs=EPOCHS, bs=256, lr=LR, teacher=None,
        T=4.0, alpha=0.5, seed=0, class_weight=None):
    """Adam with cosine annealing over `epochs`, minibatches drawn by a seeded permutation.

    With a teacher, the loss is `routes.distillation_loss` on the teacher's logits,
    computed once up front in eval mode, and `class_weight` is not used: the
    distillation loss is unweighted, while the other routes pass class weights.

    The seed is set at the start of training, after the network already exists, so it
    fixes the minibatch order but not the initial weights.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    n = len(Xt)
    cw = None if class_weight is None else torch.from_numpy(class_weight.astype(np.float32))
    if teacher is not None:
        import routes
        teacher.eval()
        with torch.no_grad():
            tl = teacher(Xt)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = model(Xt[idx])
            if teacher is None:
                loss = F.cross_entropy(out, yt[idx], weight=cw)
            else:
                loss = routes.distillation_loss(out, tl[idx], yt[idx], T=T, alpha=alpha)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model


def evaluate(model, Xte, yte, n_classes):
    """Score a model on held-out windows, beside the majority-class floor.

    Returns macro F1, balanced accuracy, accuracy, and the macro F1 and accuracy of
    always predicting the held-out set's most common class.

    Macro F1 has a choice hidden in it. Averaged over every modelled class, a class
    the held-out animal never shows scores zero and drags the mean down; averaged
    over only the classes present, it does not. Both are returned: `macro_f1` over
    all modelled classes, which is the headline figure, and `macro_f1_present` over
    the present ones.
    """
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(Xte)).argmax(1).numpy()
    labels = list(range(n_classes))
    present = sorted(set(yte.tolist()))
    maj = np.bincount(yte, minlength=n_classes).argmax()
    return dict(
        macro_f1=f1_score(yte, p, average="macro", labels=labels, zero_division=0),
        macro_f1_present=f1_score(yte, p, average="macro", labels=present, zero_division=0),
        bal_acc=balanced_accuracy_score(yte, p),
        acc=float((p == yte).mean()),
        maj_f1=f1_score(yte, np.full_like(yte, maj), average="macro", labels=labels,
                        zero_division=0),
        maj_acc=float((yte == maj).mean()))


def row(route, widths, hold, fp, scores, T=None, alpha=None, seed=0):
    """One result row: identity, footprint, scores. Plain Python types, so JSON takes it."""
    r = dict(route=route, widths=str(tuple(widths)), hold=str(hold), seed=int(seed),
             T=T, alpha=alpha)
    r.update({k: int(v) for k, v in fp.items()})
    r.update({k: float(v) for k, v in scores.items()})
    return r


def fold_data(X, y, a, hold, n_classes):
    """Split off one held-out group and standardise. Returns None if the fold is unusable.

    A fold is unusable if the held-out group has fewer than 20 windows or the
    training groups lack a class. A dropped fold contributes no rows, so it shows up
    as missing rows in the results files.
    """
    tr, te = a != hold, a == hold
    if te.sum() < 20 or len(np.unique(y[tr])) < n_classes:
        return None
    Xtr, Xte = standardise(X[tr], X[te])
    return Xtr, y[tr], Xte, y[te]


def train_teacher(Xtr, ytr, n_classes, epochs=EPOCHS, seed=0, widths=TEACHER_WIDTHS):
    """Seed the generator, build the teacher, train it with class weights, return it.

    The generator is seeded before the network is built, so the initial weights are
    fixed by `seed` too.
    """
    torch.manual_seed(seed)
    teacher = M.AccNet(widths, n_classes)
    fit(teacher, Xtr, ytr, epochs=epochs, seed=seed,
        class_weight=class_weights(ytr, n_classes))
    return teacher


def save(name, rows, config, out_dir=None):
    """Write `rows` and the configuration that produced them to results/<name>.json.

    The library versions and the date are recorded beside the rows.
    """
    import datetime, json, os, platform, sklearn
    import data as D
    out_dir = out_dir or D.RESULTS
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(config, python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__.split("+")[0], sklearn=sklearn.__version__,
                date=datetime.date.today().isoformat())
    path = os.path.join(out_dir, name + ".json")
    json.dump({"meta": meta, "rows": rows}, open(path, "w"), indent=1)
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def cli(default_seed=0):
    """--seed, --out and --procs for every run script."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=default_seed)
    ap.add_argument("--out", default=None, help="results directory")
    ap.add_argument("--procs", type=int, default=6)
    return ap.parse_args()
