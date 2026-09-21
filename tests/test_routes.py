"""The three routes: the distillation loss, the channel transfer, and one tiny fold."""
import numpy as np
import torch
import torch.nn.functional as F

import models as M
import routes as R


def test_distillation_loss_mixes_the_two_terms():
    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(16, 4, generator=g), torch.randn(16, 4, generator=g)
    y = torch.randint(0, 4, (16,), generator=g)
    assert torch.allclose(R.distillation_loss(s, t, y, T=3.0, alpha=0.0), F.cross_entropy(s, y))
    soft = F.kl_div(F.log_softmax(s / 3, 1), F.softmax(t / 3, 1), reduction="batchmean") * 9
    assert torch.allclose(R.distillation_loss(s, t, y, T=3.0, alpha=1.0), soft)


def test_pruning_copies_the_highest_l1_channels_from_the_teacher():
    torch.manual_seed(0)
    teacher = M.AccNet((8, 16, 16), 4)
    small = R.transfer_surviving_channels(teacher, M.AccNet((4, 8, 8), 4))
    conv_t = [m for m in teacher.features if isinstance(m, torch.nn.Conv1d)]
    conv_s = [m for m in small.features if isinstance(m, torch.nn.Conv1d)]
    keep = torch.sort(torch.argsort(conv_t[0].weight.abs().sum((1, 2)), descending=True)[:4])[0]
    assert torch.equal(conv_s[0].weight, conv_t[0].weight[keep])
    x = torch.randn(2, 3, 50)
    small.eval(); small(x)          # the shapes must line up end to end


def test_one_fold_returns_every_route_at_every_width():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 90)
    X = rng.normal(size=(90, 3, 50)).astype(np.float32)
    a = np.repeat([0, 1, 2], 30)
    rows = R.run_fold(X, y, a, 1, [(4, 8, 8), (2, 4, 4)], 3, epochs=1)
    routes = sorted((r["route"], r["widths"]) for r in rows)
    assert routes.count(("teacher", "(64, 128, 128)")) == 1
    for w in ("(4, 8, 8)", "(2, 4, 4)"):
        for route in ("distil", "prune", "scratch"):
            assert (route, w) in routes
    assert all(r["hold"] == "1" for r in rows)


def test_equal_schedule_gives_pruning_the_other_routes_schedule():
    import train as TR
    assert R.prune_schedule(30) == (R.prune_epochs(30), R.PRUNE_LR)
    assert R.prune_schedule(30, equal=True) == (30, TR.LR)
