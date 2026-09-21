"""Scoring and the training loop, on toy data small enough to run in seconds."""
import numpy as np
import torch

import models as M
import train as TR


def test_standardisation_uses_training_windows_only():
    Xtr = np.zeros((10, 3, 4), np.float32); Xtr[:5] = 2.0
    Xte = np.full((3, 3, 4), 100.0, np.float32)
    a, b = TR.standardise(Xtr, Xte)
    assert abs(a.mean()) < 1e-5
    assert b.min() > 50, "the held-out windows must not move the scaling"


def test_macro_f1_over_all_classes_and_over_present_classes_differ_when_one_is_absent():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(len(x), 4)
            out[torch.arange(len(x)), x[:, 0, 0].long()] = 1.0
            return out
    y = np.array([0, 0, 1, 1, 2, 2])
    X = np.zeros((6, 3, 50), np.float32); X[:, 0, 0] = y
    r = TR.evaluate(Fixed(), X, y, 4)
    assert abs(r["macro_f1_present"] - 1.0) < 1e-9
    assert abs(r["macro_f1"] - 0.75) < 1e-9
    assert abs(r["maj_acc"] - 1 / 3) < 1e-9


def test_class_weights_are_inverse_frequency():
    w = TR.class_weights(np.array([0, 0, 0, 1]), 2)
    assert np.allclose(w, [4 / 6, 4 / 2])


def test_training_learns_a_separable_toy_problem():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 240)
    X = rng.normal(size=(240, 3, 50)).astype(np.float32) * 0.1
    X[:, 1, :] += (2 * y - 1)[:, None]
    net = M.AccNet((8, 16, 16), 2)
    TR.fit(net, X, y, epochs=4, bs=32, seed=0)
    assert TR.evaluate(net, X, y, 2)["acc"] > 0.95


def test_a_seeded_teacher_is_repeatable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(64, 3, 50)).astype(np.float32)
    y = rng.integers(0, 3, 64)
    a = TR.train_teacher(X, y, 3, epochs=1, seed=5, widths=(4, 8, 8))
    b = TR.train_teacher(X, y, 3, epochs=1, seed=5, widths=(4, 8, 8))
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q), "same seed, same data: the teacher must come out identical"
