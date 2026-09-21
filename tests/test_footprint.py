"""Footprint arithmetic and int8 weight quantisation."""
import numpy as np
import torch

import models as M


def test_parameter_count_matches_the_architecture():
    net = M.AccNet((8, 16, 16), 4)
    by_hand = (3 * 5 * 8 + 8) + 2 * 8 + (8 * 5 * 16 + 16) + 2 * 16 + (16 * 5 * 16 + 16) + 2 * 16 + (16 * 4 + 4)
    assert net.n_params() == by_hand


def test_ram_is_the_largest_pair_of_consecutive_tensors():
    net = M.AccNet((10, 20, 30), 4, win=40)
    # input 120, conv 400, pool 200, conv 400, pool 200, conv 300, pool 150
    assert net.peak_activation() == 600
    fp = net.footprint_bytes()
    assert fp["ram"] == 600 and fp["flash"] == net.n_params()
    assert fp["total"] == fp["flash"] + fp["ram"]


def test_the_network_runs_on_a_window():
    out = M.AccNet((8, 16, 16), 7)(torch.zeros(5, 3, 50))
    assert tuple(out.shape) == (5, 7)


def test_int8_weights_sit_on_a_grid_of_at_most_255_levels():
    torch.manual_seed(0)
    net = M.AccNet((8, 16, 16), 4)
    q = M.quantise_int8(net)
    for a, b in zip(net.modules(), q.modules()):
        if isinstance(a, (torch.nn.Conv1d, torch.nn.Linear)):
            w, s = b.weight.detach(), a.weight.detach().abs().max() / 127
            steps = w / s
            assert torch.allclose(steps, torch.round(steps), atol=1e-3)
            assert steps.abs().max() <= 127 + 1e-3
            assert not torch.equal(a.weight, b.weight) or s == 0
    assert net is not q
