"""The integrity check and the windowing. The first two tests catch the errors that
make leave-one-animal-out mean something other than what it says."""
import numpy as np

import data as D


def _animal(rng, n=1000):
    return rng.normal(size=(n, 3)).astype(np.float32)


def test_a_block_shared_by_three_animals_counts_three_pairs_but_one_distinct_block():
    rng = np.random.default_rng(0)
    acc = {k: _animal(rng) for k in range(4)}
    shared = rng.normal(size=(100, 3)).astype(np.float32)
    for k in (0, 1, 2):
        acc[k][300:400] = shared
    data = {k: (v, np.array(["RES"] * len(v))) for k, v in acc.items()}
    rep = D.duplicate_report(data)
    assert rep["pairs"] == 6
    assert rep["pairs_sharing"] == 3
    assert rep["shared"] == 3, "summed over pairs, a block in three animals counts three times"
    assert rep["distinct"] == 1, "as signal, it is one block"


def test_a_motionless_stretch_is_not_evidence_of_duplication():
    rng = np.random.default_rng(1)
    a, b = _animal(rng), _animal(rng)
    a[500:600] = 0.5
    b[500:600] = 0.5
    data = {0: (a, np.array(["RES"] * 1000)), 1: (b, np.array(["RES"] * 1000))}
    assert D.duplicate_report(data)["shared"] == 0


def test_deduplication_keeps_the_first_copy_only():
    rng = np.random.default_rng(2)
    a, b = _animal(rng), _animal(rng)
    b[200:300] = a[200:300]
    lab = np.array(["RES"] * 1000)
    out = D.drop_cross_animal_duplicates({0: (a, lab), 1: (b, lab)})
    assert len(out[0][0]) == 1000
    assert len(out[1][0]) == 900
    assert D.duplicate_report(out)["shared"] == 0


def test_no_window_straddles_a_behaviour_change():
    rng = np.random.default_rng(3)
    acc = _animal(rng, 400)
    lab = np.array(["RES"] * 120 + ["MOV"] * 130 + ["XXX"] * 150)
    X, y, a = D.windows({7: (acc, lab)}, ["RES", "MOV"], win=50, stride=25)
    assert X.shape[1:] == (3, 50)
    assert set(y.tolist()) == {0, 1}
    assert np.all(a == 7)
    starts = [j for j in range(0, 350, 25) if len(set(lab[j:j + 50])) == 1 and lab[j] != "XXX"]
    assert len(y) == len(starts), "a window with a label change, or an unmodelled label, was kept"
