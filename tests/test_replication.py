"""Folds that keep animals whole, and a class cap that does not move with the seed."""
import numpy as np

import run_calf as C


def test_every_animal_lands_in_exactly_one_fold():
    ids = np.array(["1302", "1305", "1302", "1311", "1307", "1305", "1320", "1311"])
    f = C.assign_folds(ids, 3)
    for i in set(ids.tolist()):
        assert len(set(f[ids == i].tolist())) == 1
    assert set(f.tolist()) <= {0, 1, 2}


def test_the_class_cap_is_repeatable_and_respected():
    y = np.array([0] * 50 + [1] * 5 + [2] * 20)
    a, b = C.cap_classes(y, 10), C.cap_classes(y, 10)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)
    assert np.bincount(y[a]).tolist() == [10, 5, 10]
