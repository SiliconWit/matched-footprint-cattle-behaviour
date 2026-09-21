"""The statistics every route comparison goes through."""
import numpy as np

import analyse as A


def _rows(values):
    rows = []
    for (w, hold, route), v in values.items():
        rows.append(dict(route=route, widths=w, hold=hold, macro_f1=v, total=2048, params=1,
                         ram=0, maj_f1=0.1, maj_acc=0.5))
    return rows


def test_paired_differences_line_up_by_width_and_animal():
    vals = {}
    for w in ("(8, 16, 16)", "(4, 8, 8)"):
        for h in "012":
            base = {"0": 0.2, "1": 0.9, "2": 0.5}[h]
            vals[(w, h, "scratch")] = base
            vals[(w, h, "prune")] = base + 0.01 + 0.001 * int(h)
            vals[(w, h, "distil")] = base
    piv = A.pivot(_rows(vals))
    d = [c["prune"] - c["scratch"] for c in piv.values()]
    assert np.allclose(sorted(d), [0.01, 0.01, 0.011, 0.011, 0.012, 0.012])
    p = A.paired([c["prune"] for c in piv.values()], [c["scratch"] for c in piv.values()])
    assert p["n"] == 6 and p["wins"] == 6 and abs(p["mean"] - 0.011) < 1e-12


def test_the_leak_is_priced_on_matched_runs_only():
    clean = _rows({("(16, 32, 32)", "0", "scratch"): 0.5, ("(16, 32, 32)", "1", "scratch"): 0.6,
                   ("(8, 16, 16)", "0", "scratch"): 0.1})
    leaky = _rows({("(16, 32, 32)", "0", "scratch"): 0.7, ("(16, 32, 32)", "1", "scratch"): 0.9})
    out = A.summarise_leak(leaky, clean)
    assert out["n"] == 2 and abs(out["mean"] - 0.25) < 1e-12


def test_a_collapse_is_a_cell_far_below_the_same_size_control():
    vals = {}
    for h, gap in zip("012", (0.05, -0.05, -0.2)):
        vals[("(8, 16, 16)", h, "scratch")] = 0.6
        vals[("(8, 16, 16)", h, "distil")] = 0.6
        vals[("(8, 16, 16)", h, "prune")] = 0.6 + gap
    assert A.collapses(A.pivot(_rows(vals))) == [["(8, 16, 16)", "2"]]
