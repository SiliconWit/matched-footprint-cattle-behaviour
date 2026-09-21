"""Gather the results files into results/summary.json, the one file every reported
number is generated from.

Reads results/integrity.json, frontier.json, leak.json, calf.json and grid.json (seed
0), and every results/seed-N/ directory that holds further seeds of the same runs.
No reported number is computed anywhere else: build_numbers.py turns
summary.json into LaTeX macros and make_figs.py draws the figures from it.

    python3 analyse.py [results_dir]
"""
import json, os, re, sys
import numpy as np
from scipy import stats
import data as D

ROUTES = ["distil", "prune", "scratch"]


def load(name, res=D.RESULTS):
    p = os.path.join(res, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def paired(a, b):
    """Paired comparison of a against b: mean and sd of a - b, paired t, Wilcoxon, wins.

    A Wilcoxon test on two identical vectors is undefined and is reported as NaN rather
    than crashing. `wins` counts strictly positive differences; ties are not wins.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, pt = stats.ttest_rel(a, b)
    try:
        _, pw = stats.wilcoxon(a, b)
    except ValueError:
        pw = float("nan")
    d = a - b
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), n=int(len(d)),
                t=float(t), p_t=float(pt), p_wilcoxon=float(pw), wins=int((d > 0).sum()))


def unpaired(a, b):
    """Welch's t-test of a against b, ignoring the pairing: what comparing group means
    with their spreads would conclude."""
    t, pt = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False)
    return dict(mean=float(np.mean(a) - np.mean(b)), t=float(t), p_t=float(pt))


def pivot(rows, value="macro_f1"):
    """{(widths, hold): {route: value}} for the non-teacher rows, complete cells only,
    keyed in sorted order so every paired vector lines up by width and animal."""
    cells = {}
    for r in rows:
        if r["route"] == "teacher":
            continue
        cells.setdefault((r["widths"], str(r["hold"])), {})[r["route"]] = r[value]
    return {k: cells[k] for k in sorted(cells) if all(x in cells[k] for x in ROUTES)}


def first_by_widths(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["widths"], r[key])
    return out


def std1(x):
    return float(np.std(np.asarray(x, float), ddof=1))


def group_mean(rows, key, value="macro_f1"):
    g = {}
    for r in rows:
        g.setdefault(str(r[key]), []).append(r[value])
    return {k: float(np.mean(v)) for k, v in sorted(g.items())}


def summarise_frontier(rows, value="macro_f1"):
    """Teacher, majority floor, frontier, decomposition and paired tests for one set of rows."""
    S = {"n_animals": len({str(r["hold"]) for r in rows})}
    tea = [r for r in rows if r["route"] == "teacher"]
    S["teacher"] = dict(macro_f1=float(np.mean([r[value] for r in tea])),
                        sd=std1([r[value] for r in tea]),
                        kb=float(tea[0]["total"] / 1024), params=int(tea[0]["params"]))
    S["majority_f1"] = float(np.mean([r["maj_f1"] for r in rows]))
    S["majority_acc"] = float(np.mean([r["maj_acc"] for r in rows]))

    piv = pivot(rows, value)
    total, params, ram = (first_by_widths(rows, k) for k in ("total", "params", "ram"))
    kb = {w: total[w] / 1024 for w in total}
    frontier = []
    for w in sorted(kb, key=lambda x: -kb[x]):
        row = dict(widths=w, kb=float(kb[w]), params=int(params[w]), ram_kb=float(ram[w] / 1024))
        cells = [v for (ww, _), v in piv.items() if ww == w]
        if cells:
            for r in ROUTES:
                row[r] = float(np.mean([c[r] for c in cells]))
                row[r + "_sd"] = std1([c[r] for c in cells])
        frontier.append(row)
    S["frontier"] = frontier

    A_t = S["teacher"]["macro_f1"]
    dec = [dict(kb=row["kb"], C=float(A_t - row["scratch"]),
                **{f"B_{r}": float(row[r] - row["scratch"]) for r in ("distil", "prune")})
           for row in frontier if "scratch" in row]
    S["decomposition"] = dec
    S["capacity_deficit_max"] = float(max(d["C"] for d in dec))
    S["capacity_deficit_min"] = float(min(d["C"] for d in dec))
    S["smallest"] = min(frontier, key=lambda r: r["kb"])

    vec = {r: [c[r] for c in piv.values()] for r in ROUTES}
    S["paired"] = {a: paired(vec[a], vec["scratch"]) for a in ("distil", "prune")}
    S["paired"]["prune_vs_distil"] = paired(vec["prune"], vec["distil"])
    S["unpaired"] = {a: unpaired(vec[a], vec["scratch"]) for a in ("distil", "prune")}
    sub = [r for r in rows if r["route"] != "teacher"]
    S["between_animal_sd"] = std1(list(group_mean(sub, "hold", value).values()))
    S["between_route_sd"] = std1(list(group_mean(sub, "route", value).values()))
    widths = sorted({w for w, _ in piv})
    S["prune_gain_by_kb"] = sorted(
        [dict(kb=float(kb[w]), gain=float(np.mean([c["prune"] - c["scratch"]
                                                   for (ww, _), c in piv.items() if ww == w])))
         for w in widths], key=lambda d: -d["kb"])
    return S


def summarise_integrity(integ):
    b = integ["beef"]
    det = sorted(b["detail"], key=lambda t: -t[2])
    out = {"beef": dict(pairs=b["pairs"], shared=b["shared"], seconds=b["seconds"]),
           "calf": dict(pairs=integ["calf"]["pairs"], shared=integ["calf"]["shared"],
                        seconds=integ["calf"]["seconds"]),
           "pair_max": dict(cows=[det[0][0] + 1, det[0][1] + 1], blocks=det[0][2]),
           "pair_min": dict(cows=[det[-1][0] + 1, det[-1][1] + 1], blocks=det[-1][2])}
    w = integ["windows"]
    windows = dict(classes=w["classes"], windows_clean=w["clean"]["total"],
                   windows_raw=w["raw"]["total"], per_class_clean=w["clean"]["per_class"],
                   animals=b["animals"])
    recheck = dict(block=D.BLOCK, fs=D.FS, samples_per_animal=b["samples_per_animal"][0],
                   all_animals_same_length=b["all_animals_same_length"],
                   varying_blocks_per_animal={str(int(k) + 1): v
                                              for k, v in b["varying_blocks"].items()},
                   pairs=b["pairs"], pairs_sharing=b["pairs_sharing"],
                   pair_occurrences=b["shared"], pair_occurrence_hours=b["seconds"] / 3600,
                   distinct_blocks=b["distinct"],
                   distinct_hours=b["distinct"] * D.BLOCK / D.FS / 3600,
                   mean_fraction_of_animal_duplicated=b["mean_fraction"])
    return out, windows, recheck


def summarise_leak(leak_rows, clean_rows, value="macro_f1"):
    """Match every as-published run to the de-duplicated run with the same route, width,
    held-out animal and seed, and compare them pairwise."""
    clean = {(r["widths"], str(r["hold"]), r["route"], r.get("seed", 0)): r for r in clean_rows}
    m = [(r, clean[(r["widths"], str(r["hold"]), r["route"], r.get("seed", 0))])
         for r in leak_rows if (r["widths"], str(r["hold"]), r["route"], r.get("seed", 0)) in clean]
    out = paired([x[value] for x, _ in m], [y[value] for _, y in m])
    by = {}
    for x, y in m:
        if x["route"] != "teacher":
            by.setdefault(x["route"], []).append(x[value] - y[value])
    out.update(as_published=float(np.mean([x[value] for x, _ in m])),
               deduplicated=float(np.mean([y[value] for _, y in m])),
               by_route={k: float(np.mean(v)) for k, v in sorted(by.items())})
    return out


def summarise_calf(rows, value="macro_f1"):
    sub = [r for r in rows if r["route"] != "teacher"]
    piv = pivot(rows, value)
    total = first_by_widths(rows, "total")
    kb = {w: total[w] / 1024 for w in total}
    tea = [r[value] for r in rows if r["route"] == "teacher"]
    cteach = float(np.mean(tea))
    widths = sorted({w for w, _ in piv}, key=lambda x: -kb[x])
    mean = lambda w, r: float(np.mean([c[r] for (ww, _), c in piv.items() if ww == w]))
    vec = {r: [c[r] for c in piv.values()] for r in ROUTES}
    return dict(
        n_folds=len({str(r["hold"]) for r in sub}),
        teacher_kb=float([r for r in rows if r["route"] == "teacher"][0]["total"] / 1024),
        frontier=[dict(widths=w, kb=float(kb[w]), **{r: mean(w, r) for r in ROUTES}) for w in widths],
        capacity_deficit=[dict(kb=float(kb[w]), C=float(cteach - mean(w, "scratch")),
                               **{f"B_{r}": float(mean(w, r) - mean(w, "scratch"))
                                  for r in ("distil", "prune")}) for w in widths],
        teacher=cteach,
        majority_f1=float(np.mean([r["maj_f1"] for r in rows])),
        by_route={r: float(np.mean(vec[r])) for r in ROUTES},
        paired={a: paired(vec[a], vec["scratch"]) for a in ("distil", "prune")})


def summarise_grid(rows, value="macro_f1"):
    """Benefit of every (T, alpha) cell over the same-size control, per width, mean over folds."""
    scratch = {(r["widths"], str(r["hold"])): r[value] for r in rows if r["route"] == "scratch"}
    cells = {}
    for r in rows:
        if r["route"] == "distil":
            cells.setdefault((r["widths"], r["T"], r["alpha"]), []).append(
                r[value] - scratch[(r["widths"], str(r["hold"]))])
    total = first_by_widths(rows, "total")
    table = [dict(widths=w, kb=float(total[w] / 1024), T=T, alpha=a,
                  benefit=float(np.mean(v)), n=len(v))
             for (w, T, a), v in sorted(cells.items())]
    best = max(table, key=lambda c: c["benefit"])
    worst = min(table, key=lambda c: c["benefit"])
    return dict(cells=table, n_cells=len(table), best=best["benefit"], best_cell=best,
                worst=worst["benefit"], worst_cell=worst,
                kb=sorted({c["kb"] for c in table}, reverse=True))


def seed_summary(res, frontier=None):
    """The seed-dependent quantities for one directory of runs, from whatever it holds."""
    front = frontier or load("frontier", res)
    out = {}
    tea = load("teacher", res)
    if front:
        f = summarise_frontier(front["rows"])
        out.update(teacher=f["teacher"]["macro_f1"], paired=f["paired"],
                   capacity_deficit_max=f["capacity_deficit_max"],
                   prune_gain_by_kb=f["prune_gain_by_kb"],
                   between_animal_sd=f["between_animal_sd"])
        if all("macro_f1_present" in r for r in front["rows"]):
            out["paired_present"] = summarise_frontier(front["rows"], "macro_f1_present")["paired"]
        leak = load("leak", res)
        if leak:
            out["contamination"] = summarise_leak(leak["rows"], front["rows"])
            if all("macro_f1_present" in r for r in leak["rows"] + front["rows"]):
                out["contamination_present"] = summarise_leak(leak["rows"], front["rows"],
                                                              "macro_f1_present")
    elif tea:
        out["teacher"] = float(np.mean([r["macro_f1"] for r in tea["rows"]]))
    calf = load("calf", res)
    if calf:
        c = summarise_calf(calf["rows"])
        out["calf"] = dict(paired=c["paired"], teacher=c["teacher"],
                           capacity_deficit=c["capacity_deficit"])
    return out


def run_meta(res):
    """Configuration, library versions and dates recorded beside each results file."""
    return {n: d["meta"] for n in ("teacher", "frontier", "leak", "grid", "calf")
            for d in [load(n, res)] if d}


def main(res=D.RESULTS):
    integ, front, leak, calf = (load(n, res) for n in ("integrity", "frontier", "leak", "calf"))
    S = summarise_frontier(front["rows"])
    S["integrity"], S["windows"], S["integrity_recheck"] = summarise_integrity(integ)
    S["absent_classes"] = integ["windows"]["absent_after_dedup"]
    S["contamination"] = summarise_leak(leak["rows"], front["rows"])
    S["calf"] = summarise_calf(calf["rows"]) if calf else None

    # the same analysis scored over the classes present in each held-out animal
    if all("macro_f1_present" in r for r in front["rows"] + leak["rows"]):
        P = summarise_frontier(front["rows"], "macro_f1_present")
        S["present"] = dict(teacher=P["teacher"]["macro_f1"], paired=P["paired"],
                            between_animal_sd=P["between_animal_sd"],
                            contamination=summarise_leak(leak["rows"], front["rows"],
                                                         "macro_f1_present"))
    grid = load("grid", res)
    S["grid"] = summarise_grid(grid["rows"]) if grid else None
    S["meta"] = run_meta(res)

    # further training seeds, each in results/seed-N/
    seeds = {"0": seed_summary(res, front)}
    for d in sorted(os.listdir(res)):
        m = re.fullmatch(r"seed-(\d+)", d)
        if m and os.path.isdir(os.path.join(res, d)):
            seeds[m.group(1)] = seed_summary(os.path.join(res, d))
    S["seeds"] = seeds

    json.dump(S, open(os.path.join(res, "summary.json"), "w"), indent=1)
    print(f"wrote summary.json in {res} ({len(seeds)} seed(s))")
    return S


if __name__ == "__main__":
    s = main(sys.argv[1] if len(sys.argv) > 1 else D.RESULTS)
    print(f"  teacher {s['teacher']['macro_f1']:.3f} at {s['teacher']['kb']:.1f} KB")
    print(f"  prune - scratch {s['paired']['prune']['mean']:+.4f} p={s['paired']['prune']['p_t']:.2g}")
    print(f"  contamination {s['contamination']['mean']:+.3f} p={s['contamination']['p_t']:.2g}")
