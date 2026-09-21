"""Turn results/summary.json into LaTeX macros, so no reported number is typed by hand.

Writes results/numbers.tex. Formats belong here rather than in the text: a p-value, a
count or a signed difference is printed the way a reader expects. The run date and
library versions are read from the metadata recorded beside the results, so they
describe the environment the experiments ran in rather than the one that built the
macros.

    python3 build_numbers.py [results_dir] [numbers.tex]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import routes as R
import train as TR

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "numbers.tex")
S = json.load(open(os.path.join(RES, "summary.json")))
L = []
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def cmd(n, v):
    L.append(r"\newcommand{\%s}{%s}" % (n, v))


def f3(x):
    return sg(x, 3, plus=False)


def sg(x, d=3, plus=True):
    """A number with a typographic minus, and a plus sign if `plus`."""
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0:
        return r"\ensuremath{-" + s + "}"
    return ("+" if plus else "") + s


def p(x):
    """p-values the way a reader expects to see them."""
    if x < 1e-4:
        e = int(np.floor(np.log10(x)))
        return r"%.1f\times 10^{%d}" % (x / 10 ** e, e)
    return f"{x:.4f}".rstrip("0")


def th(n):
    return f"{n:,}".replace(",", "{,}")


def word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


t = S["teacher"]
cmd("TeacherF", f3(t["macro_f1"])); cmd("TeacherKB", f"{t['kb']:.1f}")
cmd("TeacherParams", th(t["params"]))
cmd("MajorityF", f3(S["majority_f1"])); cmd("MajorityAcc", f3(S["majority_acc"]))
cmd("NAnimals", S["n_animals"])

sm = S["smallest"]
cmd("SmallKB", f"{sm['kb']:.2f}"); cmd("SmallParams", th(sm["params"]))
cmd("SmallPrune", f3(sm["prune"])); cmd("SmallDistil", f3(sm["distil"]))
cmd("SmallScratch", f3(sm["scratch"]))
cmd("ShrinkFactor", f"{t['kb']/sm['kb']:.0f}")
cmd("CapDefMax", sg(S["capacity_deficit_max"], plus=False))
cmd("CapDefMin", sg(S["capacity_deficit_min"], plus=False))
dec = S["decomposition"]
cmd("CapDefNonPos", word(sum(1 for d in dec if d["C"] <= 0)))
cmd("NWidths", word(len(dec)))

for a, tag in (("prune", "Prune"), ("distil", "Distil")):
    d = S["paired"][a]
    cmd(f"{tag}Gain", sg(d["mean"], 4))
    cmd(f"{tag}GainAbs", f"{abs(d['mean']):.4f}")
    cmd(f"{tag}P", p(d["p_t"])); cmd(f"{tag}PW", p(d["p_wilcoxon"]))
    cmd(f"{tag}Wins", d["wins"]); cmd(f"{tag}N", d["n"])
    cmd(f"{tag}UnpairedP", p(S["unpaired"][a]["p_t"]))
cmd("PairedN", S["paired"]["prune"]["n"])
cmd("BetweenAnimalSD", f3(S["between_animal_sd"]))
cmd("BetweenRouteSD", f3(S["between_route_sd"]))

g = S["prune_gain_by_kb"]
cmd("PruneGainBig", sg(g[0]["gain"])); cmd("PruneGainBigKB", f"{g[0]['kb']:.1f}")
best = max(g, key=lambda r: r["gain"])
cmd("PruneGainPeak", sg(best["gain"])); cmd("PruneGainPeakKB", f"{best['kb']:.1f}")
smallest = min(g, key=lambda r: r["kb"])
cmd("PruneGainSmall", sg(smallest["gain"])); cmd("PruneGainSmallKB", f"{smallest['kb']:.2f}")

c = S["contamination"]
cmd("ContamInflation", sg(c["mean"]))
cmd("ContamPublished", f3(c["as_published"])); cmd("ContamClean", f3(c["deduplicated"]))
cmd("ContamP", p(c["p_t"])); cmd("ContamWins", c["wins"]); cmd("ContamN", c["n"])
b = S["integrity"]["beef"]
cmd("BeefPairs", b["pairs"]); cmd("BeefShared", th(b["shared"]))
cmd("BeefSharedHours", f"{b['seconds']/3600:.1f}")
cmd("CalfPairs", S["integrity"]["calf"]["pairs"])
pm, pn = S["integrity"]["pair_max"], S["integrity"]["pair_min"]
cmd("PairMaxBlocks", th(pm["blocks"]))
cmd("PairMaxCows", f"{pm['cows'][0]} and {pm['cows'][1]}")
cmd("PairMinBlocks", pn["blocks"])
rc = S["integrity_recheck"]
cmd("BeefPairOcc", th(rc["pair_occurrences"]))
cmd("BeefPairOccHours", f"{rc['pair_occurrence_hours']:.1f}")
cmd("BeefDistinct", th(rc["distinct_blocks"]))
cmd("BeefDistinctHours", f"{rc['distinct_hours']:.1f}")
cmd("BeefFracDup", f"{100*rc['mean_fraction_of_animal_duplicated']:.0f}")
cmd("BeefSamples", th(rc["samples_per_animal"]))
cmd("BlockLen", rc["block"])

absent = {k: v for k, v in S["absent_classes"].items() if v}
cmd("NAbsentAnimals", word(len(absent)))
cmd("AbsentList", ", ".join(f"cow {int(k) + 1} lacks {'/'.join(v)}" for k, v in sorted(absent.items())))

P = S.get("present")
if P:
    cmd("TeacherFPresent", f3(P["teacher"]))
    for a, tag in (("prune", "Prune"), ("distil", "Distil")):
        cmd(f"{tag}GainPresent", sg(P["paired"][a]["mean"], 4))
        cmd(f"{tag}PPresent", p(P["paired"][a]["p_t"]))
    cmd("BetweenAnimalSDPresent", f3(P["between_animal_sd"]))
    cp = P["contamination"]
    cmd("ContamInflationPresent", sg(cp["mean"]))
    cmd("ContamPPresent", p(cp["p_t"])); cmd("ContamWinsPresent", cp["wins"])

G = S.get("grid")
if G:
    cmd("GridCells", G["n_cells"])
    cmd("GridKBs", " and ".join(f"{k:.2f}" for k in G["kb"]))
    cmd("GridBest", sg(G["best"])); cmd("GridBestKB", f"{G['best_cell']['kb']:.2f}")
    cmd("GridWorst", sg(G["worst"]))
    Ts = sorted({c["T"] for c in G["cells"]}); As = sorted({c["alpha"] for c in G["cells"]})
    cmd("GridTs", ", ".join(f"{v:g}" for v in Ts))
    cmd("GridAlphas", " and ".join(f"{v:g}" for v in As))

cf = S.get("calf")
if cf:
    cmd("CalfFolds", cf["n_folds"]); cmd("CalfMajority", f3(cf["majority_f1"]))
    for r, tag in (("prune", "Prune"), ("distil", "Distil"), ("scratch", "Scratch")):
        cmd(f"Calf{tag}F", f3(cf["by_route"][r]))
    cap = cf["capacity_deficit"]
    cmd("CalfCapSmall", sg(cap[0]["C"])); cmd("CalfCapSmallKB", f"{cap[0]['kb']:.1f}")
    cmd("CalfCapBig", sg(cap[-1]["C"])); cmd("CalfCapBigKB", f"{cap[-1]['kb']:.1f}")
    cmd("CalfTeacherF", f3(cf["teacher"]))
    for a, tag in (("prune", "Prune"), ("distil", "Distil")):
        d = cf["paired"][a]
        cmd(f"Calf{tag}Gain", sg(d["mean"], 4))
        cmd(f"Calf{tag}P", p(d["p_t"]))
    fr = {r["widths"]: r for r in cf["frontier"]}
    small = min(cf["frontier"], key=lambda r: r["kb"])
    cmd("CalfSmallBPrune", sg(small["prune"] - small["scratch"]))
    cmd("CalfSmallBDistil", sg(small["distil"] - small["scratch"]))
    cmd("CalfNWidths", word(len(fr)))

meta = S["meta"]
cmd("Epochs", meta["frontier"]["epochs"])
cmd("PruneEpochs", R.prune_epochs(meta["frontier"]["epochs"]))
cmd("LR", f"{TR.LR:g}")
cmd("PruneLR", f"{R.PRUNE_LR:g}")
cmd("DistilT", f"{meta['frontier']['T']:g}"); cmd("DistilAlpha", f"{meta['frontier']['alpha']:g}")
if "calf" in meta:
    m = meta["calf"]
    cmd("CalfCap", th(m["cap"])); cmd("CalfEpochs", m["epochs"])
    cmd("CalfWindows", th(m["windows"]))

w = S["windows"]
cmd("BeefWindows", th(w["windows_clean"]))
cmd("BeefWindowsRaw", th(w["windows_raw"]))
cmd("BeefDropPct", f"{100*(1-w['windows_clean']/w['windows_raw']):.0f}")
cmd("BeefClassCounts", "/".join(str(v) for v in w["per_class_clean"].values()))

# the spread over training seeds, where further seeds have been run
seeds = S.get("seeds", {})


def span(name, values, d=3, signed=True):
    if values:
        f = (lambda x: sg(x, d)) if signed else (lambda x: sg(x, d, plus=False))
        cmd(name + "Lo", f(min(values))); cmd(name + "Hi", f(max(values)))


front_seeds = sorted(k for k, v in seeds.items() if "paired" in v)
cmd("NSeeds", word(len(front_seeds)))
cmd("SeedList", ", ".join(front_seeds[:-1]) + (" and " if len(front_seeds) > 1 else "")
    + front_seeds[-1])
span("TeacherFSeed", [seeds[k]["teacher"] for k in seeds if "teacher" in seeds[k]], signed=False)
span("PruneGainSeed", [seeds[k]["paired"]["prune"]["mean"] for k in front_seeds], 4)
span("DistilGainSeed", [seeds[k]["paired"]["distil"]["mean"] for k in front_seeds], 4)
cmd("PrunePSeedMax", p(max(seeds[k]["paired"]["prune"]["p_t"] for k in front_seeds)))
cmd("DistilPSeedMin", p(min(seeds[k]["paired"]["distil"]["p_t"] for k in front_seeds)))
cmd("PruneSigSeeds", word(sum(seeds[k]["paired"]["prune"]["p_t"] < 0.05 and
                               seeds[k]["paired"]["prune"]["mean"] > 0 for k in front_seeds)))
cmd("DistilSigSeeds", word(sum(seeds[k]["paired"]["distil"]["p_t"] < 0.05 for k in front_seeds)))
cmd("PruneGainSeedMean", sg(float(np.mean([seeds[k]["paired"]["prune"]["mean"]
                                           for k in front_seeds])), 4))
worst = min(front_seeds, key=lambda k: seeds[k]["paired"]["prune"]["mean"])
cmd("PruneWorstSeed", worst)
cmd("PruneWorstSeedP", p(seeds[worst]["paired"]["prune"]["p_t"]))
span("CapDefMaxSeed", [seeds[k]["capacity_deficit_max"] for k in front_seeds], signed=False)
pres = [k for k in front_seeds if "paired_present" in seeds[k]]
span("PruneGainPresentSeed", [seeds[k]["paired_present"]["prune"]["mean"] for k in pres], 4)
leak_seeds = sorted(k for k in seeds if "contamination" in seeds[k])
cmd("NLeakSeeds", word(len(leak_seeds)))
span("ContamSeed", [seeds[k]["contamination"]["mean"] for k in leak_seeds])
span("ContamPresentSeed", [seeds[k]["contamination_present"]["mean"] for k in leak_seeds
                           if "contamination_present" in seeds[k]])
calf_seeds = sorted(k for k in seeds if "calf" in seeds[k])
cmd("NCalfSeeds", word(len(calf_seeds)))
span("CalfPruneGainSeed", [seeds[k]["calf"]["paired"]["prune"]["mean"] for k in calf_seeds], 4)
span("CalfDistilGainSeed", [seeds[k]["calf"]["paired"]["distil"]["mean"] for k in calf_seeds], 4)
if calf_seeds:
    cmd("CalfPrunePSeedMin", p(min(seeds[k]["calf"]["paired"]["prune"]["p_t"] for k in calf_seeds)))
    cmd("CalfDistilPSeedMin", p(min(seeds[k]["calf"]["paired"]["distil"]["p_t"] for k in calf_seeds)))
    span("CalfCapBigSeed", [seeds[k]["calf"]["capacity_deficit"][-1]["C"] for k in calf_seeds])
    worse = [k for k in calf_seeds if seeds[k]["calf"]["paired"]["prune"]["mean"] < 0
             and seeds[k]["calf"]["paired"]["prune"]["p_t"] < 0.05]
    cmd("CalfPruneWorseN", word(len(worse)))
    if worse:
        cmd("CalfPruneWorseP", p(min(seeds[k]["calf"]["paired"]["prune"]["p_t"] for k in worse)))
# pruning's margin at each footprint, as a range over seeds
gains = {}
for k in front_seeds:
    for r in seeds[k]["prune_gain_by_kb"]:
        gains.setdefault(round(r["kb"], 3), []).append(r["gain"])
kbs = sorted(gains, reverse=True)
if kbs:
    span("PruneGainBigSeed", gains[kbs[0]])
    span("PruneGainSmallSeed", gains[kbs[-1]])
    small3 = [x for kb in kbs[-3:] for x in gains[kb]]
    big4 = [x for kb in kbs[:-3] for x in gains[kb]]
    span("PruneGainLargeWidthsSeed", big4)
    span("PruneGainSmallWidthsSeed", small3)
    cmd("PruneGainSmallWidthsKB", f"{kbs[-1]:.2f} to {kbs[-3]:.2f}")
    span("PruneGainWidthSeed", [x for kb in kbs for x in gains[kb]])
    peaks = sorted({max(seeds[k]["prune_gain_by_kb"], key=lambda r: r["gain"])["kb"]
                    for k in front_seeds}, reverse=True)
    cmd("PruneGainPeakKBs", ", ".join(f"{v:.2f}" for v in peaks))

dates = sorted(m["date"] for m in meta.values() if "date" in m)
cmd("RunStamp", dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}")
cmd("NumpyVer", meta["frontier"]["numpy"])
cmd("TorchVer", meta["frontier"]["torch"])
cmd("PythonVer", meta["frontier"]["python"])

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write("% generated by build_numbers.py -- do not edit\n")
    fh.write("\n".join(L) + "\n")
print(f"wrote {len(L)} macros to {OUT}")
