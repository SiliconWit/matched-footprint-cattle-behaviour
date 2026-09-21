"""The figures.

Sized for a 6.5 in text block so nothing is scaled down in the PDF.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAVY, GOLD, SLATE, RED, GREEN = "#0F284D", "#B0892C", "#5A6473", "#8C2F1F", "#3E6B4F"
W, H2, H3 = 6.5, 2.25, 2.15
ROUTE = {"prune": ("structured pruning", NAVY, "o", "-"),
         "distil": ("distillation", GOLD, "s", "--"),
         "scratch": ("same size, hard labels", SLATE, "^", ":")}

STYLE = {
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "figure.dpi": 200,
}


def use_style():
    plt.rcParams.update(STYLE)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_integrity(path, detail, recheck, contam):
    """What the duplication looks like, and what leaving it in costs."""
    fig, ax = plt.subplots(1, 3, figsize=(W, H3))

    M = np.full((6, 6), np.nan)
    for a, b, n in detail:
        M[a, b] = M[b, a] = n
    im = ax[0].imshow(M, cmap="cividis", vmin=0)
    ax[0].set_xticks(range(6)); ax[0].set_yticks(range(6))
    ax[0].set_xticklabels([f"c{i+1}" for i in range(6)])
    ax[0].set_yticklabels([f"c{i+1}" for i in range(6)])
    ax[0].set_title("shared blocks, every pair", pad=4)
    cb = fig.colorbar(im, ax=ax[0], pad=0.03, fraction=0.046, aspect=18)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)

    # pair-occurrences against distinct blocks
    lab = ["summed over\npairs", "distinct\nblocks"]
    val = [recheck["pair_occurrence_hours"], recheck["distinct_hours"]]
    ax[1].bar([0, 1], val, 0.55, color=[SLATE, NAVY], edgecolor="none")
    for i, v in enumerate(val):
        ax[1].text(i, v, f"{v:.1f} h", ha="center", va="bottom", fontsize=7)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(lab, fontsize=7)
    ax[1].set_ylabel("duplicated signal (hours)")
    ax[1].set_title("two ways to count it", pad=4)
    lo, hi = ax[1].get_ylim(); ax[1].set_ylim(lo, hi * 1.22)

    ax[2].bar([0, 1], [contam["as_published"], contam["deduplicated"]], 0.55,
              color=[RED, NAVY], edgecolor="none")
    for i, v in enumerate([contam["as_published"], contam["deduplicated"]]):
        ax[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["as published", "de-duplicated"], fontsize=7)
    ax[2].set_ylabel("macro F1")
    ax[2].set_title(f"cost of leaving it in: {contam['mean']:+.3f}", pad=4)
    ax[2].set_ylim(0, 1.0)
    fig.tight_layout(w_pad=1.3)
    _save(fig, path)


def fig_frontier(path, beef, calf, teacher_beef, teacher_calf, maj_beef, maj_calf):
    """Accuracy against footprint, on both datasets."""
    fig, ax = plt.subplots(1, 2, figsize=(W, H2), sharey=True)
    for i, (rows, tea, maj, name) in enumerate(
            [(beef, teacher_beef, maj_beef, f"beef, 6 cows"),
             (calf, teacher_calf, maj_calf, "calf, 30 animals")]):
        kb = [r["kb"] for r in rows if "scratch" in r]
        for key, (lab, col, mk, ls) in ROUTE.items():
            v = [r[key] for r in rows if key in r]
            if v:
                ax[i].semilogx(kb, v, color=col, marker=mk, ms=3.4, ls=ls, label=lab)
        ax[i].axhline(tea, color=RED, ls="-", lw=0.9)
        ax[i].text(max(kb), tea, " teacher", fontsize=6.6, color=RED, va="bottom", ha="right")
        ax[i].axhline(maj, color=SLATE, ls=":", lw=0.9)
        ax[i].text(min(kb), maj, " majority class", fontsize=6.6, color=SLATE, va="bottom")
        ax[i].set_xlabel("footprint (KB, int8)")
        ax[i].set_title(name, pad=4)
        if i == 0:
            ax[i].set_ylabel("macro F1")
    ax[0].set_ylim(0, 0.92)
    leg = ax[0].legend(loc="lower right", handlelength=1.6, borderaxespad=0.3,
                       fontsize=6.8, frameon=True, framealpha=0.95)
    leg.get_frame().set_edgecolor("none")
    fig.tight_layout(w_pad=1.2)
    _save(fig, path)


def fig_paired(path, per_animal, gain_by_kb, calf_gain, between_animal):
    """The effect is small, the animal effect is not, and pairing is what separates them."""
    fig, ax = plt.subplots(1, 3, figsize=(W, H3))

    for i, (key, lab) in enumerate([("prune", "pruning"), ("distil", "distillation")]):
        d = np.asarray(per_animal[key])
        ax[0].scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.11, 0.11, len(d)),
                      d, s=11, color=ROUTE[key][1], alpha=0.8)
        ax[0].plot([i - 0.24, i + 0.24], [d.mean()] * 2, color=ROUTE[key][1], lw=1.6)
    ax[0].axhline(0, color=SLATE, lw=0.8)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["pruning", "distillation"], fontsize=7.5)
    ax[0].set_ylabel("macro F1 minus same-size control")
    ax[0].set_title("paired differences, beef", pad=4)

    kb = [r["kb"] for r in gain_by_kb]
    gn = [r["gain"] for r in gain_by_kb]
    ax[1].semilogx(kb, gn, color=NAVY, marker="o", ms=3.4)
    ax[1].axhline(0, color=SLATE, lw=0.8)
    ax[1].set_xlabel("footprint (KB, int8)")
    ax[1].set_ylabel("pruning advantage")
    ax[1].set_title("pruning margin by footprint", pad=4)

    ax[2].bar([0, 1], [between_animal, abs(np.mean(per_animal["prune"]))], 0.55,
              color=[SLATE, NAVY], edgecolor="none")
    for i, v in enumerate([between_animal, abs(np.mean(per_animal["prune"]))]):
        ax[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["between\nanimals", "between\nroutes"], fontsize=7)
    ax[2].set_ylabel("standard deviation / effect")
    ax[2].set_title("why pairing is required", pad=4)
    lo, hi = ax[2].get_ylim(); ax[2].set_ylim(lo, hi * 1.25)
    fig.tight_layout(w_pad=1.4)
    _save(fig, path)


def fig_decomposition(path, beef_dec, calf_dec):
    """Capacity deficit against transfer benefit, on both datasets."""
    fig, ax = plt.subplots(1, 2, figsize=(W, H2), sharey=True)
    for i, (dec, name) in enumerate([(beef_dec, "beef, 6 cows"),
                                     (calf_dec, "calf, 30 animals")]):
        kb = [d["kb"] for d in dec]
        ax[i].semilogx(kb, [d["C"] for d in dec], color=RED, marker="o", ms=3.4,
                       label=r"capacity deficit $C(F)$")
        for key, lab, col, mk in (("B_prune", "pruning", NAVY, "s"),
                                  ("B_distil", "distillation", GOLD, "^")):
            if key in dec[0]:
                ax[i].semilogx(kb, [d[key] for d in dec], color=col, marker=mk, ms=3.4,
                               ls="--", label=r"benefit $B_r(F)$, " + lab)
        ax[i].axhline(0, color=SLATE, lw=0.8)
        ax[i].set_xlabel("footprint (KB, int8)")
        ax[i].set_title(name, pad=4)
    ax[0].set_ylabel("macro F1")
    ax[0].legend(frameon=False, loc="upper right", handlelength=1.6,
                 borderaxespad=0.3, fontsize=6.6)
    fig.tight_layout(w_pad=1.2)
    _save(fig, path)
