"""Does the ordering travel? The frontier re-derived on the calf dataset.

Thirty calves, seven classes, a different sensor and population. Calves are grouped
into five folds so that every calf is held out whole, and each class is capped to
keep the run to hours rather than days.

Three choices here change what the result means: calves are assigned to folds by
their sorted identifiers taken in turn; the class cap is applied to every window
before the folds are cut, so held-out folds are thinned as well as training ones; and
the replication uses three widths and half the epochs of the beef study.

    python3 run_calf.py               # writes ../results/calf.json
"""
from multiprocessing import Pool
import numpy as np
import torch
import data as D
import routes as R
import train as TR

torch.set_num_threads(1)
WIDTHS = [(32, 64, 64), (16, 32, 32), (8, 16, 16)]
CAP = 8000          # windows per class
EPOCHS = 15
FOLDS = 5
SEED = 0


def cap_classes(y, cap=CAP, cap_seed=0):
    """Indices, in ascending order, keeping at most `cap` windows of every class.

    Each class is drawn without replacement from one generator seeded with
    `cap_seed`, independent of the training seed, so the thinned set does not move
    when the training seed does.
    """
    rng = np.random.default_rng(cap_seed)
    keep = [rng.choice(np.flatnonzero(y == c), min(int((y == c).sum()), cap), replace=False)
            for c in np.unique(y)]
    return np.sort(np.concatenate(keep))


def assign_folds(animal, k=FOLDS):
    """Fold index per window: the sorted distinct animal ids taken in turn modulo k.

    Every window of one animal lands in the same fold, so a held-out fold never
    shares an animal with training. Ids are strings, so they are ranked, not
    reduced modulo k directly.
    """
    ids = np.unique(animal)
    return np.searchsorted(ids, animal) % k


def prepare(cap=CAP):
    """Window every calf, cap each class, assign calves to folds.

    Returns X, y, fold index per window, and the class counts after capping. Here the
    cap is applied to every window before the folds are cut, so held-out folds are
    thinned as well as training ones.
    """
    X, y, a = D.windows(D.load_calf(), D.CALF_CLASSES)
    keep = cap_classes(y, cap)
    X, y, a = X[keep], y[keep], a[keep]
    return X, y, assign_folds(a), np.bincount(y).tolist()


def _fold(g):
    X, y, fold, _ = prepare()
    return R.run_fold(X, y, fold, g, WIDTHS, len(D.CALF_CLASSES), epochs=EPOCHS, seed=SEED)


def main():
    global SEED
    args = TR.cli()
    SEED = args.seed
    _, _, fold, counts = prepare()
    with Pool(min(args.procs, FOLDS)) as p:
        rows = sum(p.map(_fold, range(FOLDS)), [])
    TR.save("calf", rows, dict(dataset="calf, capped", folds=FOLDS, seed=SEED,
                               epochs=EPOCHS, cap=CAP, windows=int(sum(counts)),
                               class_counts=counts, widths=[str(w) for w in WIDTHS],
                               windows_per_fold=np.bincount(fold).tolist()), args.out)


if __name__ == "__main__":
    main()
