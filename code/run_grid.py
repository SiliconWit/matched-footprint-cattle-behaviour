"""Was distillation given a fair chance?

A grid over temperature and the soft-loss weight at two widths, each cell scored
against the same-size control on the same fold. A null result for distillation at
one setting says little until the neighbouring settings have been tried.

    python3 run_grid.py               # writes ../results/grid.json
"""
from multiprocessing import Pool
import torch
import data as D
import routes as R
import train as TR

torch.set_num_threads(1)
WIDTHS = [(16, 32, 32), (8, 16, 16)]
GRID = [(T, a) for T in (1.0, 2.0, 4.0, 8.0) for a in (0.5, 0.9)]
SEED = 0


def _fold(hold):
    X, y, a = D.windows(D.drop_cross_animal_duplicates(D.load_beef()), D.BEEF_CLASSES)
    return R.run_fold(X, y, a, hold, WIDTHS, 4, seed=SEED, grid=GRID)


def main():
    global SEED
    args = TR.cli()
    SEED = args.seed
    with Pool(min(args.procs, 6)) as p:
        rows = sum(p.map(_fold, range(6)), [])
    TR.save("grid", rows, dict(dataset="beef, de-duplicated", folds=6, seed=SEED,
                               epochs=TR.EPOCHS, widths=[str(w) for w in WIDTHS],
                               grid=GRID), args.out)


if __name__ == "__main__":
    main()
