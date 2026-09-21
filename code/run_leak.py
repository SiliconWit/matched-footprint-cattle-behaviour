"""What the cross-animal duplication is worth.

The same fold, teacher and three routes at one width, run on the beef set as
published, without de-duplication. Matched against the de-duplicated frontier run
(same route, width, held-out animal and seed), the difference prices the leak.
Nothing else may differ between the two conditions, or the difference prices that
too.

    python3 run_leak.py               # writes ../results/leak.json
"""
from multiprocessing import Pool
import torch
import data as D
import routes as R
import train as TR

torch.set_num_threads(1)
WIDTH = (16, 32, 32)
SEED = 0


def _fold(hold):
    X, y, a = D.windows(D.load_beef(), D.BEEF_CLASSES)
    return R.run_fold(X, y, a, hold, [WIDTH], 4, seed=SEED)


def main():
    global SEED
    args = TR.cli()
    SEED = args.seed
    with Pool(min(args.procs, 6)) as p:
        rows = sum(p.map(_fold, range(6)), [])
    TR.save("leak", rows, dict(dataset="beef, as published", folds=6, seed=SEED,
                               epochs=TR.EPOCHS, widths=[str(WIDTH)]), args.out)


if __name__ == "__main__":
    main()
