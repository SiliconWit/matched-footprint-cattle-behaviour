"""The matched-footprint frontier on the de-duplicated beef set.

Teacher, then distillation, structured pruning and the same-size control at every
width on the ladder, for each of the six held-out animals. Every route sees the same
folds, which is what makes a per-animal paired comparison possible later.

    python3 run_frontier.py           # writes ../results/frontier.json
"""
from multiprocessing import Pool
import torch
import data as D
import routes as R
import train as TR

torch.set_num_threads(1)
WIDTHS = [(48, 96, 96), (32, 64, 64), (24, 48, 48), (16, 32, 32),
          (12, 24, 24), (8, 16, 16), (6, 12, 12)]
SEED = 0


def _fold(hold):
    X, y, a = D.windows(D.drop_cross_animal_duplicates(D.load_beef()), D.BEEF_CLASSES)
    return R.run_fold(X, y, a, hold, WIDTHS, 4, seed=SEED)


def main():
    global SEED
    args = TR.cli()
    SEED = args.seed
    with Pool(min(args.procs, 6)) as p:
        rows = sum(p.map(_fold, range(6)), [])
    TR.save("frontier", rows, dict(dataset="beef, de-duplicated", folds=6, seed=SEED,
                                   epochs=TR.EPOCHS, widths=[str(w) for w in WIDTHS],
                                   T=4.0, alpha=0.5), args.out)


if __name__ == "__main__":
    main()
