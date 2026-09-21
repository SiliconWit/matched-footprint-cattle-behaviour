"""The teacher alone, under leave-one-animal-out on the de-duplicated beef set,
scored beside the majority-class floor.

Every later comparison is made against this teacher and on these folds, so the
numbers here are the baseline level for the whole study.

    python3 run_teacher.py            # writes ../results/teacher.json
"""
from multiprocessing import Pool
import torch
import data as D
import models as M
import train as TR

torch.set_num_threads(1)
SEED = 0


def _fold(hold):
    X, y, a = D.windows(D.drop_cross_animal_duplicates(D.load_beef()), D.BEEF_CLASSES)
    fd = TR.fold_data(X, y, a, hold, 4)
    if fd is None:
        return []
    Xtr, ytr, Xte, yte = fd
    t = TR.train_teacher(Xtr, ytr, 4, seed=SEED)
    return [TR.row("teacher", TR.TEACHER_WIDTHS, hold, t.footprint_bytes(),
                   TR.evaluate(M.quantise_int8(t), Xte, yte, 4), seed=SEED)]


def main():
    global SEED
    args = TR.cli()
    SEED = args.seed
    with Pool(min(args.procs, 6)) as p:
        rows = sum(p.map(_fold, range(6)), [])
    TR.save("teacher", rows, dict(dataset="beef, de-duplicated", folds=6, seed=SEED,
                                  epochs=TR.EPOCHS), args.out)
    for r in rows:
        print(f"  cow {int(r['hold']) + 1}: macro F1 {r['macro_f1']:.3f}  majority {r['maj_f1']:.3f}")


if __name__ == "__main__":
    main()
