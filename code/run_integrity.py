"""Verify the files, check both datasets for cross-animal duplication, and
count what de-duplication and windowing leave behind.

No training here. Everything this writes is deterministic: a rerun on the same files
agrees exactly, not approximately.

    python3 run_integrity.py          # writes ../results/integrity.json, about a minute
"""
import json, os
import numpy as np
import data as D

FILES = D.BEEF_FILES + ["cow_ReadMe.md", D.CALF_FILE]


def count_windows(dataset, classes):
    """Windows per class overall and per animal, for a dict of recordings."""
    X, y, a = D.windows(dataset, classes)
    per_animal = {str(k): np.bincount(y[a == k], minlength=len(classes)).tolist()
                  for k in sorted(set(a.tolist()))}
    return dict(total=int(len(y)),
                per_class=dict(zip(classes, np.bincount(y, minlength=len(classes)).tolist())),
                per_animal=per_animal)


def main():
    out = {"checksums": {f: D.md5(os.path.join(D.DATA, f)) for f in FILES}}

    beef = D.load_beef()
    lengths = [len(beef[k][0]) for k in sorted(beef)]
    rep = D.duplicate_report(beef)
    out["beef"] = dict(
        animals=len(beef), samples_per_animal=lengths,
        all_animals_same_length=len(set(lengths)) == 1,
        pairs=rep["pairs"], pairs_sharing=rep["pairs_sharing"],
        shared=rep["shared"], seconds=rep["seconds"],
        distinct=rep["distinct"], distinct_seconds=rep["distinct_seconds"],
        mean_fraction=rep["mean_fraction"],
        varying_blocks={str(k): v for k, v in rep["varying_blocks"].items()},
        detail=[[int(a), int(b), int(c)] for a, b, c in rep["detail"]])

    clean = D.drop_cross_animal_duplicates(beef)
    raw_w = count_windows(beef, D.BEEF_CLASSES)
    clean_w = count_windows(clean, D.BEEF_CLASSES)
    out["windows"] = dict(classes=D.BEEF_CLASSES, raw=raw_w, clean=clean_w,
                          absent_after_dedup={k: [D.BEEF_CLASSES[i] for i, n in enumerate(v) if n == 0]
                                              for k, v in clean_w["per_animal"].items()})

    calf = D.load_calf()
    repc = D.duplicate_report(calf)
    out["calf"] = dict(animals=len(calf), pairs=repc["pairs"],
                       pairs_sharing=repc["pairs_sharing"], shared=repc["shared"],
                       seconds=repc["seconds"], distinct=repc["distinct"])
    Xc, yc, ac = D.windows(calf, D.CALF_CLASSES)
    out["calf_windows"] = dict(classes=D.CALF_CLASSES, total=int(len(yc)),
                               per_class=np.bincount(yc, minlength=len(D.CALF_CLASSES)).tolist())

    os.makedirs(D.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(D.RESULTS, "integrity.json"), "w"), indent=1)
    b = out["beef"]
    print(f"beef: {b['pairs_sharing']}/{b['pairs']} pairs share blocks; "
          f"{b['shared']} summed over pairs, {b['distinct']} distinct")
    print(f"calf: {out['calf']['pairs_sharing']}/{out['calf']['pairs']} pairs share blocks")
    print(f"beef windows: {raw_w['total']} as published, {clean_w['total']} de-duplicated")


if __name__ == "__main__":
    main()
