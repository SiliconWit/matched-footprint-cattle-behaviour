"""Loading, integrity checking and windowing for the two cattle accelerometry datasets.

Two datasets, both tri-axial neck accelerometry at 25 Hz:
  beef  Japanese Black Beef Cow Behavior Classification Dataset, six cows.
        Zenodo 10.5281/zenodo.5399259.
  calf  AcTBeCalf, thirty pre-weaned calves.
        Zenodo 10.5281/zenodo.13259482.

Licences and citations are in data/MANIFEST.md.

Leave-one-animal-out validation is only valid if no animal's signal also appears in
another animal's file. The integrity check below tests that, and run_integrity.py
applies it to both datasets before any model is trained.
"""
import collections, hashlib, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CATTLE_DATA", os.path.join(os.path.dirname(HERE), "data"))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

FS = 25                       # Hz, both datasets
BLOCK = 100                   # samples per hashed block, four seconds
BEEF_FILES = ["cow%d.csv" % i for i in range(1, 7)]
CALF_FILE = "AcTBeCalf.csv"
# the four beef behaviours with enough data to model; the rest are hundreds of samples
BEEF_CLASSES = ["RES", "RUS", "MOV", "GRZ"]
CALF_CLASSES = ["lying", "standing", "drinking_milk", "eating_concentrates",
                "grooming", "running", "walking"]


def md5(path, chunk=1 << 20):
    """MD5 of a file, read in chunks so a 180 MB file does not have to fit in memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_beef(path=DATA):
    """Return {animal index: (acc [N, 3] float32, labels [N] str)} for the six cows.

    Animal indices are 0 to 5 in file order, so cow 1 is index 0. Rows with no label
    are kept: they are part of the recording, and dropping them before the integrity
    check would change which blocks are compared.
    """
    out = {}
    for k, fn in enumerate(BEEF_FILES):
        acc, lab = [], []
        with open(os.path.join(path, fn), newline="") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split(",")
                if len(p) < 4:
                    continue
                acc.append((float(p[0]), float(p[1]), float(p[2])))
                lab.append(p[3])
        out[k] = (np.asarray(acc, np.float32), np.asarray(lab))
    return out


def load_calf(path=DATA, fn=CALF_FILE):
    """Return {calf id: (acc [N, 3] float32, labels [N] str)} for the thirty calves.

    Calf ids are strings in the file; code that needs an integer per animal maps them
    explicitly.
    """
    per = collections.defaultdict(lambda: ([], []))
    with open(os.path.join(path, fn)) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            per[p[1]][0].append((float(p[2]), float(p[3]), float(p[4])))
            per[p[1]][1].append(p[5])
    return {k: (np.asarray(a, np.float32), np.asarray(b)) for k, (a, b) in per.items()}


# --------------------------------------------------------------- integrity
def block_hashes(acc, block=BLOCK, min_distinct=None):
    """Hash consecutive, non-overlapping blocks of signal, skipping near-constant ones.

    Returns {hash: first start index}. A motionless animal really does emit the same
    reading many times, so a block with fewer than `min_distinct` distinct rows
    (default half the block) is skipped: a collision there is not evidence of
    anything. Readings are formatted to a fixed precision before hashing, so the hash
    depends on the values and not on how a float happens to print.
    """
    if min_distinct is None:
        min_distinct = block // 2
    rows = ["%.6g,%.6g,%.6g" % tuple(v) for v in acc]
    out = {}
    for j in range(0, len(rows) - block, block):
        blk = rows[j:j + block]
        if len(set(blk)) < min_distinct:
            continue
        out.setdefault(hashlib.md5("\n".join(blk).encode()).hexdigest(), j)
    return out


def duplicate_report(data, block=BLOCK, min_distinct=None):
    """Count blocks of varying signal that appear under more than one animal.

    Two counts are returned and they answer different questions. `shared` sums, over
    every pair of animals, the blocks that pair has in common, so a block present in
    k animals is counted k(k-1)/2 times; it says how pervasive the sharing is across
    pairs. `distinct` counts each shared block once; it says how much signal is
    duplicated.

    `mean_fraction` is the fraction of each animal's varying blocks that also appear
    in some other animal, averaged over animals.
    """
    sets = {aid: set(block_hashes(acc, block, min_distinct)) for aid, (acc, _) in data.items()}
    ids = sorted(sets)
    pairs, shared, detail, distinct = 0, 0, [], set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            common = sets[ids[i]] & sets[ids[j]]
            pairs += 1
            shared += len(common)
            distinct |= common
            if common:
                detail.append((ids[i], ids[j], len(common)))
    frac = [len(sets[a] & distinct) / max(len(sets[a]), 1) for a in ids]
    return dict(pairs=pairs, pairs_sharing=len(detail), shared=shared, detail=detail,
                seconds=shared * block / FS, distinct=len(distinct),
                distinct_seconds=len(distinct) * block / FS,
                mean_fraction=float(np.mean(frac)),
                varying_blocks={a: len(sets[a]) for a in ids})


def drop_cross_animal_duplicates(data, block=BLOCK):
    """Keep each duplicated block only in the first animal, in index order, that has it.

    This is a repair, not a recovery: it cannot say which animal the signal really
    came from, and it removes data unevenly across animals and classes. A class can
    vanish from an animal entirely; run_integrity.py reports what is left per animal
    and per class.
    """
    seen, out = {}, {}
    for aid in sorted(data):
        acc, lab = data[aid]
        rows = ["%.6g,%.6g,%.6g" % tuple(v) for v in acc]
        keep = np.ones(len(acc), bool)
        for j in range(0, len(rows) - block, block):
            blk = rows[j:j + block]
            if len(set(blk)) < block // 2:
                continue
            h = hashlib.md5("\n".join(blk).encode()).hexdigest()
            if h in seen and seen[h] != aid:
                keep[j:j + block] = False
            else:
                seen.setdefault(h, aid)
        out[aid] = (acc[keep], lab[keep])
    return out


# --------------------------------------------------------------- windowing
def windows(data, classes, win=50, stride=25):
    """Cut windows of `win` samples, keeping one only if every sample shares one label.

    Returns X [N, 3, win] float32, y [N] int64 (index into `classes`), and the animal
    id of each window. A window whose label is not in `classes` is dropped.

    The same stride is applied to every animal, so held-out windows overlap one
    another exactly as training windows do (by half, at the defaults). After
    de-duplication has removed stretches of signal, consecutive rows are no longer
    always consecutive in time, so a window can join the two sides of a removed block.
    """
    idx = {c: i for i, c in enumerate(classes)}
    X, y, a = [], [], []
    for aid, (acc, lab) in data.items():
        n = len(acc)
        for j in range(0, n - win, stride):
            seg = lab[j:j + win]
            c = seg[0]
            if c not in idx or not (seg == c).all():
                continue
            X.append(acc[j:j + win]); y.append(idx[c]); a.append(aid)
    return (np.asarray(X, np.float32).transpose(0, 2, 1),
            np.asarray(y, np.int64), np.asarray(a))
