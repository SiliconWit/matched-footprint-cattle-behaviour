"""Fetch both datasets from Zenodo into data/, and check them against the manifest.

    python3 data/fetch_data.py

About 220 MB. See MANIFEST.md for the licences and how to cite both datasets.
A file that does not match its checksum differs from the copy used for the reported
results, usually because the dataset was revised at its source.
"""
import hashlib, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BEEF = "https://zenodo.org/api/records/5399259/files/%s/content"
CALF = "https://zenodo.org/api/records/13259482/files/AcTBeCalf.csv/content"
FILES = [(BEEF % ("cow%d.csv" % i), "cow%d.csv" % i) for i in range(1, 7)]
FILES += [(BEEF % "ReadMe.md", "cow_ReadMe.md"), (CALF, "AcTBeCalf.csv")]

EXPECTED = {
    "cow1.csv": "fcbf640fb31471b4708daa00ba9a0a3c",
    "cow2.csv": "0177e47e7459fd15dbcc9d4d125cdbec",
    "cow3.csv": "20a17215b327a9ba0dd6cce937f8dd8f",
    "cow4.csv": "ed0f0608260734367fc21d122ec13273",
    "cow5.csv": "02ceaf9c31e24596f9b9390953dcd4a6",
    "cow6.csv": "574866df8519efabadd0cedc6d56ca72",
    "cow_ReadMe.md": "3d0effc7b8542b28e9172cf6a4f074fb",
    "AcTBeCalf.csv": "59bd00564af64d92489485fa5a8a3960",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    for url, name in FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"  fetching {name} ...", flush=True)
            urllib.request.urlretrieve(url, path)
    bad = 0
    for name, want in EXPECTED.items():
        got = md5(os.path.join(HERE, name))
        ok = got == want
        bad += not ok
        print(f"  {name:16s} {got}  {'matches' if ok else 'DIFFERS from MANIFEST.md'}")
    if bad:
        print("\nSome files differ from the copies used for the reported results; "
              "results computed from them may differ.")


if __name__ == "__main__":
    main()
