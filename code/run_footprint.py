"""The footprint of every network on the ladder, before anything is trained.

Deterministic arithmetic; tests/test_footprint.py checks it by hand for small widths.

    python3 run_footprint.py          # writes ../results/footprint.json
"""
import models as M
import train as TR

LADDER = [(64, 128, 128), (48, 96, 96), (32, 64, 64), (24, 48, 48), (16, 32, 32),
          (12, 24, 24), (8, 16, 16), (6, 12, 12)]


def main():
    args = TR.cli()
    rows = []
    for n_classes, dataset in ((4, "beef"), (7, "calf")):
        for w in LADDER:
            fp = M.AccNet(w, n_classes).footprint_bytes()
            rows.append(dict(dataset=dataset, widths=str(w), **fp,
                             peak_activation=M.AccNet(w, n_classes).peak_activation()))
    TR.save("footprint", rows, dict(ladder=[str(w) for w in LADDER]), args.out)
    for r in rows[:len(LADDER)]:
        print(f"{r['widths']:>16}  params {r['params']:>7}  flash {r['flash']:>7} B"
              f"  RAM {r['ram']:>5} B  total {r['total'] / 1024:6.2f} KB")


if __name__ == "__main__":
    main()
