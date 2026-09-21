"""The board. Export models for the vendor tool, then summarise what it measured.

    python3 run_board.py --export     # trains three widths on all six cows, writes ../results/onnx/
    python3 run_board.py              # reads ../results/board_measurements.json, writes ../results/board.json

The measurements file is written by hand from the tool's "validate on target" report,
one entry per model:

    [{"widths": "(8, 16, 16)", "route": "prune", "tool_flash_bytes": 0,
      "tool_ram_bytes": 0, "latency_ms": 0.0, "current_ma": 0.0, "voltage_v": 3.3}]

Record the clock frequency, the compiler optimisation level and the tool version
beside it; latency depends on all three.
"""
import json, os, sys
import torch
import board as B
import data as D
import models as M
import routes as R
import train as TR

torch.set_num_threads(4)
WIDTHS = [(32, 64, 64), (8, 16, 16), (6, 12, 12)]


def export():
    X, y, a = D.windows(D.drop_cross_animal_duplicates(D.load_beef()), D.BEEF_CLASSES)
    Xs, _ = TR.standardise(X, X)
    out = os.path.join(D.RESULTS, "onnx")
    os.makedirs(out, exist_ok=True)
    teacher = TR.train_teacher(Xs, y, 4)
    B.export_onnx(teacher, os.path.join(out, "teacher.onnx"))
    for w in WIDTHS:
        p = R.transfer_surviving_channels(teacher, M.AccNet(w, 4))
        TR.fit(p, Xs, y, epochs=R.prune_epochs(TR.EPOCHS), lr=R.PRUNE_LR,
               class_weight=TR.class_weights(y, 4))
        B.export_onnx(p, os.path.join(out, "prune_%d_%d_%d.onnx" % w))
        s = TR.fit(M.AccNet(w, 4), Xs, y, class_weight=TR.class_weights(y, 4))
        B.export_onnx(s, os.path.join(out, "scratch_%d_%d_%d.onnx" % w))
    print(f"exported to {out}")


def main():
    if "--export" in sys.argv:
        return export()
    src = os.path.join(D.RESULTS, "board_measurements.json")
    if not os.path.exists(src):
        sys.exit(f"no measurements yet: write {src} from the tool's report")
    out = B.summarise(json.load(open(src)))
    json.dump(out, open(os.path.join(D.RESULTS, "board.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_model"}, indent=1))


if __name__ == "__main__":
    main()
