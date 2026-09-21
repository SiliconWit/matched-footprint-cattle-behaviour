# Yardstick: matched-footprint cattle behaviour classification

Knowledge distillation and structured pruning for an on-collar cattle behaviour
classifier, compared at matched footprint against a network of the same size trained
from hard labels. Both datasets are checked for cross-animal duplication before any model
is trained.

## Requirements

Python 3.10 or later and a CPU; no GPU is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
```

## Data

Two public accelerometer datasets on Zenodo. `data/fetch_data.py` downloads them into
`data/` and checks them against the checksums in `data/MANIFEST.md`, which also gives
their sources, licences and citations.

## Running

From the repository root:

```bash
python3 data/fetch_data.py
cd code
python3 run_integrity.py     # cross-animal duplication check and window counts
python3 run_footprint.py     # flash and RAM of every network width
python3 run_frontier.py      # three routes at seven widths, leave-one-animal-out
python3 run_leak.py          # the same runs on the data with duplication left in
python3 run_grid.py          # distillation temperature and weight grid
python3 run_calf.py          # replication on the calf dataset
python3 analyse.py
python3 build_numbers.py
python3 make_figs.py
```

The full sequence takes about two hours on an eight-core CPU. Scripts that train take
`--seed` and `--out`; running `run_frontier.py`, `run_leak.py` and `run_calf.py` with
`--seed N --out ../results/seed-N` and then the last three commands adds the spread over
seeds. `run_frontier.py --equal-schedule --seed N --out ../results/equal-seed-N` fine-tunes
the pruned network for the same epochs and at the same learning rate as the other two
routes; `analyse.py` compares each such run with the standard one at the same seed.
Every run script takes `--procs` to set the number of worker processes. `run_board.py`
exports models for measurement on a microcontroller and needs the
`onnx` and `onnxscript` packages.

## Outputs

Everything is written to `results/`: one JSON file per experiment, `summary.json`,
`numbers.tex` (every reported value as a LaTeX macro), `frontier_table.tex`,
`seed_table.tex` (per-seed paired differences) and the figures in `results/figs/`.

## Citation

See `CITATION.cff`.

## License

MIT
