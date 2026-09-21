# Data

Two public datasets, both tri-axial neck accelerometry at 25 Hz. Neither
is committed to this repository; `python3 data/fetch_data.py` downloads them from Zenodo
into this directory and checks them against the table below.

| Dataset | Animals | Source | DOI |
|---|---|---|---|
| Japanese Black Beef Cow Behavior Classification Dataset (Ito, Takeda, Tokgoz and Minati, 2021) | 6 cows | Zenodo record 5399259 | [10.5281/zenodo.5399259](https://doi.org/10.5281/zenodo.5399259) |
| AcTBeCalf: accelerometer-based multivariate time-series dataset for calf behaviour classification (Dissanayake, McPherson, Allyndrée and Kennedy, 2024) | 30 calves | Zenodo record 13259482 | [10.5281/zenodo.13259482](https://doi.org/10.5281/zenodo.13259482) |

| File | Dataset | Bytes | MD5 |
|---|---|---|---|
| `cow1.csv` | beef | 6,814,084 | `fcbf640fb31471b4708daa00ba9a0a3c` |
| `cow2.csv` | beef | 7,014,154 | `0177e47e7459fd15dbcc9d4d125cdbec` |
| `cow3.csv` | beef | 7,057,143 | `20a17215b327a9ba0dd6cce937f8dd8f` |
| `cow4.csv` | beef | 6,883,569 | `ed0f0608260734367fc21d122ec13273` |
| `cow5.csv` | beef | 6,871,453 | `02ceaf9c31e24596f9b9390953dcd4a6` |
| `cow6.csv` | beef | 7,030,941 | `574866df8519efabadd0cedc6d56ca72` |
| `cow_ReadMe.md` | beef (published as `ReadMe.md`) | 3,055 | `3d0effc7b8542b28e9172cf6a4f074fb` |
| `AcTBeCalf.csv` | calf | 177,608,717 | `59bd00564af64d92489485fa5a8a3960` |

The checksums are of the copies used for every reported result. If either dataset is
revised at its source, a fresh download will not match, and results computed from it may
differ.

Run the integrity check (`code/run_integrity.py`) before using either dataset for
anything else.

## Licences

The calf record is CC BY 4.0. The beef record's licence field reads CC BY 4.0, but its
description states CC BY-NC-ND 4.0. This repository treats it as the stricter one:
it fetches both datasets and redistributes neither, in original or modified form.

