# FractalSSH-SpectrumLearning

Code accompanying the manuscript:

**Physics-Constrained Operator-Enhanced Multiscale Convolution–Transformer Learning for Variable-Length Full-Spectrum Prediction in 2D Fractal SSH Systems with Acoustic Validation**

This repository contains the database-construction code, the operator-enhanced variable-length full-spectrum learning model, the controlled ablation workflow, and the final hybrid nine-branch decoder used in the study.

## Repository structure

```text
01_database_v7/   Physics construction, 816-family database generation, and numerical validation
02_ai_v12/        Operator-enhanced learning model, training, testing, and controlled ablations
03_final_hybrid/  Final hybrid nine-branch decoder
04_acoustic_validation/ Public acoustic-analysis scripts and compact processed measurements
reproducibility/   Fixed split, final paper-consistent metrics, and small Vicsek validation records
reference/        Reserved for optional release-specific reference material
work/             Local data and run outputs; ignored by Git
```

## Fixed physical setting

The released workflow uses generation `G=2`, finite couplings `t1=-0.1` and `t2=-1.0`, and the strong-coupling reference limit `t1=0`, `t2=-1.0`. The nine reference branches are `OC`, `ED-I`, `ED-II`, `IC-I`, `IC-II`, `IC-III`, `BULK-L`, `BULK-0`, and `BULK-U`.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for the full 816-family training run.

## 1. Build and validate the physics database

```bash
cd 01_database_v7
python validate_physics_benchmarks_v7.py
python run_full_pipeline_canonical_v7_ONECLICK.py
```

The database builder produces `geometry_manifest.csv`, `masks/*.npz`, `states/*.npz`, and database metadata. The complete database is intentionally not stored in Git.

## 2. Run the v12 operator-enhanced model

Set the database path either in `02_ai_v12/DATABASE_PATH.txt` or through the environment variable `V7_DB`, then run:

```bash
cd 02_ai_v12
python SELF_TEST_V12.py
python RUN_THIS_OPERATOR_AI_v12.py
```

The production seed used by the v12 workflow is `20260912`. Training, validation, and test partitions are separated at the family level.

## 3. Controlled ablations

The final locked ablation workflow uses the existing full-model run as configuration E and trains A, B, C, D, F, and G with the same family split and scalers.

```bash
cd 02_ai_v12
python RUN_V12_ABLATIONS_FINAL_LOCKED.py --dry-run
python RUN_V12_ABLATIONS_FINAL_LOCKED.py
python summarize_v12_ablation_FINAL_LOCKED.py
```

## 4. Final hybrid nine-branch decoding

By default, the decoder expects the following local layout:

```text
work/
  database_v7/
  run_v12/
  final_9class/
```

Run:

```bash
python 03_final_hybrid/RUN_FINAL_9CLASS.py
```

Custom locations can be supplied with `--db-root`, `--v12-run-dir`, `--pred-dir`, and `--outdir`.

## Reproducing the paper exactly

The repository includes the fixed production `family_split.json`, `split_manifest.csv`, scalers, the epoch-136 test summary selected by validation projector MAE, the paper-consistent hybrid nine-branch confusion matrices and summary, and compact processed records for the standard G(2) Vicsek validation case under `reproducibility/`.

The production split contains 652 training, 81 validation, and 83 test families with no overlap. The included hybrid confusion matrix independently reproduces the manuscript values of 0.987507 nine-class accuracy and 0.987419 Macro-F1 over 332,988 test states.

The full generated per-state database and final checkpoint are distributed through a DOI-bearing research-data repository rather than committed directly to GitHub. Compact processed acoustic data are included under `04_acoustic_validation/`; large raw time-domain recordings may be deposited separately. See `DATA_PUBLICATION.md`.

## Reproducibility checks performed for this public package

- All retained Python source files compile successfully.
- `01_database_v7/validate_physics_benchmarks_v7.py` passes.
- `02_ai_v12/SELF_TEST_V12.py` passes its two-process operator and model checks.
- Local absolute paths, Python bytecode caches, logs, duplicate archives, and duplicate development scripts have been removed from the clean package.

## License

The source code in this repository is released under the MIT License. See `LICENSE`. Research data deposited separately are licensed under CC BY 4.0 unless otherwise noted.

## Author and citation

Software author: **Lei Diao**.

This repository is released under the MIT License. A `CITATION.cff` file is included so GitHub can expose the recommended software citation. After the associated Zenodo data record and journal article receive DOIs, add those persistent identifiers here.

