# Reproducibility records

This directory contains small, paper-consistent records from the final production workflow. It is intended to make the reported split and headline metrics auditable without committing the full generated database or large training artifacts to Git.

## `model_run/`

- `family_split.json`: fixed 652/81/83 family-level training/validation/test split.
- `split_manifest.csv`: family metadata and split assignment for all 816 families.
- `scalers.json`: scalers used by the production v12 run.
- `test_summary_best_projector.json`: headline full-spectrum and projector metrics from the checkpoint selected by validation projector MAE (epoch 136).
- `run_metadata_public.json`: production software/hardware settings, seed, hyperparameters, and source-code hashes. Local absolute paths have been replaced by repository-relative paths only.

## `hybrid_9class/`

- `summary.json`: final paper-consistent hybrid nine-branch metrics.
- `confusion_9class.csv` and `confusion_zero3.csv`: state-level confusion matrices.
- `family_metrics.csv`: per-family hybrid-decoder metrics for the 83 test families.
- `geometry_manifest.csv`: family-level geometry manifest used for reproducibility checks.

The confusion matrix independently reproduces the reported nine-class accuracy of 0.9875070573 and Macro-F1 of 0.9874186379 over 332,988 test states.

## `acoustic_vicsek/`

Small processed records for the standard G(2) Vicsek test family (`b3_F0001_N05`) are included. The raw acoustic measurement files should be deposited with the full research-data release rather than in the source-code repository.

The processed records reproduce the reported absent-BULK predicted mass fraction of 0.0017074856 (0.17074856%).

## Excluded generated artifacts

The full per-family state database, model checkpoints, final per-family prediction arrays, raw acoustic measurement files, and operator cache are not required in the Git source tree. They should be placed in the DOI-bearing research-data archive. The operator cache is deterministic and can also be regenerated from the geometry database and released code.
