# Public-release audit

This package was prepared from the supplied `Fractal_SSH_COMPLETE_FROM_SCRATCH_R1.rar` source archive.

## Removed from the public package

- `__pycache__` bytecode directories.
- Local training log `one_click_operator_v12.log`.
- Duplicate embedded ZIP archive.
- Duplicate development copy of the ablation script.
- Stale source-package SHA256 list, because the public package was cleaned and path-only edits were applied.
- Local benchmark text logs.
- Legacy reference outputs whose hybrid nine-class metrics did not exactly match the current manuscript values.

## Path-only/publication edits

- `DATABASE_PATH.txt` was converted to a neutral template.
- Personal Windows paths were removed from documentation.
- The final hybrid decoder now defaults to repository-relative `work/database_v7`, `work/run_v12`, and `work/final_9class` locations; command-line overrides remain available.
- The hybrid decoder entry-name text was corrected to the actual filename.
- The release note now points to the final locked ablation workflow.

No model equations, physical definitions, target construction, network architecture, loss implementation, or numerical algorithms were changed.

## Final metric reconciliation

The uploaded final production archive was checked after the initial clean-up. Its `final_9class/summary.json` reports nine-class accuracy `0.9875070573113746` and Macro-F1 `0.9874186378508107`, exactly matching the rounded manuscript values `0.987507` and `0.987419`. Recalculation directly from `confusion_9class.csv` gives the same values over 332,988 test states.

A stale file named `REFERENCE_FINAL_SUMMARY.json` in the source archive reports slightly different isolated-branch results (`0.9875671` accuracy and `0.9874899` Macro-F1). It is not included in the public repository. The final `summary.json` and the confusion matrices are the authoritative paper-consistent records.

The production `test_summary_best_projector.json` also exactly reproduces the headline manuscript metrics: energy MAE `0.0356590296`, RMSE `0.0562154947`, pooled R² `0.9976948452`, state projector MAE `0.0119634591`, and degenerate-cluster projector-trace MAE `0.0161287317`.
