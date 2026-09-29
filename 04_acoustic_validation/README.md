# Acoustic validation data and analysis

This folder contains the public analysis scripts and compact processed data used for the acoustic validation of the standard G(2) Vicsek sample.

## OC spatial-support analysis

`analysis/analyze_OC_from_raw_and_plot_S17.py` implements the common-reference H1 processing chain used for the OC spatial-support validation. The released processed tables are:

- `processed/OC_raw_file_manifest.csv`
- `processed/OC_site_response_metrics.csv`
- `processed/OC_pair_metrics_from_raw.csv`

The 23 common OC/control pairs give 16/23 positive branch-window contrasts with a median of +6.746 dB, 17/23 positive contrasts at 5235.9375 Hz with a median of +4.601 dB, and 22/23 sign agreement. Pair 33/34 is excluded from the common quantitative statistics because it used a different multichannel acquisition configuration.

## ED and IC local-phase analysis

`analysis/analyze_ED_from_raw.py` and `analysis/analyze_IC_from_raw.py` implement the synchronous four-channel phase analysis. The per-measurement and repeated-measurement summaries are provided under `processed/`.

The public repository contains compact processed data. Large raw time-domain recordings are not Git-tracked. If raw recordings are deposited separately, place them under the relative `raw/` folders or pass the appropriate data directory to the analysis script where supported.
