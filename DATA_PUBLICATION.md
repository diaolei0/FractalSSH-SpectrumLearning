# Data publication plan

The source code can be hosted on GitHub, while the generated research data should be deposited separately in a repository that supports large files and persistent identifiers.

## Minimum paper-reproduction data

The public data package should contain:

1. `geometry_manifest.csv` for all 816 families.
2. The corresponding `masks/*.npz` geometry files.
3. The per-family `states/*.npz` files needed by the production loader. At minimum these must retain the fields used by the released model and evaluation pipeline, including `eigenvalues`, `q_state`, `branch_scores`, `branch_index`, `cluster_id`, `cluster_size`, `is_degenerate`, `canonicalized`, `zero_candidate_mask`, and `ref_dims` when present.
4. The exact production split files, especially `family_split.json` and `split_manifest.csv`.
5. Final test predictions or the files required to regenerate them from the released checkpoint.
6. Final paper-consistent hybrid-decoder outputs: `summary.json`, `family_metrics.csv`, `confusion_9class.csv`, and `confusion_zero3.csv`.
7. Acoustic measurement data and branch-frequency windows used for the experimental figures.

## Files that should not be published as research data unless specifically needed

- Python bytecode caches and temporary files.
- Optimizer states and every intermediate checkpoint.
- Local logs from failed or exploratory runs.
- Duplicate archives.
- Reconstructible dense Hamiltonian matrices when geometry and construction code are already provided.
- Development-only caches that can be regenerated from the public database and code.

## Recommended split between platforms

- GitHub: source code, documentation, small reference outputs, environment files.
- Zenodo or another DOI-capable research-data repository: the complete generated database, fixed split files, final checkpoints if desired, and acoustic measurements.

A DOI-linked GitHub release can be archived after the final paper-consistent code and reference outputs have been verified.


## Verified production records already included in the Git repository

The cleaned Git package now contains the fixed 652/81/83 family split, the epoch-136 best-projector test summary, the final paper-consistent nine-class confusion matrices and metrics, and compact processed records for the standard G(2) Vicsek test family.

The supplied operator cache contains one deterministic `.npz` record for each of the 816 families. Because these records can be regenerated from the public geometry database and released operator code, the cache is optional for long-term data deposition rather than a required Git-tracked artifact.

## License choice for this release

- GitHub source code: MIT License.
- Zenodo research data: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Published article: governed by the final journal publishing agreement.

## Acoustic release note

The GitHub repository contains the compact processed OC/ED/IC tables and analysis scripts needed to inspect the reported acoustic statistics. Large raw time-domain recordings are not required in the Git repository. If full raw-to-figure reproduction is desired, deposit the raw recordings as additional files in the Zenodo record or as a linked companion record.
