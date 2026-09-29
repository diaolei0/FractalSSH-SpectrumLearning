# Validation basis behind v12

v12 adds only physics descriptors that were independently motivated before the full v12 training experiment.

## 816-family operator census

Using the formal v7 geometry set:

- `K1_OC,B0 = 0` for all 816 families.
- 117 families contain no BULK0 parent sector.
- Among the 699 families with BULK0, 594 have nonzero direct OC-BULK0 coupling at K2, 61 have K2 suppressed but K3 nonzero, and 44 remain closed through K3.
- 650 families have at least one two-step OC-BULK0 virtual path.  In 56 of those, the second-order path sum cancels to numerical zero.
- The median second-order interference efficiency among path-present families is `eta2 = 0.03225` (IQR approximately `0.02397-0.04629`).

## Exact perturbative checks

On formal exact-low-energy benchmarks, K2 improves the K1-only approximation in almost all tested families.  Adding the Hermitian K3 term further reduces the perturbative spectrum error in 48/49 tested families; the median error falls by roughly two orders of magnitude relative to K1-only in that benchmark set.

A formal K3-first family (`b7_F0003_N17`) shows K2 suppression but a large K3 correction.  A formal distance-2 cancellation family (`b5_F0024_N24`) has many nonzero K2 path contributions whose sum vanishes; its exact canonical OC-BULK0 effective block turns on at a higher power of `|t1|`.

## Relation to previous AI difficulty

On the fixed test split, eta2 is negatively associated with zero-sector prediction error:

- v9 zero-parentage: Spearman `rho=-0.4268`, `p=3.91e-4` among path-present test families.
- v10 gauge-aware zero-cluster MAE: `rho=-0.3833`, `p=1.62e-3`.
- The v10 relation remains significant after controlling rank-transformed `log(n_sites)`, `Df`, and BULK0 fraction (`rho_partial=-0.3522`, `p=4.02e-3`).
- Raw unsigned second-order path strength is not significantly correlated with the v10 zero-cluster error, supporting interference rather than path count alone as the relevant descriptor.

These are associations, not causal proof.  v12 therefore includes controlled ablations that hide K3 or eta2/path/order descriptors while keeping the split/backbone fixed.
