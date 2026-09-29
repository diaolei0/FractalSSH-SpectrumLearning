# v12 paper logic

## Empirically established before the v12 training experiment

The formal 816-family geometry census supports the following statements:

1. Direct first-order OC-BULK0 weak coupling is absent for all 816 formal families: `K1_OC,B0 = 0`.
2. Among the 699 families containing BULK0, 594 open the direct OC-BULK0 effective channel at second order, 61 have `K2=0` but nonzero `K3`, and 44 remain closed through K3.
3. The component-graph distance provides a lower bound on perturbative order, but path interference can raise the first nonzero order beyond the graph-distance bound.
4. In the distance-2 exact-cancellation families, nonzero second-order path contributions can sum to zero; hence `K2=0` does not imply absence of virtual paths.
5. Across exact low-energy benchmarks, adding K2 and then K3 systematically improves the perturbative spectrum.  These calculations motivate the hierarchy descriptors; they are not labels extracted from the finite-coupling spectrum.
6. On the fixed v9/v10 test split, stronger second-order destructive interference is associated with larger zero-sector prediction error even after controlling for system size, fractal dimension and BULK0 fraction.  This is an association and must be tested causally by v12 ablation.

## Hypothesis tested by v12

The remaining zero-sector bottleneck is not only an eigenvector-gauge problem.  Geometry fixes allowed virtual paths, while interference determines which perturbative orders survive.  Therefore explicit K3 and interference/order descriptors should be most useful for the K2-suppressed families that v9/v10 found hardest.

The controlled comparison is:

```text
geometry only
-> K1/K2
-> K1/K2/K3
-> all matrices without hierarchy descriptors
-> full K1/K2/K3 + eta2/path/order hierarchy
```

with the same family split, seed, backbone and gauge-aware losses.

## Claims that require the full v12 run

Do not claim before the 816-family experiment that v12 improves zeroP or zeroCluster.  The full run must establish:

- whether `K1/K2/K3` improves over `K1/K2`;
- whether eta2/path/order information improves over matrix tokens alone;
- whether gains concentrate in K3-first / low-eta2 families;
- whether energy/projector/isolated-six performance is preserved;
- whether the v9-comparable zero-parentage metric and the gauge-invariant cluster metric tell the same or different story.

This separation keeps the paper logic falsifiable: the operator hierarchy is motivated by exact physics, while its predictive value is determined by controlled AI ablation.
