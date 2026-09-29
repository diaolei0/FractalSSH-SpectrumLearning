# METHODS v12 — Operator hierarchy and gauge-aware full-spectrum learning

## Parent subspaces

At `t1=0`, strong-bond connected components of sizes 1, 3 and 4 generate the zero-parent sectors OC, IC-II and BULK-0, respectively.  The corresponding orthonormal projectors are denoted by `P_OC`, `P_ICII`, and `P_B0`, and `P` denotes their direct sum.  Nonzero local strong-component modes define the reduced resolvent

\[
R = Q H_0^{-1} Q.
\]

The weak-bond operator `V` is defined with unit amplitude; the physical coupling amplitude is inserted separately through `t1=-0.1`.

## Effective-operator hierarchy

v12 constructs

\[
K_1=PVP,
\]

\[
K_2=-PVRVP,
\]

and the Hermitian third-order term

\[
K_3=PVRVRVP-\frac12\{PVP,PVR^2VP\}.
\]

The fixed-coupling truncated operator is

\[
K_{\rm eff}^{(3)}=t_1K_1+t_1^2K_2+t_1^3K_3.
\]

No finite-`t1` full eigendecomposition is used to construct these model inputs.

## Second-order path interference

For the OC-BULK0 block, the second-order contribution is decomposed by the intermediate strong component `c`:

\[
K_{2,\mathrm{OC,B0}}=\sum_c A_c^{(2)}.
\]

We define

\[
\eta_2=\frac{\|\sum_c A_c^{(2)}\|_F}{\sum_c\|A_c^{(2)}\|_F}.
\]

`eta2` is only interpreted when at least one two-step path is present.  A separate path-present flag prevents the absence of a path from being conflated with exact destructive cancellation.  v12 also records the path count, unsigned path-strength sum, component-graph distances, and the first nonvanishing direct OC-BULK0 effective order within `{K2,K3,>K3}`.

## Operator representation

The cache contains 24 tokens = 4 operator kinds x 6 symmetric sector pairs:

```text
K1[6], K2[6], K3[6], Keff3[6].
```

Each token contains kind/pair one-hots, normalized Frobenius strength, a deterministic spectral-strength slot, nonzero flag, sector rank fractions, diagonal flag, neighboring-order alignment/ratio, normalized trace and effective-rank surrogate.

The 24-dimensional global descriptor contains zero-sector rank fractions; mean K1/K2/K3/Keff3 strengths; K1-K2 and K2-K3 alignments; the three component distances; diagonal Keff3 traces; `eta2`; cancellation fraction; path presence/count/absolute strength; and a four-way first-order class.

All descriptors are based on invariant block norms/traces or graph quantities and are invariant to independent orthogonal basis changes inside the three parent zero subspaces.

## Full-spectrum network

Geometry is encoded by a 3/5/7 multi-scale CNN plus Transformer.  Operator tokens are encoded by a separate Transformer.  State queries use Fourier embeddings of the ordered state coordinate `q`, cross-attend to geometry and operator memories, then communicate globally through latent-spectrum blocks.  The energy decoder predicts nonnegative positive-half spacings and mirrors them exactly, giving a nondecreasing and exactly chiral spectrum while permitting exact zero gaps.

## Gauge-factorized projector head

The zero-sector probability is factorized into total mass and conditional parentage:

\[
C_{\rm zero}=m_0\,p(\mathrm{OC},\mathrm{ICII},\mathrm{B0}|\mathrm{zero}),
\]

while the six isolated branches use `(1-m0)` times a conditional six-way distribution.  Chiral pairing of projector probabilities is enforced by construction.

## Degenerate supervision

State-level projector/zero-parentage supervision is used on nondegenerate states and very high-purity degenerate states.  Mixed exact-degenerate subspaces are supervised through multiplicity-normalized aggregate projector traces and aggregate zero-sector parentage.  Cluster aggregation is a single GPU `index_add_` pass; there is no Python loop over clusters.

## Controlled operator ablations

`--operator-mode` supports:

```text
none
k1
k1k2
k1k2k3
full_no_hierarchy
full
```

`full_no_hierarchy` retains K1/K2/K3/Keff3 matrix tokens but hides eta2/path/order global descriptors, so the incremental value of the interference hierarchy can be tested directly.
