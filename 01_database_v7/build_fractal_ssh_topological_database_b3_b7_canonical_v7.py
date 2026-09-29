# -*- coding: utf-8 -*-
"""
Fractal SSH Topological Spectrum Database Generator — Canonical-Basis v6 (experimental)
===================================================

Locked database definition
--------------------------
Geometry:
    b = 3,4,5,6,7
    G = 2
    generator M is b x b binary mask
    exact G(2) = kron(M, M)

Strict family rule:
    C4 symmetric
    + exact recursive G2 self-similarity
    + G2 four-neighbor connected
    + nondegenerate (1 < N < b^2)
    + D4 deduplication

Expected strict family counts:
    b=3 ->   2
    b=4 ->   3
    b=5 ->  24
    b=6 ->  66
    b=7 -> 721
    total -> 816

Physics:
    finite model:
        t1 = -0.1
        t2 = -1.0

    reference model:
        t1 = 0.0
        t2 = -1.0

Hamiltonian rule:
    Build COMPLETE parent SSH Hamiltonian first.
    Each family is a principal-submatrix projection:

        H_family = P H_parent P

    Never infer t1/t2 from cropped geometry.

Reference components:
    size 1 monomer  -> OC
    size 2 dimer    -> ED-I / ED-II
    size 3 trimer   -> IC-I / IC-II / IC-III
    size 4 tetramer -> BULK-L / BULK-0 / BULK-U

Database outputs
----------------
1. geometry_manifest.csv
   one row per geometry/family

2. masks/*.npz
   M, G2

3. states/*.npz
   one file per family containing:
       eigenvalues, q_state
       branch_scores          # canonical state-resolved v6 ML target
       branch_scores_raw      # raw LAPACK-basis diagnostic
       branch_index           # canonical hard branch label
       cluster_id / cluster_size / cluster_energy
       cluster_scores_sum / cluster_scores_mean
       is_degenerate / is_zero_cluster
   NOTE: v6 fixes a deterministic canonical basis inside strictly degenerate
       eigenspaces by maximizing joint reference-projector purity, then fixes a
       deterministic within-cluster branch/purity/site order for q_n.
       Cluster projector weights are still saved as basis-invariant diagnostics.

4. state_manifest.csv
   one row per eigenstate:
       family_id
       state_index
       q_state
       energy
       branch
       max_overlap
       9 reference-projector scores

5. spectrum_status.csv
   resume/progress status

Usage examples
--------------
Geometry only:
    python build_fractal_ssh_topological_database_b3_b7_canonical_v7.py --stage geometry

Run b=3 spectrum/classification:
    python build_fractal_ssh_topological_database_b3_b7_canonical_v7.py --stage spectrum --only-b 3

Run b=3,4,5:
    python build_fractal_ssh_topological_database_b3_b7_canonical_v7.py --stage spectrum --only-b 3 4 5

Resume automatically:
    rerun the same command; completed state files are skipped.

Test only small matrices:
    python build_fractal_ssh_topological_database_b3_b7_canonical_v7.py --stage spectrum --max-sites 1200

Full:
    python build_fractal_ssh_topological_database_b3_b7_canonical_v7.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


# =============================================================================
# 0. LOCKED PARAMETERS
# =============================================================================

B_VALUES = (3, 4, 5, 6, 7)
EXPECTED_COUNTS = {3: 2, 4: 3, 5: 24, 6: 66, 7: 721}
EXPECTED_TOTAL = 816

GENERATION = 2

FINITE_T1 = -0.1
FINITE_T2 = -1.0

REF_T1 = 0.0
REF_T2 = -1.0

# Numerical validation controls.
#
# The key physical acceptance criterion is the projector rank identity
#
#   sum_n C[n,alpha] = Tr(P_alpha) = d_alpha.
#
# v7 does NOT apply a global QR repair to the full eigensystem, because such
# a repair could mix states belonging to distinct eigenvalues.  Orthogonality,
# eigenpair residual, projector completeness and projector-rank identities are
# checked explicitly; a family that exceeds the hard tolerances fails loudly.
# The legacy QR-related output fields are retained only for file compatibility
# and are always False/0 in v7.
EIGENVECTOR_ORTHO_WARN_TOL = 1e-10
EIGENVECTOR_ORTHO_HARD_TOL = 1e-7
REFERENCE_ORTHO_TOL = 1e-10
PROJECTOR_COMPLETENESS_TOL = 1e-10
PROJECTOR_RANK_TOL = 1e-8
EIGENPAIR_RESIDUAL_TOL = 1e-8
QR_REPAIR_DELTA_TOL = 1e-6

# Numerical degeneracy clustering. Only eigenvalues indistinguishable at numerical
# precision are merged by default; physically distinct nearby levels remain separate.
DEGENERACY_ABS_TOL = 1e-9
DEGENERACY_REL_TOL = 1e-10
ZERO_CLUSTER_ENERGY_TOL = 1e-9

# Canonical-basis v6. Only clusters whose energy spread is at machine-
# degeneracy scale are rotated; nearby but physically distinct levels are never
# mixed. The final definition maximizes projector purity jointly across all nine
# branches. The fixed codes below are used only for deterministic optimization
# initialization and are never added to H, used as labels, or used in the loss.
CANONICAL_ROTATION_ABS_TOL = 5e-12
CANONICAL_ROTATION_REL_TOL = 5e-12
CANONICAL_GAUGE_EIG_TOL = 1e-10
PURITY_INIT_BRANCH_CODES = {
    "bulk_lower": -3.0,
    "inner_type_I": -2.0,
    "edge_lower": -1.0,
    "outer_corner": 0.0,
    "inner_type_II": 1.0,
    "bulk_zero": 2.0,
    "edge_upper": 3.0,
    "inner_type_III": 4.0,
    "bulk_upper": 5.0,
}

BRANCH_ORDER = [
    "outer_corner",
    "edge_lower",
    "edge_upper",
    "inner_type_I",
    "inner_type_II",
    "inner_type_III",
    "bulk_lower",
    "bulk_zero",
    "bulk_upper",
]

BRANCH_LABEL = {
    "outer_corner": "OC",
    "edge_lower": "ED-I",
    "edge_upper": "ED-II",
    "inner_type_I": "IC-I",
    "inner_type_II": "IC-II",
    "inner_type_III": "IC-III",
    "bulk_lower": "BULK-L",
    "bulk_zero": "BULK-0",
    "bulk_upper": "BULK-U",
}

BRANCH_TO_INDEX = {lab: i for i, lab in enumerate(BRANCH_ORDER)}
INDEX_TO_BRANCH = {i: lab for lab, i in BRANCH_TO_INDEX.items()}

# v6.1 deterministic hard-label tie convention.
# Scores that differ only below these tolerances are treated as a physical/numerical tie
# and resolved by the fixed BRANCH_ORDER above. This prevents platform-dependent
# np.argmax flips in projector-flat sectors (notably Vicsek zero modes).
BRANCH_TIE_ABS_TOL = 1e-10
BRANCH_TIE_REL_TOL = 1e-10


def stable_branch_index(scores: np.ndarray) -> np.ndarray:
    """Return deterministic branch indices with tolerance-aware tie breaking.

    Accepts either shape (..., 9) or (9, n) only through explicit caller
    orientation; this helper assumes the last axis is the branch axis. For each
    state, every branch within atol+rtol*maxscore of the maximum is considered
    tied, and the first branch in the fixed BRANCH_ORDER is chosen.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.shape[-1] != len(BRANCH_ORDER):
        raise ValueError(f"stable_branch_index expects last axis={len(BRANCH_ORDER)}, got {S.shape}")
    mx = np.max(S, axis=-1, keepdims=True)
    tol = BRANCH_TIE_ABS_TOL + BRANCH_TIE_REL_TOL * np.maximum(1.0, np.abs(mx))
    tied = (mx - S) <= tol
    # argmax on bool returns the first True -> fixed BRANCH_ORDER tie break.
    return np.argmax(tied, axis=-1).astype(np.int16)


def stable_branch_index_columns(scores_9xn: np.ndarray) -> np.ndarray:
    """Column-oriented wrapper for projector_scores arrays of shape (9,n)."""
    S = np.asarray(scores_9xn, dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != len(BRANCH_ORDER):
        raise ValueError(f"expected (9,n) scores, got {S.shape}")
    return stable_branch_index(S.T)


# =============================================================================
# v7 HARD-LABEL DEFINITION
# =============================================================================
# The continuous projector scores remain the physical target.  v7 defines an
# additional *discrete* canonical label only for visualization/classification.
# The three zero-reference sectors are assigned globally with their exact
# reference ranks as capacities, maximizing total projector overlap.
ZERO_BRANCH_LABELS = ("outer_corner", "inner_type_II", "bulk_zero")
ZERO_BRANCH_INDICES = np.asarray([BRANCH_TO_INDEX[x] for x in ZERO_BRANCH_LABELS], dtype=np.int16)
NONZERO_BRANCH_INDICES = np.asarray([i for i in range(len(BRANCH_ORDER)) if i not in set(ZERO_BRANCH_INDICES.tolist())], dtype=np.int16)
ZERO_ASSIGN_TIE_EPS = 1e-9
ZERO_ASSIGN_INTEGRAL_TOL = 1e-7


def _deterministic_assignment_tie(state_indices: np.ndarray, nclass: int = 3) -> np.ndarray:
    """Small deterministic, branch-neutral tie perturbation.

    The perturbation is nine orders below an O(1) projector score and is used
    only to make exactly/near-exactly tied transportation optima reproducible.
    It depends on the immutable global eigenstate slot, not on input row order.
    """
    si = np.asarray(state_indices, dtype=np.int64)[:, None] + 1
    kk = np.arange(nclass, dtype=np.int64)[None, :] + 1
    h = (si * np.int64(73856093) + kk * np.int64(19349663) + si * kk * np.int64(83492791)) % np.int64(1000003)
    return h.astype(np.float64) / 1000003.0 - 0.5


def _capacity_assignment_three(scores: np.ndarray, capacities: np.ndarray, state_indices: np.ndarray):
    """Exact 3-class transportation LP for v7 zero-sector hard labels.

    Maximize sum_i score[i,label_i] subject to exact class capacities.  The LP
    relaxation is integral because this is a transportation / bipartite-flow
    polytope.  We nevertheless check integrality and counts explicitly.
    """
    S = np.asarray(scores, dtype=np.float64)
    caps = np.asarray(capacities, dtype=np.int64)
    state_indices = np.asarray(state_indices, dtype=np.int64)
    if S.ndim != 2 or S.shape[1] != 3:
        raise ValueError(f"zero assignment expects (n,3) scores, got {S.shape}")
    n = int(S.shape[0])
    if caps.shape != (3,) or np.any(caps < 0) or int(caps.sum()) != n:
        raise ValueError(f"invalid zero capacities {caps.tolist()} for n={n}")
    if n == 0:
        return np.empty(0, dtype=np.int16), {
            "objective": 0.0, "argmax_objective": 0.0, "regret": 0.0,
            "fractionality": 0.0,
        }

    K = 3
    var = np.arange(n * K, dtype=np.int64)
    item = np.repeat(np.arange(n, dtype=np.int64), K)
    cls = np.tile(np.arange(K, dtype=np.int64), n)
    rows = np.concatenate([item, n + cls])
    cols = np.concatenate([var, var])
    data = np.ones(rows.size, dtype=np.float64)
    Aeq = coo_matrix((data, (rows, cols)), shape=(n + K, n * K)).tocsr()
    beq = np.concatenate([np.ones(n, dtype=np.float64), caps.astype(np.float64)])

    effective = S + ZERO_ASSIGN_TIE_EPS * _deterministic_assignment_tie(state_indices, K)
    res = linprog(
        -effective.reshape(-1),
        A_eq=Aeq,
        b_eq=beq,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not res.success:
        raise RuntimeError(f"v7 zero-sector capacity assignment failed: {res.message}")

    X = np.asarray(res.x, dtype=np.float64).reshape(n, K)
    fractionality = float(np.max(np.minimum(np.abs(X), np.abs(1.0 - X))))
    if fractionality > ZERO_ASSIGN_INTEGRAL_TOL:
        raise RuntimeError(f"v7 zero assignment returned fractional solution: {fractionality:.3e}")
    local = np.argmax(X, axis=1).astype(np.int16)
    got = np.bincount(local, minlength=K).astype(np.int64)
    if not np.array_equal(got, caps):
        raise RuntimeError(f"v7 zero assignment capacity failure: got={got.tolist()} expected={caps.tolist()}")

    objective = float(np.sum(S[np.arange(n), local]))
    argmax_objective = float(np.sum(np.max(S, axis=1)))
    return local, {
        "objective": objective,
        "argmax_objective": argmax_objective,
        "regret": float(argmax_objective - objective),
        "fractionality": fractionality,
    }


def rank_consistent_hard_labels(eigvals: np.ndarray, canonical_scores_9xn: np.ndarray, reference):
    """Construct v7 hard labels without changing the continuous projector target.

    1) The zero-sector pool is the exact central spectral sector: the R0 states
       with smallest |E|, where R0 is the sum of reference ranks of
       OC/IC-II/BULK-0.
    2) As a physical audit, this pool must coincide with the states whose
       tolerance-stable projector argmax lies in those three zero branches.
    3) Within that pool solve the rank-constrained maximum-overlap assignment.
    4) The six nonzero branches keep their stable projector-argmax labels.

    Hence the hard-label counts equal the reference ranks exactly, while the
    soft projector scores remain untouched and continue to satisfy the trace
    identity independently.
    """
    E = np.asarray(eigvals, dtype=np.float64)
    S = np.asarray(canonical_scores_9xn, dtype=np.float64)
    n = E.size
    argmax_labels = stable_branch_index_columns(S)
    ref_dims = np.asarray([reference["ref_dims"][lab] for lab in BRANCH_ORDER], dtype=np.int64)
    zcaps = ref_dims[ZERO_BRANCH_INDICES.astype(np.int64)]
    nz = int(zcaps.sum())

    # Stable central-sector selection: |E| primary, original spectral slot secondary.
    central_order = np.lexsort((np.arange(n, dtype=np.int64), np.abs(E)))
    zero_states = np.sort(central_order[:nz].astype(np.int64))
    zero_mask = np.zeros(n, dtype=bool)
    zero_mask[zero_states] = True
    argmax_zero_mask = np.isin(argmax_labels, ZERO_BRANCH_INDICES)
    pool_mismatch = int(np.sum(zero_mask ^ argmax_zero_mask))
    if pool_mismatch != 0:
        raise RuntimeError(
            "v7 zero-sector pool mismatch: central-|E| sector and projector-argmax "
            f"sector differ on {pool_mismatch} states. Hard-label convention is not safe for this family."
        )

    # The six nonzero sectors are already rank-consistent in the locked t1/t2 regime.
    nonzero_counts = np.bincount(argmax_labels[~zero_mask], minlength=len(BRANCH_ORDER)).astype(np.int64)
    for k in NONZERO_BRANCH_INDICES.astype(np.int64):
        if int(nonzero_counts[k]) != int(ref_dims[k]):
            raise RuntimeError(
                f"v7 nonzero branch rank mismatch for {BRANCH_ORDER[k]}: "
                f"argmax={int(nonzero_counts[k])}, ref={int(ref_dims[k])}"
            )

    local_scores = S[ZERO_BRANCH_INDICES.astype(np.int64)][:, zero_states].T
    local_label, info = _capacity_assignment_three(local_scores, zcaps, zero_states)
    hard = argmax_labels.copy()
    hard[zero_states] = ZERO_BRANCH_INDICES[local_label]

    counts = np.bincount(hard, minlength=len(BRANCH_ORDER)).astype(np.int64)
    if not np.array_equal(counts, ref_dims):
        raise RuntimeError(
            f"v7 hard-label rank identity failed: counts={counts.tolist()} ref={ref_dims.tolist()}"
        )

    assigned_overlap = S[hard.astype(np.int64), np.arange(n)]
    changed = hard != argmax_labels
    return hard.astype(np.int16), {
        "argmax_labels": argmax_labels.astype(np.int16),
        "zero_candidate_mask": zero_mask,
        "zero_assignment_changed": changed & zero_mask,
        "assigned_overlap": assigned_overlap.astype(np.float64),
        "ref_dims": ref_dims,
        "counts": counts,
        "pool_mismatch": pool_mismatch,
        "zero_objective": info["objective"],
        "zero_argmax_objective": info["argmax_objective"],
        "zero_regret": info["regret"],
        "zero_fractionality": info["fractionality"],
        "n_zero_reassigned": int(np.sum(changed & zero_mask)),
    }


# =============================================================================
# 1. STRICT FRACTAL FAMILY ENUMERATION
# =============================================================================

def d4_images(A: np.ndarray):
    imgs = []
    X = A.copy()
    for _ in range(4):
        imgs.append(X.copy())
        imgs.append(np.fliplr(X))
        X = np.rot90(X)
    return imgs


def mask_signature(A: np.ndarray) -> str:
    return "".join("1" if x else "0" for x in A.astype(np.uint8).ravel())


def canonical_d4_signature(A: np.ndarray) -> str:
    return min(mask_signature(X) for X in d4_images(A))


def canonical_d4_mask(A: np.ndarray) -> np.ndarray:
    imgs = d4_images(A)
    sigs = [mask_signature(X) for X in imgs]
    return imgs[int(np.argmin(sigs))].astype(np.uint8)


def c4_orbits(b: int):
    unseen = {(r, c) for r in range(b) for c in range(b)}
    orbits = []

    while unseen:
        r, c = min(unseen)
        orbit = set()
        x, y = r, c

        for _ in range(4):
            orbit.add((x, y))
            x, y = y, b - 1 - x

        orbit = sorted(orbit)
        orbits.append(orbit)
        unseen -= set(orbit)

    return orbits


def mask_from_orbit_bits(b: int, orbits, code: int) -> np.ndarray:
    M = np.zeros((b, b), dtype=np.uint8)

    for k, orbit in enumerate(orbits):
        if (code >> k) & 1:
            for r, c in orbit:
                M[r, c] = 1

    return M


def is_four_neighbor_connected(mask: np.ndarray) -> bool:
    pos = np.argwhere(mask != 0)

    if len(pos) == 0:
        return False

    start = tuple(pos[0])
    q = deque([start])
    seen = {start}

    ny, nx = mask.shape

    while q:
        y, x = q.popleft()

        for dy, dx in ((1,0), (-1,0), (0,1), (0,-1)):
            yy, xx = y + dy, x + dx

            if (
                0 <= yy < ny
                and 0 <= xx < nx
                and mask[yy, xx]
                and (yy, xx) not in seen
            ):
                seen.add((yy, xx))
                q.append((yy, xx))

    return len(seen) == int(mask.sum())


def build_g2_mask(M: np.ndarray) -> np.ndarray:
    return np.kron(M, M).astype(np.uint8, copy=False)


@dataclass(frozen=True)
class FamilyRecord:
    family_id: str
    b: int
    family_index: int
    N: int
    Df: float
    n_cells_g2: int
    n_sites: int
    M: np.ndarray
    G2: np.ndarray


def enumerate_strict_families_for_b(b: int):
    orbits = c4_orbits(b)
    unique = {}

    for code in range(1 << len(orbits)):
        M = mask_from_orbit_bits(b, orbits, code)
        N = int(M.sum())

        # nondegenerate
        if N <= 1 or N >= b * b:
            continue

        G2 = build_g2_mask(M)

        # connectivity is checked on exact G2
        if not is_four_neighbor_connected(G2):
            continue

        sig = canonical_d4_signature(M)

        if sig not in unique:
            unique[sig] = canonical_d4_mask(M)

    families = list(unique.values())
    families.sort(key=lambda A: (int(A.sum()), mask_signature(A)))

    return families


def enumerate_all_families():
    records = []

    for b in B_VALUES:
        fams = enumerate_strict_families_for_b(b)

        if len(fams) != EXPECTED_COUNTS[b]:
            raise RuntimeError(
                f"STRICT FAMILY COUNT ERROR: b={b}, "
                f"got {len(fams)}, expected {EXPECTED_COUNTS[b]}"
            )

        for j, M in enumerate(fams, start=1):
            N = int(M.sum())
            Df = math.log(N) / math.log(b)
            G2 = build_g2_mask(M)

            n_cells_g2 = int(G2.sum())
            n_sites = 4 * n_cells_g2

            # exact G=2 identity
            if n_cells_g2 != N * N:
                raise RuntimeError("G2 retained-cell count != N^2")

            family_id = f"b{b}_F{j:04d}_N{N:02d}"

            records.append(
                FamilyRecord(
                    family_id=family_id,
                    b=b,
                    family_index=j,
                    N=N,
                    Df=Df,
                    n_cells_g2=n_cells_g2,
                    n_sites=n_sites,
                    M=M,
                    G2=G2,
                )
            )

    if len(records) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Total family count {len(records)} != {EXPECTED_TOTAL}"
        )

    return records


# =============================================================================
# 2. PARENT SSH HAMILTONIAN -> FAMILY PROJECTION
# =============================================================================

def parent_site_index(cx, cy, sx, sy, nx_cells):
    return (((cy * nx_cells + cx) * 2 + sy) * 2 + sx)


def build_parent_sparse_ssh_hamiltonian(
    nx_cells: int,
    ny_cells: int,
    t1: float,
    t2: float,
) -> csr_matrix:

    nsite = 4 * nx_cells * ny_cells

    rows = []
    cols = []
    vals = []

    def add_symmetric(i, j, amp):
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((amp, amp))

    for cy in range(ny_cells):
        for cx in range(nx_cells):

            # ---------------------------------------------------------
            # intracell t1
            # ---------------------------------------------------------
            for sy in (0, 1):
                add_symmetric(
                    parent_site_index(cx, cy, 0, sy, nx_cells),
                    parent_site_index(cx, cy, 1, sy, nx_cells),
                    t1,
                )

            for sx in (0, 1):
                add_symmetric(
                    parent_site_index(cx, cy, sx, 0, nx_cells),
                    parent_site_index(cx, cy, sx, 1, nx_cells),
                    t1,
                )

            # ---------------------------------------------------------
            # intercell t2
            # ---------------------------------------------------------
            if cx + 1 < nx_cells:
                for sy in (0, 1):
                    add_symmetric(
                        parent_site_index(cx, cy, 1, sy, nx_cells),
                        parent_site_index(cx + 1, cy, 0, sy, nx_cells),
                        t2,
                    )

            if cy + 1 < ny_cells:
                for sx in (0, 1):
                    add_symmetric(
                        parent_site_index(cx, cy, sx, 1, nx_cells),
                        parent_site_index(cx, cy + 1, sx, 0, nx_cells),
                        t2,
                    )

    H = coo_matrix(
        (vals, (rows, cols)),
        shape=(nsite, nsite),
        dtype=np.float64,
    ).tocsr()

    H.sum_duplicates()
    return H


def retained_parent_site_indices(cell_mask: np.ndarray):
    ny, nx = cell_mask.shape
    keep = []

    for cy in range(ny):
        for cx in range(nx):
            if not cell_mask[cy, cx]:
                continue

            for sy in (0, 1):
                for sx in (0, 1):
                    keep.append(
                        parent_site_index(cx, cy, sx, sy, nx)
                    )

    return np.asarray(keep, dtype=np.int64)


def project_family_hamiltonian(
    parent_H: csr_matrix,
    cell_mask: np.ndarray,
):
    keep = retained_parent_site_indices(cell_mask)
    H = parent_H[keep, :][:, keep].tocsr()

    expected = 4 * int(cell_mask.sum())

    if H.shape != (expected, expected):
        raise RuntimeError(
            f"Projected shape {H.shape}, expected {(expected, expected)}"
        )

    return H, keep


# =============================================================================
# 3. REFERENCE COMPONENTS AND PROJECTOR BASES
# =============================================================================

def classify_reference_local_state(
    size: int,
    energy: float,
    tol: float = 1e-8,
):
    if size == 1:
        return "outer_corner"

    if size == 2:
        if energy < 0:
            return "edge_lower"
        return "edge_upper"

    if size == 3:
        if energy < -tol:
            return "inner_type_I"
        if energy > tol:
            return "inner_type_III"
        return "inner_type_II"

    if size == 4:
        if energy < -tol:
            return "bulk_lower"
        if energy > tol:
            return "bulk_upper"
        return "bulk_zero"

    raise RuntimeError(
        f"Unexpected reference component size = {size}"
    )


def build_reference_from_parent(
    parent_ref: csr_matrix,
    G2: np.ndarray,
):
    H0_sparse, kept = project_family_hamiltonian(
        parent_ref,
        G2,
    )
    H0_sparse.eliminate_zeros()

    # strong-bond graph
    A = H0_sparse.copy()
    A.data = np.ones_like(A.data)

    ncomp, labels = connected_components(
        A,
        directed=False,
        return_labels=True,
    )

    comp_sizes = np.bincount(labels)

    if np.any((comp_sizes < 1) | (comp_sizes > 4)):
        raise RuntimeError(
            f"Unexpected component sizes: "
            f"{sorted(set(comp_sizes.tolist()))}"
        )

    component_counts = {
        "n_monomer": int(np.sum(comp_sizes == 1)),
        "n_dimer": int(np.sum(comp_sizes == 2)),
        "n_trimer": int(np.sum(comp_sizes == 3)),
        "n_tetramer": int(np.sum(comp_sizes == 4)),
    }

    nsite = H0_sparse.shape[0]

    ref_vectors = []
    ref_labels = []
    ref_energies = []

    # diagonalize each disconnected local component separately
    for cid in range(ncomp):
        comp = np.flatnonzero(labels == cid)
        Hc = H0_sparse[comp, :][:, comp].toarray()

        evals, evecs = np.linalg.eigh(Hc)

        for k, energy in enumerate(evals):
            lab = classify_reference_local_state(
                len(comp),
                float(energy),
            )

            full = np.zeros(nsite, dtype=float)
            full[comp] = evecs[:, k]

            ref_vectors.append(full)
            ref_labels.append(lab)
            ref_energies.append(float(energy))

    ref_vectors = np.column_stack(ref_vectors)
    ref_labels = np.asarray(ref_labels, dtype=object)
    ref_energies = np.asarray(ref_energies, dtype=float)

    order = np.argsort(ref_energies)

    ref_vectors = ref_vectors[:, order]
    ref_labels = ref_labels[order]
    ref_energies = ref_energies[order]

    branch_basis = {}
    ref_dims = {}

    for lab in BRANCH_ORDER:
        idx = np.where(ref_labels == lab)[0]
        B = ref_vectors[:, idx]

        branch_basis[lab] = B
        ref_dims[lab] = int(B.shape[1])

    # exact completeness check
    if sum(ref_dims.values()) != nsite:
        raise RuntimeError(
            "Reference branch dimensions do not sum to nsite."
        )

    component_site_count = (
        component_counts["n_monomer"]
        + 2 * component_counts["n_dimer"]
        + 3 * component_counts["n_trimer"]
        + 4 * component_counts["n_tetramer"]
    )

    if component_site_count != nsite:
        raise RuntimeError(
            "Reference component count does not reproduce nsite."
        )

    bulk_state_free = (
        component_counts["n_tetramer"] == 0
    )

    return {
        "branch_basis": branch_basis,
        "ref_dims": ref_dims,
        "component_counts": component_counts,
        "bulk_state_free": bulk_state_free,
        "ref_energies": ref_energies,
        "n_components": int(ncomp),
    }


# =============================================================================
# 4. FINITE SPECTRUM + REFERENCE-PROJECTOR CLASSIFICATION
# =============================================================================

def cluster_sorted_eigenvalues(
    eigvals: np.ndarray,
    abs_tol: float = DEGENERACY_ABS_TOL,
    rel_tol: float = DEGENERACY_REL_TOL,
):
    """Cluster a sorted Hermitian spectrum into numerically degenerate blocks.

    The comparison is anchored at the first eigenvalue of each cluster, which
    prevents a long chain of merely-close levels from being merged transitively.
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    n = len(eigvals)
    if n == 0:
        return np.empty(0, dtype=np.int32), []

    cluster_id = np.empty(n, dtype=np.int32)
    clusters = []
    start = 0
    cid = 0

    for i in range(1, n + 1):
        close = False
        if i < n:
            anchor = float(eigvals[start])
            x = float(eigvals[i])
            tol = max(abs_tol, rel_tol * max(1.0, abs(anchor), abs(x)))
            close = abs(x - anchor) <= tol

        if i == n or not close:
            inds = np.arange(start, i, dtype=np.int32)
            cluster_id[inds] = cid
            clusters.append(inds)
            cid += 1
            start = i

    return cluster_id, clusters


def _fix_column_signs(V: np.ndarray) -> np.ndarray:
    """Deterministically fix the real sign of each vector for reproducibility."""
    V = np.asarray(V, dtype=np.float64).copy()
    for j in range(V.shape[1]):
        col = V[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0:
            V[:, j] *= -1.0
    return V


def _projector_restrictions(Vd: np.ndarray, reference):
    """Return G_alpha = Vd^T P_alpha^ref Vd for all nine branches."""
    out = []
    for lab in BRANCH_ORDER:
        B = reference["branch_basis"][lab]
        if B.shape[1] == 0:
            G = np.zeros((Vd.shape[1], Vd.shape[1]), dtype=np.float64)
        else:
            O = B.conj().T @ Vd
            G = (O.conj().T @ O).real
        out.append((G + G.T) * 0.5)
    return out


def _projector_purity_objective_from_mats(mats) -> float:
    """J(U)=sum_alpha ||diag(U^T G_alpha U)||_2^2.

    Since the Frobenius norm of each G_alpha is invariant under orthogonal
    rotation, maximizing J is exactly equivalent to minimizing the summed
    off-diagonal projector mixing.
    """
    return float(sum(np.sum(np.diag(A) ** 2) for A in mats))


def _deterministic_initializer(Vd: np.ndarray, reference, kept: np.ndarray):
    """Gauge-covariant deterministic initializer from the subspace projector.

    v6.0 used an eigendecomposition of a weighted projector sum as the Jacobi
    initializer.  In exactly symmetric blocks that auxiliary operator can have
    residual degeneracies, and different LAPACK implementations may choose
    different vectors inside those ties.  A subsequent non-linear Jacobi sweep
    can then converge to symmetry-related but differently ordered stationary
    points.

    This hotfix builds the initializer directly from projected *site* vectors
    using deterministic pivoted modified Gram-Schmidt.  It depends only on the
    physical subspace projector P_D=Vd Vd^T and the fixed parent-site ordering.
    Under Vd -> Vd Q the coordinate initializer transforms as U0 -> Q^T U0,
    so the physical starting basis Vd U0 is gauge invariant (up to roundoff).

    The site rule is only a deterministic tie-break / starting convention.  It
    is not added to H, is not a label, and does not change the final symmetric
    joint-projector-purity objective.
    """
    Vd = np.asarray(Vd, dtype=np.float64)
    m = int(Vd.shape[1])
    if m <= 1:
        return np.eye(m, dtype=np.float64)

    # Each row of Vd is the coordinate vector of P_D |site> in the Vd basis.
    # Residual rows therefore transform covariantly under Vd -> Vd Q.
    R = Vd.copy()                         # shape (n_site, m)
    nsite = R.shape[0]
    kept = np.asarray(kept, dtype=np.int64)
    if kept.shape[0] != nsite:
        raise ValueError("kept/site dimension mismatch in canonical initializer")

    U0 = np.zeros((m, m), dtype=np.float64)
    used = np.zeros(nsite, dtype=bool)

    for j in range(m):
        norm2 = np.einsum("ij,ij->i", R, R)
        norm2[used] = -1.0
        best = float(np.max(norm2))
        if not np.isfinite(best) or best <= 1e-24:
            raise RuntimeError(
                f"Projected-site initializer lost rank at column {j}/{m}; best residual={best:.3e}"
            )

        # Treat machine-level residual-norm ties as exact ties and resolve them
        # with the fixed parent-site index.  This avoids BLAS/LAPACK dependent
        # pivot flips on symmetry-related sites.
        tie_tol = max(1e-14, 2e-12 * max(1.0, abs(best)))
        cand = np.flatnonzero((~used) & (norm2 >= best - tie_tol))
        if cand.size == 0:
            pivot = int(np.argmax(norm2))
        else:
            pivot = int(cand[np.argmin(kept[cand])])

        v = R[pivot].copy()
        # Two-pass reorthogonalization makes the physical starting basis stable
        # even for large exact-degeneracy blocks.
        if j:
            Qprev = U0[:, :j]
            v -= Qprev @ (Qprev.T @ v)
            v -= Qprev @ (Qprev.T @ v)
        nv = float(np.linalg.norm(v))
        if nv <= 1e-12:
            # Extremely rare: the pivot was numerically tied before the second
            # reorthogonalization. Recompute all residuals from the original
            # site vectors and choose the strongest surviving candidate.
            Qprev = U0[:, :j]
            RR = Vd.copy()
            if j:
                RR -= (RR @ Qprev) @ Qprev.T
                RR -= (RR @ Qprev) @ Qprev.T
            nn = np.einsum("ij,ij->i", RR, RR)
            nn[used] = -1.0
            best2 = float(np.max(nn))
            cand2 = np.flatnonzero((~used) & (nn >= best2 - max(1e-14, 2e-12 * max(1.0, abs(best2)))))
            pivot = int(cand2[np.argmin(kept[cand2])]) if cand2.size else int(np.argmax(nn))
            v = RR[pivot].copy()
            nv = float(np.linalg.norm(v))
            if nv <= 1e-12:
                raise RuntimeError(f"Projected-site initializer unstable at column {j}/{m}")
            R = RR

        v /= nv
        # The construction from the pivot residual already fixes orientation;
        # enforce it again using the physical pivot amplitude for safety.
        phys = Vd @ v
        if phys[pivot] < 0.0:
            v *= -1.0

        U0[:, j] = v
        used[pivot] = True

        # Deflate this direction from all projected-site coordinate vectors.
        coeff = R @ v
        R -= coeff[:, None] * v[None, :]
        # A second deflation pass controls accumulated orthogonality error.
        coeff = R @ v
        R -= coeff[:, None] * v[None, :]

    # Final polar repair is coordinate-covariant and only removes accumulated
    # floating-point drift; it does not change the spanned physical basis.
    gram = (U0.T @ U0 + (U0.T @ U0).T) * 0.5
    ew, EV = np.linalg.eigh(gram)
    if float(np.min(ew)) <= 0.0:
        raise RuntimeError("Projected-site initializer Gram matrix is not positive definite")
    U0 = U0 @ (EV @ np.diag(1.0 / np.sqrt(ew)) @ EV.T)
    return U0

def _jacobi_projector_purity(
    Gs,
    U0: np.ndarray,
    max_sweeps: int = 8,
    angle_tol: float = 1e-11,
    flat_rel_tol: float = 5e-13,
    gain_rel_tol: float = 5e-14,
):
    """Stable joint diagonalization of the restricted projectors.

    v6.0/v6.0.1 could apply an O(1) Jacobi rotation in a *flat* two-state
    plane where the purity objective changed only at roundoff level.  The
    maximizing angle is mathematically undefined in that situation, so tiny
    BLAS/LAPACK differences could choose different symmetry-related rotations
    (this is exactly what happened for VicsekCross_G2 on Windows).

    v6.1 therefore accepts a plane rotation only when BOTH:
      1) the 2x2 purity quadratic form has a resolved eigenvalue gap; and
      2) the candidate rotation gives a resolved positive increase of the
         actual diagonal-purity objective for that (p,q) plane.

    Flat/indistinguishable planes are left in the projected-site deterministic
    initializer.  That initializer depends only on P_D and fixed site order,
    so retaining it is gauge covariant and reproducible.
    """
    m = U0.shape[0]
    if m <= 1:
        mats = [A.copy() for A in Gs]
        return U0.copy(), mats, 0

    U = np.asarray(U0, dtype=np.float64).copy()
    mats = [U.T @ A @ U for A in Gs]
    mats = [(A + A.T) * 0.5 for A in mats]

    for sweep in range(max_sweeps):
        nrot = 0
        max_angle = 0.0

        for p in range(m - 1):
            for q in range(p + 1, m):
                M00 = 0.0
                M01 = 0.0
                M11 = 0.0
                local_before = 0.0
                for A in mats:
                    app = float(A[p, p])
                    aqq = float(A[q, q])
                    apq = float(A[p, q])
                    delta = app - aqq
                    M00 += delta * delta
                    M01 += 2.0 * delta * apq
                    M11 += 4.0 * apq * apq
                    local_before += app * app + aqq * aqq

                # If the 2x2 quadratic form is (nearly) isotropic, the optimal
                # angle is undefined.  DO NOT rotate: keep the deterministic
                # projected-site basis instead of amplifying roundoff noise.
                qscale = max(1.0, abs(M00), abs(M11), abs(M01))
                eig_gap = float(np.hypot(M00 - M11, 2.0 * M01))
                if eig_gap <= flat_rel_tol * qscale:
                    continue

                phi = 0.5 * np.arctan2(2.0 * M01, M00 - M11)
                c2a, s2a = np.cos(phi), np.sin(phi)
                qa = M00*c2a*c2a + 2*M01*c2a*s2a + M11*s2a*s2a
                phi_b = phi + 0.5*np.pi
                c2b, s2b = np.cos(phi_b), np.sin(phi_b)
                qb = M00*c2b*c2b + 2*M01*c2b*s2b + M11*s2b*s2b
                phi2 = phi_b if qb > qa else phi

                while phi2 > 0.5*np.pi:
                    phi2 -= np.pi
                while phi2 < -0.5*np.pi:
                    phi2 += np.pi
                theta = 0.5 * phi2
                if abs(theta) <= angle_tol:
                    continue

                c = float(np.cos(theta))
                ss = float(np.sin(theta))

                # Explicitly evaluate the actual objective gain before touching
                # U/mats.  This rejects numerically arbitrary rotations in flat
                # symmetry sectors even if atan2 returns a large angle.
                local_after = 0.0
                c2 = c*c
                s2 = ss*ss
                cs2 = 2.0*c*ss
                for A in mats:
                    app = float(A[p, p])
                    aqq = float(A[q, q])
                    apq = float(A[p, q])
                    app_new = c2*app + cs2*apq + s2*aqq
                    aqq_new = s2*app - cs2*apq + c2*aqq
                    local_after += app_new*app_new + aqq_new*aqq_new

                gain = local_after - local_before
                gain_scale = max(1.0, abs(local_before), abs(local_after))
                if gain <= gain_rel_tol * gain_scale:
                    continue

                # In-place A <- R^T A R for R=[[c,-s],[s,c]].
                for A in mats:
                    cp = A[:, p].copy()
                    cq = A[:, q].copy()
                    A[:, p] = c * cp + ss * cq
                    A[:, q] = -ss * cp + c * cq

                    rp = A[p, :].copy()
                    rq = A[q, :].copy()
                    A[p, :] = c * rp + ss * rq
                    A[q, :] = -ss * rp + c * rq
                    # Restore symmetry explicitly after finite-precision updates.
                    apq_sym = 0.5 * (A[p, q] + A[q, p])
                    A[p, q] = A[q, p] = apq_sym

                up = U[:, p].copy()
                uq = U[:, q].copy()
                U[:, p] = c * up + ss * uq
                U[:, q] = -ss * up + c * uq
                nrot += 1
                max_angle = max(max_angle, abs(theta))

        if nrot == 0 or max_angle <= angle_tol:
            return U, mats, sweep + 1

    return U, mats, max_sweeps

def _state_scores_for_block(Vc: np.ndarray, reference) -> np.ndarray:
    """Return shape (m, 9) state-resolved reference-projector scores."""
    m = Vc.shape[1]
    S = np.zeros((m, len(BRANCH_ORDER)), dtype=np.float64)
    for a, lab in enumerate(BRANCH_ORDER):
        B = reference["branch_basis"][lab]
        if B.shape[1]:
            S[:, a] = np.sum(np.abs(B.conj().T @ Vc) ** 2, axis=0)
    return S



def _resolve_projector_signature_degeneracy(
    Vc: np.ndarray,
    scores: np.ndarray,
    kept: np.ndarray,
    score_tol: float = 2e-9,
):
    """Fix residual basis freedom invisible to all branch projectors.

    Joint projector diagonalization can leave an arbitrary orthogonal rotation
    among states that have the same nine-component projector signature.  Such a
    rotation does not change the branch-purity objective.  We resolve only this
    residual freedom with deterministic site-index operators, preserving the
    projector definition while making the actual basis reproducible.
    """
    m = Vc.shape[1]
    if m <= 1:
        return _fix_column_signs(Vc)

    # Greedy equivalence classes of equal projector signatures.
    unused = set(range(m))
    groups = []
    while unused:
        i = min(unused)
        group = [j for j in sorted(unused) if np.max(np.abs(scores[j] - scores[i])) <= score_tol]
        for j in group:
            unused.remove(j)
        groups.append(group)

    nparent = max(int(np.max(kept)) + 1, 1)
    x = (kept.astype(np.float64) + 1.0) / float(nparent)
    w1 = x + 0.17320508075688773 * x**2 + 0.04142135623730951 * x**3
    w2 = x**2 + 0.2718281828459045 * x**4 + 0.01973920880217872 * x**5

    out = Vc.copy()
    for group in groups:
        if len(group) <= 1:
            continue
        idx = np.asarray(group, dtype=np.int32)
        W = out[:, idx]
        S1 = W.T @ (w1[:, None] * W)
        mu, R = np.linalg.eigh((S1 + S1.T) * 0.5)
        oo = np.argsort(mu, kind="stable")
        mu = mu[oo]
        W = W @ R[:, oo]

        # Resolve any residual site-operator degeneracy with an independent
        # deterministic moment, but never mix distinct first-operator values.
        a = 0
        while a < len(idx):
            b = a + 1
            while b < len(idx):
                tol = 1e-11 * max(1.0, abs(mu[a]), abs(mu[b]))
                if abs(mu[b] - mu[a]) > tol:
                    break
                b += 1
            if b - a > 1:
                Z = W[:, a:b]
                S2 = Z.T @ (w2[:, None] * Z)
                nu, R2 = np.linalg.eigh((S2 + S2.T) * 0.5)
                o2 = np.argsort(nu, kind="stable")
                W[:, a:b] = Z @ R2[:, o2]
            a = b
        out[:, idx] = W

    return _fix_column_signs(out)

def _canonical_sequence_order(Vc: np.ndarray, scores: np.ndarray, kept: np.ndarray):
    """Deterministic within-cluster ordering for sequence learning.

    Primary key   : fixed nine-branch order (OC, ED-I, ..., BULK-U)
    Secondary key : decreasing projector purity
    Tie-breakers  : deterministic site-index moments

    This changes no physical eigenvalue or subspace. It only assigns a stable
    q-index order to states that are exactly degenerate at machine precision.
    """
    branch = stable_branch_index(scores)
    purity = np.max(scores, axis=1)

    nparent = max(int(np.max(kept)) + 1, 1)
    x = (kept.astype(np.float64) + 1.0) / float(nparent)
    w1 = x + 0.17320508075688773 * x**2 + 0.04142135623730951 * x**3
    w2 = x**2 + 0.2718281828459045 * x**4
    dens = np.abs(Vc) ** 2
    site1 = np.sum(dens * w1[:, None], axis=0)
    site2 = np.sum(dens * w2[:, None], axis=0)

    # Quantize sorting keys well above machine noise so an arbitrary rotation
    # of the input degenerate basis cannot flip states that are physically tied.
    purity_key = np.round(purity, 12)
    site1_key = np.round(site1, 12)
    site2_key = np.round(site2, 12)

    # np.lexsort: last key is primary.
    order = np.lexsort((site2_key, site1_key, -purity_key, branch))
    return order.astype(np.int32), branch[order], purity[order]


def _canonicalize_degenerate_block(
    Vd: np.ndarray,
    reference,
    kept: np.ndarray,
):
    """v7 canonical basis: projector-purity optimization + canonical ordering.

    Final mathematical definition
    -----------------------------
      U* = argmax_U sum_{j,alpha} [u_j^T G_alpha u_j]^2,
      G_alpha = Vd^T P_alpha^ref Vd,
      U^T U = I.

    Equivalently, v6 minimizes the summed off-diagonal mixing of all restricted
    reference projectors.  The objective treats all nine branches symmetrically.
    After the basis is fixed, exactly-degenerate states are deterministically
    ordered by branch -> purity -> site-index tie-breaker so q_n is reproducible.
    """
    m = Vd.shape[1]
    if m <= 1:
        Vc = _fix_column_signs(Vd)
        S = _state_scores_for_block(Vc, reference)
        return Vc, {
            "purity_objective_before": float(np.sum(S**2)),
            "purity_objective_after": float(np.sum(S**2)),
            "joint_diag_sweeps": 0,
            "sequence_order": np.arange(m, dtype=np.int32),
            "branch_index": stable_branch_index(S),
            "purity": np.max(S, axis=1),
        }

    Gs = _projector_restrictions(Vd, reference)
    U0 = _deterministic_initializer(Vd, reference, kept)
    mats0 = [U0.T @ G @ U0 for G in Gs]
    obj0 = _projector_purity_objective_from_mats(mats0)

    U, mats, sweeps = _jacobi_projector_purity(Gs, U0)
    obj1 = _projector_purity_objective_from_mats(mats)
    if obj1 + 1e-10 < obj0:
        raise RuntimeError(
            f"Projector-purity optimization decreased objective: {obj0:.12g} -> {obj1:.12g}"
        )

    Vc = _fix_column_signs(Vd @ U)
    S = _state_scores_for_block(Vc, reference)

    # v6.1: do NOT rotate groups merely because their *diagonal* projector
    # score vectors happen to coincide. Equal diagonals do not prove that all
    # restricted projectors are scalar on that group; the old resolver could
    # therefore introduce a platform-dependent basis rotation. Any genuinely
    # flat residual freedom is left in the gauge-covariant projected-site basis.

    order, branch_sorted, purity_sorted = _canonical_sequence_order(Vc, S, kept)
    Vc = _fix_column_signs(Vc[:, order])

    return Vc, {
        "purity_objective_before": obj0,
        "purity_objective_after": obj1,
        "joint_diag_sweeps": int(sweeps),
        "sequence_order": order,
        "branch_index": branch_sorted,
        "purity": purity_sorted,
    }

def solve_finite_and_classify(
    parent_finite: csr_matrix,
    G2: np.ndarray,
    reference,
    degeneracy_abs_tol: float = DEGENERACY_ABS_TOL,
    degeneracy_rel_tol: float = DEGENERACY_REL_TOL,
):
    """Solve H and construct v7 canonical state-resolved projector labels.

    The physical Hamiltonian and eigenvalues are unchanged.  Only an *exactly
    degenerate* eigenspace is rotated to maximize joint reference-projector
    purity and then deterministically ordered for sequence learning.  This
    converts the otherwise arbitrary LAPACK basis/order into a reproducible
    state-resolved convention. Basis-invariant cluster weights are retained as
    an independent audit quantity.
    """
    H_sparse, kept = project_family_hamiltonian(parent_finite, G2)
    H = H_sparse.toarray()

    eigvals, eigvecs_raw = linalg.eigh(
        H, overwrite_a=True, check_finite=False, driver="evd"
    )
    nstate = len(eigvals)

    def orthogonality_error(V):
        G = V.conj().T @ V
        return float(np.max(np.abs(G - np.eye(G.shape[0], dtype=G.dtype))))

    def eigenpair_residual_error(V):
        R = H_sparse @ V - V * eigvals[np.newaxis, :]
        scale = max(1.0, float(np.max(np.abs(eigvals))))
        return float(np.max(np.abs(R)) / scale)

    def projector_scores(V):
        scores_local = np.zeros((len(BRANCH_ORDER), V.shape[1]), dtype=np.float64)
        for a, lab in enumerate(BRANCH_ORDER):
            B = reference["branch_basis"][lab]
            if B.shape[1] == 0:
                continue
            overlaps = B.conj().T @ V
            scores_local[a, :] = np.sum(np.abs(overlaps) ** 2, axis=0)
        return scores_local

    def projector_diagnostics(scores_local):
        completeness = float(np.max(np.abs(np.sum(scores_local, axis=0) - 1.0)))
        ref_dims_vec = np.asarray(
            [reference["ref_dims"][lab] for lab in BRANCH_ORDER], dtype=np.float64
        )
        trace_vec = np.sum(scores_local, axis=1)
        rank_errors = np.abs(trace_vec - ref_dims_vec)
        return completeness, ref_dims_vec, trace_vec, rank_errors, float(np.max(rank_errors))

    ref_blocks = [
        reference["branch_basis"][lab]
        for lab in BRANCH_ORDER
        if reference["branch_basis"][lab].shape[1] > 0
    ]
    Bref = np.column_stack(ref_blocks)
    reference_orthogonality_error = orthogonality_error(Bref)
    if reference_orthogonality_error > REFERENCE_ORTHO_TOL:
        raise RuntimeError(
            f"Reference basis orthogonality failed: {reference_orthogonality_error:.3e}"
        )

    raw_ortho = orthogonality_error(eigvecs_raw)
    raw_resid = eigenpair_residual_error(eigvecs_raw)
    raw_scores = projector_scores(eigvecs_raw)
    (
        completeness_error,
        ref_dims_vec,
        projector_trace_vec,
        projector_rank_errors,
        projector_rank_error,
    ) = projector_diagnostics(raw_scores)

    if raw_ortho > EIGENVECTOR_ORTHO_HARD_TOL:
        raise RuntimeError(f"Raw eigensystem orthogonality failed: {raw_ortho:.3e}")
    if raw_resid > EIGENPAIR_RESIDUAL_TOL:
        raise RuntimeError(f"Raw eigensystem residual failed: {raw_resid:.3e}")
    if completeness_error > PROJECTOR_COMPLETENESS_TOL:
        raise RuntimeError(f"Raw projector completeness failed: {completeness_error:.3e}")
    if projector_rank_error > PROJECTOR_RANK_TOL:
        raise RuntimeError(f"Raw projector rank identity failed: {projector_rank_error:.3e}")

    cluster_id, clusters = cluster_sorted_eigenvalues(
        eigvals, abs_tol=degeneracy_abs_tol, rel_tol=degeneracy_rel_tol
    )
    ncluster = len(clusters)
    cluster_size = np.empty(nstate, dtype=np.int32)
    cluster_energy = np.empty(nstate, dtype=np.float64)
    is_degenerate = np.zeros(nstate, dtype=bool)
    is_zero_cluster = np.zeros(nstate, dtype=bool)
    canonicalized = np.zeros(nstate, dtype=bool)
    # Legacy v5 field retained for backward file compatibility only.
    canonical_gauge_value = np.full(nstate, np.nan, dtype=np.float64)
    canonical_purity_value = np.full(nstate, np.nan, dtype=np.float64)
    canonical_preorder_index = np.full(nstate, -1, dtype=np.int32)

    cluster_scores_sum = np.zeros((ncluster, len(BRANCH_ORDER)), dtype=np.float64)
    cluster_scores_mean = np.zeros_like(cluster_scores_sum)
    cluster_energies = np.zeros(ncluster, dtype=np.float64)
    cluster_sizes = np.zeros(ncluster, dtype=np.int32)
    cluster_purity_objective_before = np.full(ncluster, np.nan, dtype=np.float64)
    cluster_purity_objective_after = np.full(ncluster, np.nan, dtype=np.float64)
    cluster_joint_diag_sweeps = np.zeros(ncluster, dtype=np.int16)
    cluster_branch_runs = np.zeros(ncluster, dtype=np.int16)

    eigvecs = eigvecs_raw.copy()

    # First compute basis-invariant cluster weights from the raw eigensystem,
    # then canonicalize only machine-degenerate clusters.
    for cid, inds in enumerate(clusters):
        m = int(len(inds))
        ebar = float(np.mean(eigvals[inds]))
        spread = float(np.ptp(eigvals[inds])) if m > 1 else 0.0
        wsum = np.sum(raw_scores[:, inds], axis=1)

        cluster_scores_sum[cid, :] = wsum
        cluster_scores_mean[cid, :] = wsum / float(m)
        cluster_energies[cid] = ebar
        cluster_sizes[cid] = m
        cluster_size[inds] = m
        cluster_energy[inds] = ebar
        is_degenerate[inds] = m > 1
        is_zero_cluster[inds] = (m > 1 and abs(ebar) <= ZERO_CLUSTER_ENERGY_TOL)

        rot_tol = max(
            CANONICAL_ROTATION_ABS_TOL,
            CANONICAL_ROTATION_REL_TOL * max(1.0, abs(ebar)),
        )
        if m > 1 and spread <= rot_tol:
            Vc, can_info = _canonicalize_degenerate_block(
                eigvecs_raw[:, inds], reference, kept
            )
            # IMPORTANT: eigenvalues remain in their original ascending solver
            # slots.  Because this block is machine-degenerate, assigning the
            # branch-ordered canonical basis to those slots preserves the exact
            # numerical spectrum while giving q a deterministic branch order.
            eigvecs[:, inds] = Vc
            canonicalized[inds] = True
            canonical_purity_value[inds] = np.asarray(can_info["purity"], dtype=np.float64)
            canonical_preorder_index[inds] = np.asarray(can_info["sequence_order"], dtype=np.int32)
            cluster_purity_objective_before[cid] = float(can_info["purity_objective_before"])
            cluster_purity_objective_after[cid] = float(can_info["purity_objective_after"])
            cluster_joint_diag_sweeps[cid] = int(can_info["joint_diag_sweeps"])
            bseq = np.asarray(can_info["branch_index"], dtype=np.int16)
            cluster_branch_runs[cid] = int(1 + np.sum(bseq[1:] != bseq[:-1])) if m else 0

    canonical_scores = projector_scores(eigvecs)

    # v6 sequence-order audit: inside every canonicalized exact-degenerate
    # block the hard branch indices must be nondecreasing by construction.
    max_canonical_order_violation = 0
    for inds in clusters:
        if len(inds) <= 1 or not np.all(canonicalized[inds]):
            continue
        bseq = stable_branch_index_columns(canonical_scores[:, inds])
        if np.any(np.diff(bseq) < 0):
            max_canonical_order_violation = 1
            break
    if max_canonical_order_violation:
        raise RuntimeError("v7 canonical sequence ordering failed inside a degenerate cluster")

    (
        canonical_completeness_error,
        _,
        canonical_trace_vec,
        canonical_rank_errors,
        canonical_rank_error,
    ) = projector_diagnostics(canonical_scores)
    can_ortho = orthogonality_error(eigvecs)
    can_resid = eigenpair_residual_error(eigvecs)

    if can_ortho > EIGENVECTOR_ORTHO_HARD_TOL:
        raise RuntimeError(f"Canonical eigensystem orthogonality failed: {can_ortho:.3e}")
    if can_resid > EIGENPAIR_RESIDUAL_TOL:
        raise RuntimeError(
            "Canonical rotation mixed nondegenerate energies: "
            f"residual={can_resid:.3e}. Tighten CANONICAL_ROTATION_*_TOL."
        )
    if canonical_completeness_error > PROJECTOR_COMPLETENESS_TOL:
        raise RuntimeError(
            f"Canonical projector completeness failed: {canonical_completeness_error:.3e}"
        )
    if canonical_rank_error > PROJECTOR_RANK_TOL:
        raise RuntimeError(f"Canonical projector rank identity failed: {canonical_rank_error:.3e}")

    # Cluster sums must be invariant under canonical rotations.
    max_cluster_invariance_error = 0.0
    for cid, inds in enumerate(clusters):
        w_can = np.sum(canonical_scores[:, inds], axis=1)
        max_cluster_invariance_error = max(
            max_cluster_invariance_error,
            float(np.max(np.abs(w_can - cluster_scores_sum[cid, :]))),
        )
    if max_cluster_invariance_error > 1e-8:
        raise RuntimeError(
            f"Cluster projector invariance failed: {max_cluster_invariance_error:.3e}"
        )

    cluster_completeness_error = float(
        np.max(np.abs(np.sum(cluster_scores_sum, axis=1) - cluster_sizes))
    ) if ncluster else 0.0

    raw_branch_idx = stable_branch_index_columns(raw_scores)
    canonical_argmax_idx = stable_branch_index_columns(canonical_scores)
    hard_branch_idx, hard_info = rank_consistent_hard_labels(eigvals, canonical_scores, reference)
    raw_max_overlap = np.max(raw_scores, axis=0)
    canonical_max_overlap = np.max(canonical_scores, axis=0)

    assigned_counts_raw = {
        lab: int(np.sum(raw_branch_idx == BRANCH_TO_INDEX[lab])) for lab in BRANCH_ORDER
    }
    assigned_counts_argmax = {
        lab: int(np.sum(canonical_argmax_idx == BRANCH_TO_INDEX[lab])) for lab in BRANCH_ORDER
    }
    assigned_counts_v7 = {
        lab: int(np.sum(hard_branch_idx == BRANCH_TO_INDEX[lab])) for lab in BRANCH_ORDER
    }

    q_state = (np.arange(nstate, dtype=np.float64) + 0.5) / nstate

    return {
        "eigenvalues": eigvals,
        "q_state": q_state,
        "branch_index": hard_branch_idx,
        "branch_index_canonical_argmax": canonical_argmax_idx,
        "branch_index_raw": raw_branch_idx,
        "max_overlap": canonical_max_overlap,
        "max_overlap_raw": raw_max_overlap,
        "assigned_overlap": hard_info["assigned_overlap"],
        "zero_candidate_mask": hard_info["zero_candidate_mask"],
        "zero_assignment_changed": hard_info["zero_assignment_changed"],
        "zero_assignment_pool_mismatch": np.int32(hard_info["pool_mismatch"]),
        "zero_assignment_objective": np.float64(hard_info["zero_objective"]),
        "zero_assignment_argmax_objective": np.float64(hard_info["zero_argmax_objective"]),
        "zero_assignment_regret": np.float64(hard_info["zero_regret"]),
        "zero_assignment_fractionality": np.float64(hard_info["zero_fractionality"]),
        "n_zero_reassigned": np.int32(hard_info["n_zero_reassigned"]),
        "branch_scores": canonical_scores.T,
        "branch_scores_canonical": canonical_scores.T,
        "branch_scores_raw": raw_scores.T,
        "assigned_counts": assigned_counts_v7,
        "assigned_counts_canonical_argmax": assigned_counts_argmax,
        "assigned_counts_raw": assigned_counts_raw,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "cluster_energy": cluster_energy,
        "is_degenerate": is_degenerate,
        "is_zero_cluster": is_zero_cluster,
        "canonicalized": canonicalized,
        "canonical_gauge_value": canonical_gauge_value,
        "canonical_purity_value": canonical_purity_value,
        "canonical_preorder_index": canonical_preorder_index,
        "cluster_purity_objective_before": cluster_purity_objective_before,
        "cluster_purity_objective_after": cluster_purity_objective_after,
        "cluster_joint_diag_sweeps": cluster_joint_diag_sweeps,
        "cluster_branch_runs": cluster_branch_runs,
        "max_canonical_order_violation": int(max_canonical_order_violation),
        "cluster_scores_sum": cluster_scores_sum,
        "cluster_scores_mean": cluster_scores_mean,
        "cluster_energies": cluster_energies,
        "cluster_sizes": cluster_sizes,
        "n_clusters": int(ncluster),
        "n_degenerate_clusters": int(np.sum(cluster_sizes > 1)),
        "n_degenerate_states": int(np.sum(is_degenerate)),
        "n_zero_degenerate_clusters": int(
            np.sum((cluster_sizes > 1) & (np.abs(cluster_energies) <= ZERO_CLUSTER_ENERGY_TOL))
        ),
        "n_canonicalized_states": int(np.sum(canonicalized)),
        "n_canonicalized_clusters": int(sum(np.any(canonicalized[inds]) for inds in clusters)),
        "mean_max_overlap": float(np.mean(canonical_max_overlap)),
        "mean_max_overlap_raw": float(np.mean(raw_max_overlap)),
        "mean_canonical_purity_gain": float(np.nanmean(
            cluster_purity_objective_after - cluster_purity_objective_before
        )) if np.any(np.isfinite(cluster_purity_objective_after)) else 0.0,
        "max_canonical_order_violation": int(max_canonical_order_violation),
        "completeness_error": completeness_error,
        "target_completeness_error": canonical_completeness_error,
        "cluster_completeness_error": cluster_completeness_error,
        "canonical_cluster_invariance_error": max_cluster_invariance_error,
        "eigenvector_orthogonality_error_raw": raw_ortho,
        "eigenvector_orthogonality_error": can_ortho,
        "reference_orthogonality_error": reference_orthogonality_error,
        "eigenpair_residual_error_raw": raw_resid,
        "eigenpair_residual_error": can_resid,
        "projector_rank_error": projector_rank_error,
        "projector_rank_errors": projector_rank_errors,
        "projector_trace": projector_trace_vec,
        "target_projector_rank_error": canonical_rank_error,
        "target_projector_rank_errors": canonical_rank_errors,
        "target_projector_trace": canonical_trace_vec,
        "eigenvector_qr_repaired": False,
        "qr_repair_delta": 0.0,
    }


# =============================================================================
# 5. FILE OUTPUT
# =============================================================================

def ensure_output_tree(outdir: Path):
    (outdir / "masks").mkdir(parents=True, exist_ok=True)
    (outdir / "states").mkdir(parents=True, exist_ok=True)


def write_metadata_json(outdir: Path):
    metadata = {
        "database_name": "fractal_ssh_topological_database_b3_b7_canonical_v7",
        "b_values": list(B_VALUES),
        "generation": GENERATION,
        "expected_family_counts": EXPECTED_COUNTS,
        "expected_total_families": EXPECTED_TOTAL,
        "finite_t1": FINITE_T1,
        "finite_t2": FINITE_T2,
        "reference_t1": REF_T1,
        "reference_t2": REF_T2,
        "branch_order_internal": BRANCH_ORDER,
        "branch_labels": [
            BRANCH_LABEL[x]
            for x in BRANCH_ORDER
        ],
        "hamiltonian_rule": "H_family = P H_parent P",
        "spectrum_representation": "variable length",
        "finite_projector_saved": False,
        "raw_state_projector_scores_saved": True,
        "ml_projector_targets": "continuous v6.1-style canonical projector scores, unchanged by v7 hard assignment",
        "canonical_basis": {
            "enabled": True,
            "definition": "maximize summed squared diagonal projector weights / minimize summed off-diagonal projector mixing",
            "objective": "max_U sum_{j,alpha} (u_j^T G_alpha u_j)^2, G_alpha=V_D^T P_alpha^ref V_D",
            "all_branches_symmetric_in_final_objective": True,
            "initializer_branch_codes": PURITY_INIT_BRANCH_CODES,
            "initializer_codes_role": "deterministic optimizer initialization only; not Hamiltonian, not loss, not labels",
            "rotation_abs_tol": CANONICAL_ROTATION_ABS_TOL,
            "rotation_rel_tol": CANONICAL_ROTATION_REL_TOL,
            "sequence_order": "within exact-degenerate cluster: branch order -> decreasing purity -> deterministic site moments",
            "hard_label_tie_break": "scores within 1e-10 abs/rel tolerance are tied; choose first fixed BRANCH_ORDER",
            "tie_breaker": "deterministic projected site-index moments; gauge only, not added to H",
        },
        "degeneracy_abs_tol": DEGENERACY_ABS_TOL,
        "degeneracy_rel_tol": DEGENERACY_REL_TOL,
        "finite_eigensolver": "scipy.linalg.eigh driver=evd",
        "numerical_validation": {
            "orthogonality_warning_tol": EIGENVECTOR_ORTHO_WARN_TOL,
            "orthogonality_hard_tol": EIGENVECTOR_ORTHO_HARD_TOL,
            "reference_orthogonality_tol": REFERENCE_ORTHO_TOL,
            "projector_completeness_tol": PROJECTOR_COMPLETENESS_TOL,
            "projector_rank_tol": PROJECTOR_RANK_TOL,
            "eigenpair_residual_tol": EIGENPAIR_RESIDUAL_TOL,
            "qr_repair_delta_tol": QR_REPAIR_DELTA_TOL,
        },
        "projector_rank_identity": (
            "sum_n C[n,alpha] = Tr(P_ref,alpha) = ref_dims[alpha]"
        ),
        "classification": (
            "Two distinct targets: continuous canonical projector scores, plus v7 rank-consistent hard labels. "
            "OC/IC-II/BULK-0 hard labels are obtained by exact-capacity maximum-overlap assignment; "
            "the six nonzero labels keep their stable projector argmax. No hard assignment modifies projector scores."
        ),
        "v7_zero_hard_assignment": {
            "branches": ["OC", "IC-II", "BULK-0"],
            "candidate_pool": "R0 smallest-|E| states; must exactly equal the stable zero-projector-argmax pool",
            "objective": "maximize sum_n C[n,label_n]",
            "capacities": "exact reference ranks for OC, IC-II, BULK-0",
            "solver": "scipy.optimize.linprog HiGHS transportation LP; integral solution checked explicitly",
            "tie_eps": ZERO_ASSIGN_TIE_EPS,
            "inference_note": "training hard head predicts freely; no exact-capacity decoder is used for primary test metrics",
        },
    }

    with (
        outdir / "database_metadata.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


def geometry_rows(records):
    rows = []

    for rec in records:
        rows.append({
            "family_id": rec.family_id,
            "b": rec.b,
            "family_index_within_b": rec.family_index,
            "N": rec.N,
            "Df": rec.Df,
            "generation": GENERATION,
            "n_cells_g2": rec.n_cells_g2,
            "n_sites": rec.n_sites,
            "n_eigenvalues": rec.n_sites,
            "mask_file": f"masks/{rec.family_id}.npz",
        })

    return rows


def save_geometry_database(
    outdir: Path,
    records,
):
    ensure_output_tree(outdir)
    write_metadata_json(outdir)

    # masks
    for rec in records:
        path = outdir / "masks" / f"{rec.family_id}.npz"

        np.savez_compressed(
            path,
            family_id=np.asarray(rec.family_id),
            b=np.int16(rec.b),
            N=np.int16(rec.N),
            Df=np.float64(rec.Df),
            generation=np.int16(GENERATION),
            n_cells_g2=np.int32(rec.n_cells_g2),
            n_sites=np.int32(rec.n_sites),
            M=rec.M.astype(np.uint8),
            G2=rec.G2.astype(np.uint8),
        )

    # geometry manifest
    rows = geometry_rows(records)

    manifest_path = (
        outdir / "geometry_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # counts by b
    count_path = outdir / "family_counts_by_b.csv"

    with count_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["b", "family_count"])

        for b in B_VALUES:
            writer.writerow(
                [b, EXPECTED_COUNTS[b]]
            )

        writer.writerow(
            ["total", EXPECTED_TOTAL]
        )


def state_file_complete(path: Path):
    if not path.exists():
        return False

    try:
        with np.load(path, allow_pickle=False) as z:
            required = {
                "eigenvalues",
                "q_state",
                "branch_index",
                "branch_index_canonical_argmax",
                "zero_candidate_mask",
                "zero_assignment_changed",
                "max_overlap",
                "branch_scores",
                "branch_scores_raw",
                "canonicalized",
                "canonical_gauge_value",
                "canonical_purity_value",
                "canonical_preorder_index",
                "cluster_purity_objective_before",
                "cluster_purity_objective_after",
                "cluster_joint_diag_sweeps",
                "cluster_branch_runs",
                "cluster_id",
                "cluster_size",
                "cluster_energy",
                "is_degenerate",
                "cluster_scores_sum",
                "cluster_scores_mean",
                "eigenvector_orthogonality_error",
                "reference_orthogonality_error",
                "projector_rank_error",
                "projector_rank_errors",
                "projector_trace",
            }

            return required.issubset(z.files)

    except Exception:
        return False


def process_spectrum_database(
    outdir: Path,
    records,
    only_b=None,
    max_sites=None,
    force=False,
):
    ensure_output_tree(outdir)

    # cache parent Hamiltonians by b
    finite_parent_cache = {}
    ref_parent_cache = {}

    status_rows = []

    # state manifest is appended incrementally
    state_manifest_path = outdir / "state_manifest.csv"
    write_state_header = (
        force
        or not state_manifest_path.exists()
    )

    if force and state_manifest_path.exists():
        state_manifest_path.unlink()
        write_state_header = True

    state_columns = [
        "family_id",
        "b",
        "N",
        "Df",
        "state_index",
        "q_state",
        "energy",
        "branch",
        "branch_index",
        "canonical_argmax_branch",
        "canonical_argmax_branch_index",
        "zero_assignment_changed",
        "assigned_overlap",
        "raw_branch",
        "raw_branch_index",
        "max_overlap",
        "max_overlap_raw",
        "cluster_id",
        "cluster_size",
        "cluster_energy",
        "is_degenerate",
        "is_zero_cluster",
        "canonicalized",
        "canonical_gauge_value",
        "canonical_purity_value",
        "canonical_preorder_index",
    ] + [
        f"score_{BRANCH_LABEL[lab]}"
        for lab in BRANCH_ORDER
    ] + [
        f"raw_score_{BRANCH_LABEL[lab]}"
        for lab in BRANCH_ORDER
    ]

    for k, rec in enumerate(records, start=1):

        if only_b is not None and rec.b not in only_b:
            continue

        if max_sites is not None and rec.n_sites > max_sites:
            status_rows.append({
                "family_id": rec.family_id,
                "b": rec.b,
                "N": rec.N,
                "n_sites": rec.n_sites,
                "status": "skipped_max_sites",
                "mean_max_overlap": "",
                "completeness_error": "",
                "eigenvector_orthogonality_error": "",
                "eigenpair_residual_error": "",
                "reference_orthogonality_error": "",
                "projector_rank_error": "",
                "eigenvector_qr_repaired": "",
                "qr_repair_delta": "",
            })
            continue

        state_path = (
            outdir
            / "states"
            / f"{rec.family_id}.npz"
        )

        if (
            not force
            and state_file_complete(state_path)
        ):
            print(
                f"[skip] {rec.family_id} already complete",
                flush=True,
            )
            status_rows.append({
                "family_id": rec.family_id,
                "b": rec.b,
                "N": rec.N,
                "n_sites": rec.n_sites,
                "status": "already_complete",
                "mean_max_overlap": "",
                "completeness_error": "",
                "eigenvector_orthogonality_error": "",
                "eigenpair_residual_error": "",
                "reference_orthogonality_error": "",
                "projector_rank_error": "",
                "eigenvector_qr_repaired": "",
                "qr_repair_delta": "",
            })
            continue

        side = rec.G2.shape[0]

        if rec.b not in finite_parent_cache:
            finite_parent_cache[rec.b] = (
                build_parent_sparse_ssh_hamiltonian(
                    side,
                    side,
                    FINITE_T1,
                    FINITE_T2,
                )
            )

        if rec.b not in ref_parent_cache:
            Href = (
                build_parent_sparse_ssh_hamiltonian(
                    side,
                    side,
                    REF_T1,
                    REF_T2,
                )
            )
            Href.eliminate_zeros()
            ref_parent_cache[rec.b] = Href

        print(
            f"[solve] {rec.family_id} "
            f"({k}/{len(records)}) "
            f"sites={rec.n_sites}",
            flush=True,
        )

        # ---------------------------------------------------------
        # reference
        # ---------------------------------------------------------
        reference = build_reference_from_parent(
            ref_parent_cache[rec.b],
            rec.G2,
        )

        # ---------------------------------------------------------
        # finite
        # ---------------------------------------------------------
        result = solve_finite_and_classify(
            finite_parent_cache[rec.b],
            rec.G2,
            reference,
        )

        comp = reference["component_counts"]

        # ---------------------------------------------------------
        # save per-family state file
        # ---------------------------------------------------------
        np.savez_compressed(
            state_path,

            family_id=np.asarray(rec.family_id),
            b=np.int16(rec.b),
            N=np.int16(rec.N),
            Df=np.float64(rec.Df),
            generation=np.int16(GENERATION),
            n_sites=np.int32(rec.n_sites),

            n_monomer=np.int32(comp["n_monomer"]),
            n_dimer=np.int32(comp["n_dimer"]),
            n_trimer=np.int32(comp["n_trimer"]),
            n_tetramer=np.int32(comp["n_tetramer"]),
            bulk_state_free=np.bool_(
                reference["bulk_state_free"]
            ),

            ref_dims=np.asarray(
                [
                    reference["ref_dims"][lab]
                    for lab in BRANCH_ORDER
                ],
                dtype=np.int32,
            ),

            assigned_counts=np.asarray(
                [
                    result["assigned_counts"][lab]
                    for lab in BRANCH_ORDER
                ],
                dtype=np.int32,
            ),

            assigned_counts_canonical_argmax=np.asarray(
                [
                    result["assigned_counts_canonical_argmax"][lab]
                    for lab in BRANCH_ORDER
                ],
                dtype=np.int32,
            ),

            assigned_counts_raw=np.asarray(
                [
                    result["assigned_counts_raw"][lab]
                    for lab in BRANCH_ORDER
                ],
                dtype=np.int32,
            ),

            eigenvalues=result["eigenvalues"].astype(
                np.float64
            ),
            q_state=result["q_state"].astype(
                np.float64
            ),
            branch_index=result["branch_index"].astype(np.int16),
            branch_index_canonical_argmax=result["branch_index_canonical_argmax"].astype(np.int16),
            branch_index_raw=result["branch_index_raw"].astype(np.int16),
            assigned_overlap=result["assigned_overlap"].astype(np.float64),
            zero_candidate_mask=result["zero_candidate_mask"].astype(np.bool_),
            zero_assignment_changed=result["zero_assignment_changed"].astype(np.bool_),
            zero_assignment_pool_mismatch=np.int32(result["zero_assignment_pool_mismatch"]),
            zero_assignment_objective=np.float64(result["zero_assignment_objective"]),
            zero_assignment_argmax_objective=np.float64(result["zero_assignment_argmax_objective"]),
            zero_assignment_regret=np.float64(result["zero_assignment_regret"]),
            zero_assignment_fractionality=np.float64(result["zero_assignment_fractionality"]),
            n_zero_reassigned=np.int32(result["n_zero_reassigned"]),
            max_overlap=result["max_overlap"].astype(np.float64),
            max_overlap_raw=result["max_overlap_raw"].astype(np.float64),
            branch_scores=result["branch_scores"].astype(np.float64),
            branch_scores_raw=result["branch_scores_raw"].astype(np.float64),
            branch_scores_canonical=result["branch_scores_canonical"].astype(np.float64),
            canonicalized=result["canonicalized"].astype(np.bool_),
            canonical_gauge_value=result["canonical_gauge_value"].astype(np.float64),
            canonical_purity_value=result["canonical_purity_value"].astype(np.float64),
            canonical_preorder_index=result["canonical_preorder_index"].astype(np.int32),

            cluster_id=result["cluster_id"].astype(np.int32),
            cluster_size=result["cluster_size"].astype(np.int32),
            cluster_energy=result["cluster_energy"].astype(np.float64),
            is_degenerate=result["is_degenerate"].astype(np.bool_),
            is_zero_cluster=result["is_zero_cluster"].astype(np.bool_),
            cluster_scores_sum=result["cluster_scores_sum"].astype(np.float64),
            cluster_scores_mean=result["cluster_scores_mean"].astype(np.float64),
            cluster_energies=result["cluster_energies"].astype(np.float64),
            cluster_sizes=result["cluster_sizes"].astype(np.int32),
            cluster_purity_objective_before=result["cluster_purity_objective_before"].astype(np.float64),
            cluster_purity_objective_after=result["cluster_purity_objective_after"].astype(np.float64),
            cluster_joint_diag_sweeps=result["cluster_joint_diag_sweeps"].astype(np.int16),
            cluster_branch_runs=result["cluster_branch_runs"].astype(np.int16),
            n_clusters=np.int32(result["n_clusters"]),
            n_degenerate_clusters=np.int32(result["n_degenerate_clusters"]),
            n_degenerate_states=np.int32(result["n_degenerate_states"]),
            n_zero_degenerate_clusters=np.int32(result["n_zero_degenerate_clusters"]),
            n_canonicalized_states=np.int32(result["n_canonicalized_states"]),
            n_canonicalized_clusters=np.int32(result["n_canonicalized_clusters"]),

            mean_max_overlap=np.float64(
                result["mean_max_overlap"]
            ),
            mean_canonical_purity_gain=np.float64(result["mean_canonical_purity_gain"]),
            max_canonical_order_violation=np.int16(result["max_canonical_order_violation"]),
            completeness_error=np.float64(
                result["completeness_error"]
            ),

            eigenvector_orthogonality_error_raw=np.float64(
                result["eigenvector_orthogonality_error_raw"]
            ),
            eigenvector_orthogonality_error=np.float64(
                result["eigenvector_orthogonality_error"]
            ),
            reference_orthogonality_error=np.float64(
                result["reference_orthogonality_error"]
            ),

            eigenpair_residual_error_raw=np.float64(
                result["eigenpair_residual_error_raw"]
            ),
            eigenpair_residual_error=np.float64(
                result["eigenpair_residual_error"]
            ),

            projector_rank_error=np.float64(
                result["projector_rank_error"]
            ),
            projector_rank_errors=result[
                "projector_rank_errors"
            ].astype(np.float64),
            projector_trace=result["projector_trace"].astype(np.float64),
            target_projector_rank_error=np.float64(result["target_projector_rank_error"]),
            target_projector_rank_errors=result["target_projector_rank_errors"].astype(np.float64),
            target_projector_trace=result["target_projector_trace"].astype(np.float64),
            target_completeness_error=np.float64(result["target_completeness_error"]),
            cluster_completeness_error=np.float64(result["cluster_completeness_error"]),
            canonical_cluster_invariance_error=np.float64(result["canonical_cluster_invariance_error"]),

            eigenvector_qr_repaired=np.bool_(
                result["eigenvector_qr_repaired"]
            ),
            qr_repair_delta=np.float64(
                result["qr_repair_delta"]
            ),
        )

        # ---------------------------------------------------------
        # append per-state manifest
        # ---------------------------------------------------------
        with state_manifest_path.open(
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=state_columns,
            )

            if write_state_header:
                writer.writeheader()
                write_state_header = False

            for n, energy in enumerate(
                result["eigenvalues"]
            ):
                bi = int(
                    result["branch_index"][n]
                )

                row = {
                    "family_id": rec.family_id,
                    "b": rec.b,
                    "N": rec.N,
                    "Df": rec.Df,
                    "state_index": n + 1,
                    "q_state": float(
                        result["q_state"][n]
                    ),
                    "energy": float(energy),
                    "branch": BRANCH_LABEL[
                        INDEX_TO_BRANCH[bi]
                    ],
                    "branch_index": bi,
                    "canonical_argmax_branch": BRANCH_LABEL[INDEX_TO_BRANCH[int(result["branch_index_canonical_argmax"][n])]],
                    "canonical_argmax_branch_index": int(result["branch_index_canonical_argmax"][n]),
                    "zero_assignment_changed": int(result["zero_assignment_changed"][n]),
                    "assigned_overlap": float(result["assigned_overlap"][n]),
                    "raw_branch": BRANCH_LABEL[INDEX_TO_BRANCH[int(result["branch_index_raw"][n])]],
                    "raw_branch_index": int(result["branch_index_raw"][n]),
                    "max_overlap": float(result["max_overlap"][n]),
                    "max_overlap_raw": float(result["max_overlap_raw"][n]),
                    "cluster_id": int(result["cluster_id"][n]),
                    "cluster_size": int(result["cluster_size"][n]),
                    "cluster_energy": float(result["cluster_energy"][n]),
                    "is_degenerate": int(result["is_degenerate"][n]),
                    "is_zero_cluster": int(result["is_zero_cluster"][n]),
                    "canonicalized": int(result["canonicalized"][n]),
                    "canonical_gauge_value": float(result["canonical_gauge_value"][n]) if np.isfinite(result["canonical_gauge_value"][n]) else "",
                    "canonical_purity_value": float(result["canonical_purity_value"][n]) if np.isfinite(result["canonical_purity_value"][n]) else "",
                    "canonical_preorder_index": int(result["canonical_preorder_index"][n]) if int(result["canonical_preorder_index"][n]) >= 0 else "",
                }

                for a, lab in enumerate(
                    BRANCH_ORDER
                ):
                    row[f"score_{BRANCH_LABEL[lab]}"] = float(
                        result["branch_scores"][n, a]
                    )
                    row[f"raw_score_{BRANCH_LABEL[lab]}"] = float(
                        result["branch_scores_raw"][n, a]
                    )

                writer.writerow(row)

        status_rows.append({
            "family_id": rec.family_id,
            "b": rec.b,
            "N": rec.N,
            "n_sites": rec.n_sites,
            "status": "complete",
            "mean_max_overlap": result[
                "mean_max_overlap"
            ],
            "completeness_error": result[
                "completeness_error"
            ],
            "eigenvector_orthogonality_error": result[
                "eigenvector_orthogonality_error"
            ],
            "eigenpair_residual_error": result[
                "eigenpair_residual_error"
            ],
            "reference_orthogonality_error": result[
                "reference_orthogonality_error"
            ],
            "projector_rank_error": result[
                "projector_rank_error"
            ],
            "eigenvector_qr_repaired": int(
                result["eigenvector_qr_repaired"]
            ),
            "qr_repair_delta": result[
                "qr_repair_delta"
            ],
        })

        print(
            "  components: "
            f"M={comp['n_monomer']} "
            f"D={comp['n_dimer']} "
            f"T={comp['n_trimer']} "
            f"Q={comp['n_tetramer']}",
            flush=True,
        )

        print(
            f"  mean max overlap = "
            f"{result['mean_max_overlap']:.6f}",
            flush=True,
        )

        ortho_flag = (
            "WARN"
            if result["eigenvector_orthogonality_error"]
            > EIGENVECTOR_ORTHO_WARN_TOL
            else "OK"
        )

        print(
            "  numerical checks: "
            f"eigvec_ortho="
            f"{result['eigenvector_orthogonality_error']:.3e}"
            f"({ortho_flag}), "
            f"residual="
            f"{result['eigenpair_residual_error']:.3e}, "
            f"ref_ortho="
            f"{result['reference_orthogonality_error']:.3e}, "
            f"completeness="
            f"{result['completeness_error']:.3e}, "
            f"rank="
            f"{result['projector_rank_error']:.3e}, "
            f"QR_repaired="
            f"{result['eigenvector_qr_repaired']}",
            flush=True,
        )

    # write current run status
    status_path = (
        outdir / "spectrum_status.csv"
    )

    if status_rows:
        with status_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    status_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(status_rows)


# =============================================================================
# 6. MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--outdir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "fractal_ssh_topological_database_b3_b7_canonical_v7"
        ),
    )

    p.add_argument(
        "--stage",
        choices=("geometry", "spectrum", "all"),
        default="all",
    )

    p.add_argument(
        "--only-b",
        type=int,
        nargs="*",
        default=None,
        help="Example: --only-b 3 4 5",
    )

    p.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help=(
            "Optional safety limit for spectrum stage. "
            "Example: --max-sites 1200"
        ),
    )

    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute spectrum files and rebuild "
            "state_manifest.csv"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Fractal SSH database v7 — canonical projector scores + rank-consistent zero hard labels")
    print("=" * 80)
    print(
        f"finite:    t1={FINITE_T1}, "
        f"t2={FINITE_T2}"
    )
    print(
        f"reference: t1={REF_T1}, "
        f"t2={REF_T2}"
    )
    print(
        "branch labels:",
        [BRANCH_LABEL[x] for x in BRANCH_ORDER],
    )
    print()

    print("[1] Enumerating strict families...")
    records = enumerate_all_families()

    print(
        f"    validated total = {len(records)}"
    )

    for b in B_VALUES:
        print(
            f"    b={b}: {EXPECTED_COUNTS[b]}"
        )

    if args.stage in ("geometry", "all"):
        print("[2] Writing geometry database...")
        save_geometry_database(
            args.outdir,
            records,
        )
        print("    geometry done")

    if args.stage in ("spectrum", "all"):
        # make sure geometry files also exist
        if not (
            args.outdir
            / "geometry_manifest.csv"
        ).exists():
            save_geometry_database(
                args.outdir,
                records,
            )

        only_b = (
            set(args.only_b)
            if args.only_b
            else None
        )

        print(
            "[3] Solving spectra + canonical projectors + "
            "v7 rank-constrained zero-sector hard assignment..."
        )

        process_spectrum_database(
            args.outdir,
            records,
            only_b=only_b,
            max_sites=args.max_sites,
            force=args.force,
        )

    print()
    print("DONE")
    print(f"Output: {args.outdir}")


if __name__ == "__main__":
    main()
