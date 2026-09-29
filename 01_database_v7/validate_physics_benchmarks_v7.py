# -*- coding: utf-8 -*-
"""Strict local physics + canonical-gauge checks for the v7 full-clean database.

Benchmarks
----------
1) Vicsek-cross G(2): 25 SSH cells / 100 sites, no tetramer-derived bulk.
   Reference ranks: [24, 8, 8, 20, 20, 20, 0, 0, 0].
2) Sierpinski G(1): 8 cells / 32 sites; 4 monomers, 8 dimers, 4 trimers.
3) Sierpinski G(2): 64 cells / 256 sites;
   4 monomers, 40 dimers, 36 trimers, 16 tetramers.
4) H_crop = P H_parent P agrees exactly with an independent direct builder.
5) Hermiticity, chiral symmetry and eigenpair residual checks.
6) Degenerate-subspace projector weights are invariant under arbitrary
   orthogonal rotations.
7) The canonicalization maps arbitrary input bases of an exact degenerate
   eigenspace to the same canonical projector-score sequence (up to signs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix

HERE = Path(__file__).resolve().parent
DB_SCRIPT = HERE / "build_fractal_ssh_topological_database_b3_b7_canonical_v7.py"


def load_db_module():
    spec = importlib.util.spec_from_file_location("fractal_db_v7", DB_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sierpinski_generator():
    M = np.ones((3, 3), dtype=np.uint8)
    M[1, 1] = 0
    return M


def vicsek_cross_generator():
    M = np.zeros((3, 3), dtype=np.uint8)
    M[0, 1] = M[1, 0] = M[1, 1] = M[1, 2] = M[2, 1] = 1
    return M


def direct_family_hamiltonian(mask, t1, t2):
    """Independent OBC implementation, used only as a cross-check."""
    ny, nx = mask.shape
    index = {}
    keys = []
    for cy in range(ny):
        for cx in range(nx):
            if not mask[cy, cx]:
                continue
            for sy in (0, 1):
                for sx in (0, 1):
                    index[(cx, cy, sx, sy)] = len(keys)
                    keys.append((cx, cy, sx, sy))

    rows, cols, vals = [], [], []
    def add(a, b, amp):
        if a not in index or b not in index:
            return
        i, j = index[a], index[b]
        rows.extend([i, j]); cols.extend([j, i]); vals.extend([amp, amp])

    for cy in range(ny):
        for cx in range(nx):
            if not mask[cy, cx]:
                continue
            for sy in (0, 1):
                add((cx, cy, 0, sy), (cx, cy, 1, sy), t1)
            for sx in (0, 1):
                add((cx, cy, sx, 0), (cx, cy, sx, 1), t1)
            if cx + 1 < nx and mask[cy, cx + 1]:
                for sy in (0, 1):
                    add((cx, cy, 1, sy), (cx + 1, cy, 0, sy), t2)
            if cy + 1 < ny and mask[cy + 1, cx]:
                for sx in (0, 1):
                    add((cx, cy, sx, 1), (cx, cy + 1, sx, 0), t2)
    H = coo_matrix((vals, (rows, cols)), shape=(len(keys), len(keys))).toarray()
    return H, keys


def chiral_error(H, keys):
    gamma = np.asarray([(-1.0) ** (2 * cx + sx + 2 * cy + sy) for cx, cy, sx, sy in keys])
    return float(np.max(np.abs((gamma[:, None] * H) * gamma[None, :] + H)))


def reference_for_mask(db, mask):
    side = mask.shape[0]
    parent = db.build_parent_sparse_ssh_hamiltonian(side, side, db.REF_T1, db.REF_T2)
    parent.eliminate_zeros()
    return db.build_reference_from_parent(parent, mask)


def projector_cluster_weights(reference, V):
    out = []
    for lab in db.BRANCH_ORDER:
        B = reference["branch_basis"][lab]
        if B.shape[1] == 0:
            out.append(0.0)
        else:
            out.append(float(np.sum(np.abs(B.conj().T @ V) ** 2)))
    return np.asarray(out)


def validate_case(db, name, mask, expected_components, expected_ref_dims, finite_t1=-0.1, finite_t2=-1.0, seed=20260909):
    report = {"name": name, "n_cells": int(mask.sum()), "n_sites": int(4 * mask.sum())}
    side = mask.shape[0]

    ref = reference_for_mask(db, mask)
    assert ref["component_counts"] == expected_components, (name, ref["component_counts"], expected_components)
    dims = np.asarray([ref["ref_dims"][lab] for lab in db.BRANCH_ORDER], dtype=int)
    assert np.array_equal(dims, np.asarray(expected_ref_dims, dtype=int)), (name, dims, expected_ref_dims)

    parent = db.build_parent_sparse_ssh_hamiltonian(side, side, finite_t1, finite_t2)
    Hproj, kept = db.project_family_hamiltonian(parent, mask)
    H = Hproj.toarray()
    Hdirect, keys = direct_family_hamiltonian(mask, finite_t1, finite_t2)
    crop_err = float(np.max(np.abs(H - Hdirect)))
    assert crop_err < 1e-14

    herm_err = float(np.max(np.abs(H - H.T)))
    assert herm_err < 1e-14
    chi_err = chiral_error(H, keys)
    assert chi_err < 1e-14

    E, V = linalg.eigh(H, check_finite=False, driver="evd")
    ortho = float(np.max(np.abs(V.T @ V - np.eye(len(E)))))
    residual = float(np.max(np.abs(H @ V - V * E[None, :])))
    spectral_sym = float(np.max(np.abs(E + E[::-1])))
    assert ortho < 1e-10 and residual < 1e-10 and spectral_sym < 1e-10

    cid, clusters = db.cluster_sorted_eigenvalues(E)
    rng = np.random.default_rng(seed)
    max_gauge_error = 0.0
    ndeg = 0
    max_target_internal_error = 0.0
    max_canonical_score_error = 0.0
    max_canonical_basis_error = 0.0
    max_purity_gain = 0.0
    max_branch_runs = 0
    ncanonical = 0
    n_canonical_gauge_trials = 0

    raw = np.zeros((len(E), len(db.BRANCH_ORDER)))
    for a, lab in enumerate(db.BRANCH_ORDER):
        B = ref["branch_basis"][lab]
        if B.shape[1]:
            raw[:, a] = np.sum(np.abs(B.T @ V) ** 2, axis=0)

    for inds in clusters:
        m = len(inds)
        target = raw[inds].sum(axis=0) / m
        repeated = np.repeat(target[None, :], m, axis=0)
        max_target_internal_error = max(max_target_internal_error, float(np.max(np.abs(repeated - repeated[0]))))
        if m <= 1:
            continue
        ndeg += 1
        VD = V[:, inds]
        w0 = projector_cluster_weights(ref, VD)

        ebar = float(np.mean(E[inds]))
        spread = float(np.ptp(E[inds]))
        rot_tol = max(
            db.CANONICAL_ROTATION_ABS_TOL,
            db.CANONICAL_ROTATION_REL_TOL * max(1.0, abs(ebar)),
        )
        if spread <= rot_tol:
            ncanonical += 1
            C0, info0 = db._canonicalize_degenerate_block(VD, ref, kept)
            S0 = np.zeros((m, len(db.BRANCH_ORDER)))
            for aa, lab in enumerate(db.BRANCH_ORDER):
                B = ref["branch_basis"][lab]
                if B.shape[1]:
                    S0[:, aa] = np.sum(np.abs(B.T @ C0) ** 2, axis=0)

            # Stronger gauge stress test than v6.0/v6.0.1.  Large degenerate
            # blocks receive more independent random rotations because this is
            # exactly where flat purity sectors can hide.
            ntrial = 8 if m > 2 else 3
            for _ in range(ntrial):
                A = rng.normal(size=(m, m))
                Q, _ = np.linalg.qr(A)
                w1 = projector_cluster_weights(ref, VD @ Q)
                max_gauge_error = max(max_gauge_error, float(np.max(np.abs(w0 - w1))))

                C1, _ = db._canonicalize_degenerate_block(VD @ Q, ref, kept)
                S1 = np.zeros_like(S0)
                for aa, lab in enumerate(db.BRANCH_ORDER):
                    B = ref["branch_basis"][lab]
                    if B.shape[1]:
                        S1[:, aa] = np.sum(np.abs(B.T @ C1) ** 2, axis=0)
                max_canonical_score_error = max(
                    max_canonical_score_error, float(np.max(np.abs(S0 - S1)))
                )
                O = np.abs(C0.T @ C1)
                max_canonical_basis_error = max(
                    max_canonical_basis_error, float(np.max(np.abs(O - np.eye(m))))
                )
                n_canonical_gauge_trials += 1

            if info0["purity_objective_after"] + 1e-10 < info0["purity_objective_before"]:
                raise AssertionError("canonical purity objective decreased")
            bseq = db.stable_branch_index(S0)
            if np.any(np.diff(bseq) < 0):
                raise AssertionError("canonical argmax branch sequence is not ordered")
            max_purity_gain = max(max_purity_gain, float(info0["purity_objective_after"] - info0["purity_objective_before"]))
            runs = int(1 + np.sum(bseq[1:] != bseq[:-1])) if m else 0
            max_branch_runs = max(max_branch_runs, runs)
        else:
            # Subspace-weight invariance is still checked once for clusters that
            # are degenerate by the spectral clustering tolerance but are not
            # strict enough to rotate/canonicalize.
            A = rng.normal(size=(m, m))
            Q, _ = np.linalg.qr(A)
            w1 = projector_cluster_weights(ref, VD @ Q)
            max_gauge_error = max(max_gauge_error, float(np.max(np.abs(w0 - w1))))

    print(
        f"[HOTFIX CHECK] {name}: gauge_err={max_gauge_error:.3e}, "
        f"canonical_score_err={max_canonical_score_error:.3e}, "
        f"canonical_basis_err={max_canonical_basis_error:.3e}, "
        f"purity_gain_max={max_purity_gain:.3e}"
    )
    assert max_gauge_error < 1e-10, f"{name}: gauge invariance failed: {max_gauge_error:.3e}"
    assert max_canonical_score_error < 1e-9, (
        f"{name}: canonical score gauge test failed: {max_canonical_score_error:.3e}"
    )
    assert max_canonical_basis_error < 1e-7, (
        f"{name}: canonical basis gauge test failed: {max_canonical_basis_error:.3e}"
    )

    # ------------------------------------------------------------------
    # v7 hard-label definition audit. This is intentionally separate from
    # the continuous projector-score physics above.
    # ------------------------------------------------------------------
    solved = db.solve_finite_and_classify(parent, mask, ref)
    hard = solved["branch_index"].astype(np.int64)
    argmax_hard = solved["branch_index_canonical_argmax"].astype(np.int64)
    counts = np.bincount(hard, minlength=len(db.BRANCH_ORDER)).astype(np.int64)
    if not np.array_equal(counts, dims):
        raise AssertionError(f"{name}: v7 hard counts {counts.tolist()} != ref dims {dims.tolist()}")
    if int(solved["zero_assignment_pool_mismatch"]) != 0:
        raise AssertionError(f"{name}: zero candidate pool mismatch")
    if float(solved["zero_assignment_fractionality"]) > db.ZERO_ASSIGN_INTEGRAL_TOL:
        raise AssertionError(f"{name}: fractional transportation solution")

    # Permuting rows of the zero-sector LP must not change the final assignment
    # after mapping back to the immutable global eigenstate indices.
    zstates = np.flatnonzero(solved["zero_candidate_mask"]).astype(np.int64)
    zcaps = dims[db.ZERO_BRANCH_INDICES.astype(np.int64)]
    zscores = solved["branch_scores"][zstates][:, db.ZERO_BRANCH_INDICES.astype(np.int64)]
    lab0, _ = db._capacity_assignment_three(zscores, zcaps, zstates)
    if len(zstates) > 1:
        perm = rng.permutation(len(zstates))
        labp, _ = db._capacity_assignment_three(zscores[perm], zcaps, zstates[perm])
        back = np.empty_like(labp)
        back[perm] = labp
        if not np.array_equal(lab0, back):
            raise AssertionError(f"{name}: v7 zero assignment depends on row order")

    report.update({
        "v7_hard_counts": counts.tolist(),
        "v7_argmax_counts": np.bincount(argmax_hard, minlength=len(db.BRANCH_ORDER)).astype(int).tolist(),
        "v7_n_zero_reassigned": int(solved["n_zero_reassigned"]),
        "v7_zero_assignment_regret": float(solved["zero_assignment_regret"]),
        "v7_zero_assignment_fractionality": float(solved["zero_assignment_fractionality"]),
        "v7_zero_pool_mismatch": int(solved["zero_assignment_pool_mismatch"]),
    })

    report.update({
        "component_counts": ref["component_counts"],
        "ref_dims": dims.tolist(),
        "crop_projection_error": crop_err,
        "hermiticity_error": herm_err,
        "chiral_error": chi_err,
        "eigenvector_orthogonality_error": ortho,
        "eigenpair_residual_maxabs": residual,
        "spectral_chiral_symmetry_error": spectral_sym,
        "n_clusters": len(clusters),
        "n_degenerate_clusters": ndeg,
        "max_gauge_rotation_error": max_gauge_error,
        "max_invariant_target_internal_error": max_target_internal_error,
        "n_canonicalizable_clusters": ncanonical,
        "n_canonical_gauge_trials": n_canonical_gauge_trials,
        "max_canonical_score_rotation_error": max_canonical_score_error,
        "max_canonical_basis_rotation_error": max_canonical_basis_error,
        "max_projector_purity_gain": max_purity_gain,
        "max_canonical_branch_runs_per_cluster": max_branch_runs,
        "zero_eigenvalue_count_1e-10": int(np.sum(np.abs(E) < 1e-10)),
        "energy_min": float(E[0]),
        "energy_max": float(E[-1]),
    })
    return report


def main():
    global db
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "benchmark_report_v7.json")
    args = ap.parse_args()
    db = load_db_module()

    # Cross-platform hard-label regression: tiny BLAS/LAPACK perturbations must
    # not flip a projector tie. Fixed BRANCH_ORDER resolves all near ties.
    tie0 = np.zeros((4, len(db.BRANCH_ORDER)), dtype=np.float64)
    tie0[0, [0, 4]] = [0.5 + 4e-13, 0.5 - 4e-13]
    tie0[1, [0, 4]] = [0.5 - 4e-13, 0.5 + 4e-13]
    tie0[2, [4, 7]] = [0.25 + 3e-13, 0.25 - 3e-13]
    tie0[3, [4, 7]] = [0.25 - 3e-13, 0.25 + 3e-13]
    tie_idx = db.stable_branch_index(tie0)
    expected_tie_idx = np.asarray([0, 0, 4, 4], dtype=np.int16)
    if not np.array_equal(tie_idx, expected_tie_idx):
        raise AssertionError(f"stable branch tie-break regression failed: {tie_idx}")
    print("[V7 TIE CHECK] PASS — near-equal projector scores use fixed BRANCH_ORDER")

    S1 = sierpinski_generator()
    S2 = np.kron(S1, S1).astype(np.uint8)
    V1 = vicsek_cross_generator()
    V2 = np.kron(V1, V1).astype(np.uint8)

    print("*" * 96)
    print("*** V7 FULL CLEAN ACTIVE — RANK-CONSISTENT ZERO ASSIGNMENT + TWO-TARGET PHYSICS ***")
    print("*** 若看到这一行，说明不是旧版 v6。                                  ***")
    print("*" * 96)

    reports = []
    reports.append(validate_case(
        db, "Sierpinski_G1", S1,
        {"n_monomer":4,"n_dimer":8,"n_trimer":4,"n_tetramer":0},
        [4,8,8,4,4,4,0,0,0],
    ))
    reports.append(validate_case(
        db, "Sierpinski_G2", S2,
        {"n_monomer":4,"n_dimer":40,"n_trimer":36,"n_tetramer":16},
        [4,40,40,36,36,36,16,32,16],
    ))
    reports.append(validate_case(
        db, "VicsekCross_G2", V2,
        {"n_monomer":24,"n_dimer":8,"n_trimer":20,"n_tetramer":0},
        [24,8,8,20,20,20,0,0,0],
    ))

    obj = {
        "status": "PASS",
        "finite_parameters": {"t1": -0.1, "t2": -1.0},
        "reference_parameters": {"t1": 0.0, "t2": -1.0},
        "branch_order": [db.BRANCH_LABEL[x] for x in db.BRANCH_ORDER],
        "cases": reports,
    }
    args.out.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 84)
    print("*** CANONICAL-BASIS v7 FULL CLEAN PHYSICS BENCHMARK: PASS ***")
    print("=" * 84)
    for r in reports:
        print(
            f"{r['name']:18s} cells={r['n_cells']:3d} sites={r['n_sites']:4d} "
            f"components={r['component_counts']} zero={r['zero_eigenvalue_count_1e-10']:3d} "
            f"gauge_err={r['max_gauge_rotation_error']:.3e} can_err={r['max_canonical_score_rotation_error']:.3e} "
            f"purity_gain={r['max_projector_purity_gain']:.3e} runs<={r['max_canonical_branch_runs_per_cluster']} reassigned={r['v7_n_zero_reassigned']}"
        )
    print("Report:", args.out)


if __name__ == "__main__":
    main()
