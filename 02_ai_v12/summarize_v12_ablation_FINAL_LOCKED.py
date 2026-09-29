# -*- coding: utf-8 -*-
"""Merge existing full-v12 E with A/B/C/D/F/G ablation test summaries.

Canonical paper comparison = each configuration's best_projector checkpoint.
E is read from the already-published/used work/run_v12 result and is not
retrained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NAME_TO_KEY = {
    "A_factorized_no_operator": "A",
    "B_factorized_K1K2": "B",
    "C_factorized_K1K2K3": "C",
    "D_full_matrices_no_hierarchy": "D",
    "F_flat_full_v12": "F",
    "G_full_no_cluster_loss": "G",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(path: Path, key: str, label: str, source: str):
    s = load_json(path)
    e = s.get("state_weighted_energy", {})
    p = s.get("primary_gauge_aware", {})
    v = s.get("v9_comparable_state_weighted", {})
    return {
        "config_key": key,
        "config": label,
        "source": source,
        "tag": path.stem.replace("test_summary_", ""),
        "checkpoint": s.get("checkpoint"),
        "checkpoint_epoch": s.get("checkpoint_epoch"),
        "operator_mode": s.get("operator_mode"),
        "projector_head_mode": s.get("projector_head_mode"),
        "num_test_families": s.get("num_test_families"),
        "num_test_states": s.get("num_test_states"),
        "energy_MAE": e.get("MAE"),
        "energy_RMSE": e.get("RMSE"),
        "energy_R2": e.get("pooled_R2"),
        "pred_chiral_maxerr": e.get("pred_chiral_maxerr"),
        "monotonic_violations": e.get("total_monotonic_violations"),
        "projector_MAE": p.get("state_projector_MAE_nondeg_or_highpurity"),
        "zero_parentage_MAE": p.get("state_zero_parentage_MAE_weighted_nondeg_or_highpurity"),
        "zero_mass_MAE": p.get("state_zero_mass_MAE_nondeg_or_highpurity"),
        "cluster_trace_MAE": p.get("degenerate_cluster_normalized_projector_trace_MAE"),
        "cluster_zero_parentage_MAE": p.get("degenerate_cluster_zero_parentage_MAE"),
        "cluster_zero_mass_MAE": p.get("degenerate_cluster_zero_mass_MAE"),
        "isolated6_macro_F1": p.get("isolated6_macro_F1"),
        "v9_projector_MAE": v.get("projector_score_MAE"),
        "v9_zero_parentage_MAE": v.get("zero_parentage_MAE_weighted"),
        "v9_zero_mass_MAE": v.get("zero_mass_MAE"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-run", type=Path, required=True,
                    help="Existing full-v12 E run (work/run_v12).")
    ap.add_argument("--ablation-root", type=Path, required=True,
                    help="New A/B/C/D/F/G root (work/ablation_v12).")
    a = ap.parse_args()

    main_run = a.main_run.resolve()
    root = a.ablation_root.resolve()
    rows = []

    # Existing E: include every available summary so all-checkpoints table stays complete.
    e_paths = sorted(main_run.glob("test_summary_*.json"))
    if not e_paths:
        raise FileNotFoundError(f"No existing E test summaries found in {main_run}")
    for p in e_paths:
        rows.append(flatten(p, "E", "E_full_v12_existing", "existing work/run_v12"))

    # Missing ablations A/B/C/D/F/G.
    for dirname, key in NAME_TO_KEY.items():
        rd = root / dirname
        if not rd.exists():
            continue
        for p in sorted(rd.glob("test_summary_*.json")):
            rows.append(flatten(p, key, dirname, "new ablation_v12"))

    if not rows:
        raise RuntimeError("No summaries found.")

    order = {k: i for i, k in enumerate("ABCDEFG")}
    df = pd.DataFrame(rows)
    df["_order"] = df["config_key"].map(order)
    df = df.sort_values(["_order", "config_key", "tag"]).drop(columns="_order")

    all_out = root / "ablation_A_to_G_all_checkpoints.csv"
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(all_out, index=False)

    canonical = df[df["tag"] == "best_projector"].copy()
    canonical["_order"] = canonical["config_key"].map(order)
    canonical = canonical.sort_values("_order").drop(columns="_order")

    # Guard against duplicates and incomplete final table.
    dup = canonical["config_key"].duplicated(keep=False)
    if dup.any():
        raise RuntimeError(
            "Duplicate best_projector rows detected for configs: "
            + ", ".join(canonical.loc[dup, "config_key"].tolist())
        )
    present = set(canonical["config_key"].tolist())
    missing = [k for k in "ABCDEFG" if k not in present]
    if missing:
        print("WARNING: final canonical table is incomplete; missing:", ", ".join(missing))

    canonical_out = root / "ablation_A_to_G_best_projector_full.csv"
    canonical.to_csv(canonical_out, index=False)

    compact_cols = [
        "config_key", "config", "checkpoint_epoch",
        "energy_MAE", "energy_RMSE", "energy_R2",
        "projector_MAE", "cluster_trace_MAE",
        "zero_parentage_MAE", "cluster_zero_parentage_MAE",
        "zero_mass_MAE", "cluster_zero_mass_MAE",
        "isolated6_macro_F1",
        "pred_chiral_maxerr", "monotonic_violations",
        "num_test_families", "num_test_states",
    ]
    compact = canonical[compact_cols].copy()
    compact_out = root / "ablation_A_to_G_best_projector_compact.csv"
    compact.to_csv(compact_out, index=False)

    print("\nCanonical A--G paper table (best_projector per configuration):\n")
    print(compact.to_string(index=False))
    print("\nSaved:")
    print(" ", all_out)
    print(" ", canonical_out)
    print(" ", compact_out)


if __name__ == "__main__":
    main()
