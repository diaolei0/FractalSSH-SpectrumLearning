# -*- coding: utf-8 -*-
"""Locked paper-final ablation runner for Fractal SSH v12.

Purpose
-------
Run ONLY the six missing controlled ablations A/B/C/D/F/G while reusing the
already-trained full-v12 model as configuration E.  The script is locked to the
existing project layout:

    Fractal_SSH_COMPLETE_FROM_SCRATCH_R1/
      02_ai_v12/                         <- put this script here
      work/
        database_v7/                     <- existing 816-family database
        run_v12/                         <- existing full v12 (configuration E)
          operator_cache/
        ablation_v12/                    <- new A/B/C/D/F/G outputs

Scientific safeguards
---------------------
* E is NEVER retrained by this script.
* A/B/C/D/F/G keep the original RUN_V12_ABLATIONS.py definitions unchanged.
* The exact E family_split.json and scalers.json are copied into every fresh
  ablation run, so all variants use the identical 652/81/83 split and scaling.
* The current core source files are SHA-256 checked against E/run_metadata.json.
  A mismatch aborts by default.
* Training settings are explicitly locked to the formal E run (seed 20260912,
  140 epochs target, lr 3e-5, etc.).
* Fresh training is the default.  Existing last_model.pt causes an abort unless
  --resume is explicitly supplied.
* Each variant is tested at the same six specialized checkpoints as the main
  v12 pipeline.
* At the end, summarize_v12_ablation_FINAL_LOCKED.py merges the existing E
  results with A/B/C/D/F/G into paper-ready CSV files.

Recommended commands
--------------------
1) Validate paths/configuration only (no training):
   python RUN_V12_ABLATIONS_FINAL_LOCKED.py --dry-run

2) Formal paper run:
   python RUN_V12_ABLATIONS_FINAL_LOCKED.py

3) Resume after an interruption of the SAME formal run:
   python RUN_V12_ABLATIONS_FINAL_LOCKED.py --resume

4) Run selected variants only, e.g. A and B:
   python RUN_V12_ABLATIONS_FINAL_LOCKED.py --configs A,B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
WORK = PROJECT_ROOT / "work"
MODEL = HERE / "fractal_ssh_full_spectrum_operator_v12.py"
SUMMARY = HERE / "summarize_v12_ablation_FINAL_LOCKED.py"
PY = Path(sys.executable)

DEFAULT_DB = WORK / "database_v7"
DEFAULT_MAIN_RUN = WORK / "run_v12"
DEFAULT_OUTPUT = WORK / "ablation_v12"

# Same six missing configurations as the original RUN_V12_ABLATIONS.py.
# E is intentionally absent: existing work/run_v12 is the canonical E.
CONFIGS = {
    "A": ("A_factorized_no_operator", "none", "factorized", []),
    "B": ("B_factorized_K1K2", "k1k2", "factorized", []),
    "C": ("C_factorized_K1K2K3", "k1k2k3", "factorized", []),
    "D": ("D_full_matrices_no_hierarchy", "full_no_hierarchy", "factorized", []),
    "F": ("F_flat_full_v12", "full", "flat", []),
    "G": (
        "G_full_no_cluster_loss",
        "full",
        "factorized",
        [
            "--cluster-trace-weight", "0",
            "--zero-cluster-parentage-weight", "0",
            "--zero-cluster-mass-weight", "0",
        ],
    ),
}

CHECKPOINTS = [
    ("best_model.pt", "best_model"),
    ("best_projector_model.pt", "best_projector"),
    ("best_zero_parentage_model.pt", "best_zero_parentage"),
    ("best_zero_cluster_model.pt", "best_zero_cluster"),
    ("best_v9_comparable_zero_model.pt", "best_v9_comparable_zero"),
    ("best_cluster_trace_model.pt", "best_cluster_trace"),
]

CORE_HASH_FILES = [
    "fractal_ssh_full_spectrum_operator_v12.py",
    "model_blocks_v12.py",
    "v9_common.py",
    "operator_cache_v12.py",
    "operator_core_v12.py",
    "operator_features_v12.py",
    "validate_operator_informed_v12.py",
    "run_full_pipeline_operator_v12_REUSE_V7_DB.py",
]

EXPECTED_E = {
    "seed": 20260912,
    "num_families": 816,
    "num_train": 652,
    "num_val": 81,
    "num_test": 83,
    "operator_mode": "full",
    "projector_head_mode": "factorized",
    "epochs": 140,
    "lr": 3e-5,
    "min_lr": 1e-6,
    "weight_decay": 1e-5,
    "lr_patience": 7,
    "early_stop_patience": 28,
    "grad_accum_steps": 4,
    "grad_clip": 1.0,
    "balance_strength": 0.35,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_json(a: Path, b: Path) -> bool:
    return load_json(a) == load_json(b)


def parse_configs(raw: str):
    keys = [x.strip().upper() for x in raw.replace(";", ",").split(",") if x.strip()]
    if not keys:
        raise ValueError("--configs is empty")
    bad = [x for x in keys if x not in CONFIGS]
    if bad:
        raise ValueError(f"Unknown configs {bad}. Allowed: A,B,C,D,F,G. E is existing and is not retrained.")
    out = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def ensure_db_complete(db: Path):
    man = db / "geometry_manifest.csv"
    states = db / "states"
    if not man.exists() or not states.is_dir():
        raise FileNotFoundError(
            f"Database does not look complete: {db}\n"
            "Expected geometry_manifest.csv and states/."
        )
    n_states = sum(1 for _ in states.glob("*.npz"))
    if n_states < 816:
        raise RuntimeError(f"Database has only {n_states} state files; expected at least 816.")


def validate_existing_E(main_run: Path, cache: Path, allow_code_mismatch: bool):
    required = [
        main_run / "run_metadata.json",
        main_run / "family_split.json",
        main_run / "scalers.json",
        main_run / "test_summary_best_projector.json",
        main_run / "best_projector_model.pt",
        cache / "operator_cache_audit.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Existing E is incomplete. Missing:\n" + "\n".join(map(str, missing)))

    meta = load_json(main_run / "run_metadata.json")
    args = meta.get("args", {})
    checks = {
        "seed": meta.get("seed"),
        "num_families": meta.get("num_families"),
        "num_train": meta.get("num_train"),
        "num_val": meta.get("num_val"),
        "num_test": meta.get("num_test"),
        "operator_mode": args.get("operator_mode"),
        "projector_head_mode": args.get("projector_head_mode"),
        "epochs": args.get("epochs"),
        "lr": args.get("lr"),
        "min_lr": args.get("min_lr"),
        "weight_decay": args.get("weight_decay"),
        "lr_patience": args.get("lr_patience"),
        "early_stop_patience": args.get("early_stop_patience"),
        "grad_accum_steps": args.get("grad_accum_steps"),
        "grad_clip": args.get("grad_clip"),
        "balance_strength": args.get("balance_strength"),
    }
    bad = {k: (checks.get(k), v) for k, v in EXPECTED_E.items() if checks.get(k) != v}
    if bad:
        raise RuntimeError(f"Existing E metadata does not match the locked formal configuration: {bad}")

    split = load_json(main_run / "family_split.json")
    lengths = {k: len(split.get(k, [])) for k in ("train", "val", "test")}
    if lengths != {"train": 652, "val": 81, "test": 83}:
        raise RuntimeError(f"Unexpected E split lengths: {lengths}")

    recorded = meta.get("code_sha256", {})
    mismatches = []
    for name in CORE_HASH_FILES:
        p = HERE / name
        if not p.exists():
            mismatches.append((name, "MISSING", recorded.get(name)))
            continue
        actual = sha256(p)
        expected = recorded.get(name)
        if expected and actual != expected:
            mismatches.append((name, actual, expected))
    if mismatches and not allow_code_mismatch:
        lines = ["Core code SHA-256 differs from the code used for existing E:"]
        lines += [f"  {n}\n    current={a}\n    E_used ={e}" for n, a, e in mismatches]
        lines.append("Abort to preserve a controlled comparison. Use --allow-code-mismatch only if intentional.")
        raise RuntimeError("\n".join(lines))

    es = load_json(main_run / "test_summary_best_projector.json")
    if es.get("num_test_families") != 83 or es.get("num_test_states") != 332988:
        raise RuntimeError("Existing E best_projector summary does not match 83 families / 332,988 states.")

    return meta, es, mismatches


def prepare_exact_split_and_scalers(main_run: Path, rd: Path):
    """Copy exact E split/scalers into a fresh run, or verify an existing run."""
    for name in ("family_split.json", "scalers.json"):
        src = main_run / name
        dst = rd / name
        if dst.exists():
            if not same_json(src, dst):
                raise RuntimeError(
                    f"{rd.name}: existing {name} differs from E.\n"
                    "Use a fresh run directory; do not mix incompatible experiments."
                )
        else:
            shutil.copy2(src, dst)


def validate_run_after_train(rd: Path, main_meta: dict, key: str, op: str, head: str):
    p = rd / "run_metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"{key}: training did not create run_metadata.json: {p}")
    m = load_json(p)
    a = m.get("args", {})
    expected_split = main_meta.get("split_sha256")
    if m.get("split_sha256") != expected_split:
        raise RuntimeError(
            f"{key}: split SHA mismatch. E={expected_split}, {key}={m.get('split_sha256')}"
        )
    if m.get("seed") != 20260912:
        raise RuntimeError(f"{key}: unexpected seed {m.get('seed')}")
    if a.get("operator_mode") != op or a.get("projector_head_mode") != head:
        raise RuntimeError(
            f"{key}: run metadata mode mismatch: operator={a.get('operator_mode')}, head={a.get('projector_head_mode')}"
        )
    if int(a.get("epochs", -1)) != 140:
        raise RuntimeError(f"{key}: formal run must target 140 epochs, got {a.get('epochs')}")


def run_command(cmd, log_path: Path, label: str):
    print("\n" + "=" * 110)
    print(label)
    print("=" * 110)
    print(">", " ".join(map(str, cmd)), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {label}\n")
        log.write("> " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = p.wait()
    if rc:
        raise RuntimeError(f"{label} failed with return code {rc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-root", type=Path, default=DEFAULT_DB)
    ap.add_argument("--main-run", type=Path, default=DEFAULT_MAIN_RUN,
                    help="Existing full-v12 E run; it will be reused, never retrained.")
    ap.add_argument("--operator-cache", type=Path, default=None,
                    help="Default: <main-run>/operator_cache")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="",
                    help="Blank uses model default; e.g. cuda:0. Use the same environment/GPU setup as E.")
    ap.add_argument("--configs", default="A,B,C,D,F,G")
    ap.add_argument("--resume", action="store_true",
                    help="Resume existing last_model.pt for the SAME configuration. Fresh is default.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only validate files, E metadata, code hashes, paths and planned commands.")
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--allow-code-mismatch", action="store_true")
    a = ap.parse_args()

    db = a.db_root.expanduser().resolve()
    main_run = a.main_run.expanduser().resolve()
    cache = (a.operator_cache.expanduser().resolve() if a.operator_cache else main_run / "operator_cache")
    out_root = a.output_root.expanduser().resolve()
    selected = parse_configs(a.configs)

    if not MODEL.exists():
        raise FileNotFoundError(f"Model script not found: {MODEL}")
    ensure_db_complete(db)
    main_meta, e_summary, mismatches = validate_existing_E(main_run, cache, a.allow_code_mismatch)

    print("\n" + "=" * 110)
    print("FRACTAL SSH v12 — FINAL LOCKED ABLATION PLAN")
    print("Python executable :", PY)
    print("Project root      :", PROJECT_ROOT)
    print("Database          :", db)
    print("Existing E run    :", main_run, "(REUSE; DO NOT RETRAIN)")
    print("Operator cache    :", cache)
    print("New output root   :", out_root)
    print("Configs to train  :", ", ".join(selected))
    print("Seed              : 20260912")
    print("Epoch target      : 140")
    print("E split SHA-256   :", main_meta.get("split_sha256"))
    print("E best-projector  : epoch", e_summary.get("checkpoint_epoch"))
    print("E test families   :", e_summary.get("num_test_families"))
    print("E test states     :", e_summary.get("num_test_states"))
    if mismatches:
        print("WARNING: code mismatches were explicitly allowed:", [x[0] for x in mismatches])
    print("=" * 110)

    if a.dry_run:
        print("\nDRY RUN PASSED. No training/test files were written.")
        print("Next command for the formal run:")
        print(f'  "{PY}" "{Path(__file__).resolve()}"')
        return

    out_root.mkdir(parents=True, exist_ok=True)
    master_log = out_root / "ABLATION_FINAL_LOCKED.log"
    device = (["--device", a.device] if a.device else [])

    # Explicitly mirror the formal E training hyperparameters.
    locked_train_args = [
        "--seed", "20260912",
        "--epochs", "140",
        "--lr", "3e-5",
        "--min-lr", "1e-6",
        "--weight-decay", "1e-5",
        "--lr-patience", "7",
        "--early-stop-patience", "28",
        "--grad-accum-steps", "4",
        "--grad-clip", "1.0",
        "--balance-strength", "0.35",
        "--progress-every", "50",
    ]

    for key in selected:
        name, op, head, extra = CONFIGS[key]
        rd = out_root / name
        rd.mkdir(parents=True, exist_ok=True)

        last = rd / "last_model.pt"
        if last.exists() and not a.resume:
            raise RuntimeError(
                f"{key}: {last} already exists.\n"
                "For a clean controlled run, the default is to refuse accidental reuse.\n"
                "If this is the SAME interrupted 140-epoch formal run, rerun with --resume.\n"
                "Otherwise rename/remove the old configuration directory yourself."
            )
        if a.resume and not last.exists():
            print(f"[{key}] --resume requested but no last_model.pt exists; starting fresh.")

        prepare_exact_split_and_scalers(main_run, rd)

        common = [
            "--db-root", db,
            "--run-dir", rd,
            "--operator-cache", cache,
            "--operator-mode", op,
            "--projector-head-mode", head,
            "--seed", "20260912",
            *device,
        ]

        if not a.skip_check:
            run_command(
                [PY, MODEL, "--mode", "check", *common, *extra],
                master_log,
                f"[{key}] {name} — wiring check",
            )

        train = [
            PY, MODEL, "--mode", "train",
            "--db-root", db,
            "--run-dir", rd,
            "--operator-cache", cache,
            "--operator-mode", op,
            "--projector-head-mode", head,
            *device,
            *locked_train_args,
            *extra,
        ]
        if last.exists() and a.resume:
            train.append("--resume")

        run_command(train, master_log, f"[{key}] {name} — formal training (target 140 epochs)")
        validate_run_after_train(rd, main_meta, key, op, head)

        if not a.skip_tests:
            for ck, tag in CHECKPOINTS:
                cp = rd / ck
                if not cp.exists():
                    raise FileNotFoundError(f"{key}: expected checkpoint missing after training: {cp}")
                run_command(
                    [
                        PY, MODEL, "--mode", "test",
                        "--db-root", db,
                        "--run-dir", rd,
                        "--operator-cache", cache,
                        "--checkpoint-name", ck,
                        "--test-tag", tag,
                        "--seed", "20260912",
                        *device,
                    ],
                    master_log,
                    f"[{key}] {name} — test {tag}",
                )

    if not a.no_summary and not a.skip_tests:
        if not SUMMARY.exists():
            raise FileNotFoundError(
                f"Summarizer not found: {SUMMARY}\n"
                "Put summarize_v12_ablation_FINAL_LOCKED.py next to this runner."
            )
        run_command(
            [PY, SUMMARY, "--main-run", main_run, "--ablation-root", out_root],
            master_log,
            "Merge existing E with A/B/C/D/F/G and build paper tables",
        )

    print("\n" + "=" * 110)
    print("FINAL ABLATION RUN FINISHED")
    print("E reused from      :", main_run)
    print("A/B/C/D/F/G output :", out_root)
    if not a.no_summary and not a.skip_tests:
        print("Paper compact table:", out_root / "ablation_A_to_G_best_projector_compact.csv")
        print("All checkpoints    :", out_root / "ablation_A_to_G_all_checkpoints.csv")
    print("=" * 110)


if __name__ == "__main__":
    main()
