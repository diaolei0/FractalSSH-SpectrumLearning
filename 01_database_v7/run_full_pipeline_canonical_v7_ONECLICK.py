# -*- coding: utf-8 -*-
"""
Fractal SSH Canonical v7 FULL CLEAN — 一键从头完整运行

核心变化：
  A) 连续 projector target 与 hard branch target 分开；
  B) OC / IC-II / BULK-0 的 hard truth 使用 exact-rank maximum-overlap assignment；
  C) 模型使用 projector head + hard branch head 两个独立输出头；
  D) primary test hard metrics 由 hard head 自由预测，不使用 capacity decoder；
  E) 首次运行完全随机初始化，不读取 v5/v6 checkpoint。

流程：
  1) 严格物理 + canonicalization + v7 assignment benchmark
  2) 从头生成/续跑 816-family v7 数据库
  3) 数据库与双头模型检查
  4) 从随机初始化训练完整 v7
  5) 测试 best_model.pt 并输出九分支指标

直接在 PyCharm 里运行本文件即可。
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = Path(sys.executable)

DB = HERE / "fractal_ssh_topological_database_b3_b7_canonical_v7"
RUN = HERE / "run_ms_cnn_transformer_canonical_v7"

DB_SCRIPT = HERE / "build_fractal_ssh_topological_database_b3_b7_canonical_v7.py"
MODEL_SCRIPT = HERE / "fractal_ssh_multiscale_cnn_transformer_canonical_v7.py"
BENCH_SCRIPT = HERE / "validate_physics_benchmarks_v7.py"
LOG = HERE / "one_click_v7.log"

# 正式默认参数：从头跑时建议先不要改。
EPOCHS = 160
BATCH_SIZE = 8
QUERIES_PER_FAMILY = 96
LR = 3e-5
NUM_WORKERS = 0


def stamp(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def banner() -> None:
    print("\n" + "*" * 104)
    print("***                                                                                                  ***")
    print("***   FRACTAL SSH CANONICAL v7 FULL CLEAN — RANK-CONSISTENT ZERO LABELS + DUAL HEAD                  ***")
    print("***                                                                                                  ***")
    print("***   *** FROM SCRATCH: 首次运行不加载任何 v5 / v6 checkpoint ***                                  ***")
    print("***   Soft target : continuous reference-projector composition                                      ***")
    print("***   Hard target : OC / IC-II / BULK-0 exact-rank maximum-overlap canonical assignment             ***")
    print("***   Model       : projector head 与 hard-branch head 完全分开                                      ***")
    print("***   Test        : hard head 自由预测；PRIMARY METRIC 不使用 exact-capacity decoder                  ***")
    print("***   Hamiltonian / t1,t2 / 816 geometries / reference projectors 全部不改                           ***")
    print("***                                                                                                  ***")
    print("*" * 104 + "\n")


def run(cmd: list[str], title: str) -> None:
    stamp("=" * 92)
    stamp(title)
    stamp("=" * 92)
    stamp("$ " + " ".join(map(str, cmd)))
    with LOG.open("a", encoding="utf-8") as f:
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(HERE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"{title} 失败，return code={rc}。请查看终端与 {LOG.name}")
    stamp(f"完成：{title}")


def main() -> None:
    banner()
    LOG.write_text("", encoding="utf-8")
    stamp("Fractal SSH canonical v7 FULL CLEAN 一键流程开始")
    stamp(f"Python : {PY}")
    stamp(f"ROOT   : {HERE}")
    stamp(f"DB     : {DB}")
    stamp(f"RUN    : {RUN}")

    for p in (DB_SCRIPT, MODEL_SCRIPT, BENCH_SCRIPT):
        if not p.exists():
            raise FileNotFoundError(f"缺少必要文件：{p.name}")

    # 1. Physics + label-definition benchmark
    run(
        [str(PY), str(BENCH_SCRIPT)],
        "STEP 1/5 v7 物理 / canonicalization / rank-constrained assignment benchmark",
    )

    # 2. Full v7 database; builder supports safe resume.
    run([
        str(PY), str(DB_SCRIPT),
        "--stage", "all",
        "--outdir", str(DB),
    ], "STEP 2/5 从头生成/续跑 v7 完整 816-family 数据库")

    # 3. Database/model check
    run([
        str(PY), str(MODEL_SCRIPT),
        "--mode", "check",
        "--db-root", str(DB),
        "--run-dir", str(RUN),
        "--queries-per-family", "32",
    ], "STEP 3/5 v7 数据库 + projector/hard 双头模型检查")

    # 4. Full training from scratch. Rerun resumes only this same v7 run.
    train = [
        str(PY), str(MODEL_SCRIPT),
        "--mode", "train",
        "--db-root", str(DB),
        "--run-dir", str(RUN),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--queries-per-family", str(QUERIES_PER_FAMILY),
        "--num-workers", str(NUM_WORKERS),
        "--lr", str(LR),
        "--train-backbone",
        "--uniform-query-fraction", "0.65",
        "--class-balance-power", "0.25",
        "--branch-weight", "0.20",
        "--hard-ce-weight", "0.20",
        "--subspace-weight", "0.10",
        "--rank-weight", "0.08",
        "--hard-rank-weight", "0.04",
        "--monotonic-weight", "0.05",
        "--lr-patience", "6",
        "--early-stop-patience", "25",
    ]

    last_ckpt = RUN / "last_model.pt"
    if last_ckpt.exists():
        stamp("*** 发现本目录 v7 last_model.pt：自动续跑同一个 v7 训练。***")
        train.append("--resume")
    else:
        stamp("*** V7 FULL CLEAN：随机初始化，从头训练；不会读取任何旧 v5/v6 checkpoint。***")

    run(train, "STEP 4/5 v7 从头全网络训练/续跑")

    # 5. Formal test
    run([
        str(PY), str(MODEL_SCRIPT),
        "--mode", "test",
        "--db-root", str(DB),
        "--run-dir", str(RUN),
        "--checkpoint-name", "best_model.pt",
        "--test-tag", "best_model",
        "--save-test-plots", "20",
    ], "STEP 5/5 v7 九分支正式测试")

    print("\n" + "*" * 104)
    print("*** V7 FULL CLEAN PIPELINE FINISHED ***")
    print(f"*** 结果目录：{RUN}")
    print("*** 重点文件：test_branch_metrics_best_model.csv")
    print("***           test_confusion_matrix_best_model.csv")
    print("***           test_summary_best_model.json")
    print("***           test_predictions_best_model/  (保存 soft projector + hard-head probability) ")
    print("*" * 104)


if __name__ == "__main__":
    main()
