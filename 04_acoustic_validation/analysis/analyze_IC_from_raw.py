# -*- coding: utf-8 -*-
"""
IC 本地检测｜COMSOL校准稳定版 v5

改动：
1. 明确区分 E1-C、E2-C、E1-E2 三种关系。
2. 不再把两条端-中混成 n=6。
3. IC-II 端-中：中心点 eta 仅显示，不作为节点否决条件。
4. IC-II 主判据：中心受抑 + E1-E2 反相。
5. 每个 trimer 自动输出 fingerprint 总结。
6. 直接运行即分析。
"""

from pathlib import Path
import csv
import re
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "raw" / "ic"
RECURSIVE = True

WINDOWS = {
    "IC-I":   (4839.0, 4884.0),
    "IC-II":  (5363.0, 5401.0),
    "IC-III": (6103.0, 6168.0),
}

PHASE_TOL_DEG = 30.0
CORR_MIN = 0.90
ANTI_MIN = 0.80
NODE_STRONG_DB = -10.0
NODE_WEAK_DB = -5.0

TRIMERS = [
    (1, 11, 8), (2, 12, 15), (6, 9, 19), (13, 10, 20),
    (21, 39, 36), (22, 40, 43), (25, 51, 48), (26, 52, 55),
    (29, 63, 60), (30, 64, 67), (34, 37, 71), (41, 38, 72),
    (46, 49, 75), (53, 50, 76), (58, 61, 79), (65, 62, 80),
    (81, 91, 88), (82, 92, 95), (86, 89, 99), (93, 90, 100),
]

NAME_RE_4CH = re.compile(
    r"^(?P<A>\d+)\((?P<Aup>[1-4])(?P<Adn>[1-4])\)"
    r"-(?P<B>\d+)\((?P<Bup>[1-4])(?P<Bdn>[1-4])\)"
    r"(?P<rep>[A-Za-z]+|\d+)?$",
    re.IGNORECASE,
)


def parse_filename(path: Path):
    m = NAME_RE_4CH.match(path.stem.strip())
    if not m:
        return None
    g = m.groupdict()
    return {
        "点A": int(g["A"]), "A上_CH": int(g["Aup"]), "A下_CH": int(g["Adn"]),
        "点B": int(g["B"]), "B上_CH": int(g["Bup"]), "B下_CH": int(g["Bdn"]),
        "重复编号": (g["rep"] or "").lower(),
    }


def identify_pair(a, b):
    s = {a, b}
    for idx, (e1, c, e2) in enumerate(TRIMERS, start=1):
        if s == {e1, c}:
            return {"trimer编号": idx, "E1": e1, "C": c, "E2": e2, "关系": "E1-C"}
        if s == {e2, c}:
            return {"trimer编号": idx, "E1": e1, "C": c, "E2": e2, "关系": "E2-C"}
        if s == {e1, e2}:
            return {"trimer编号": idx, "E1": e1, "C": c, "E2": e2, "关系": "E1-E2"}
    return None


def read_dat(path: Path):
    arr = np.fromfile(path, dtype=">f8")
    if arr.size < 10:
        raise ValueError("文件太短")
    fs = float(arr[0])
    nch = int(round(float(arr[1])))
    body = arr[2:]
    if nch <= 0:
        raise ValueError(f"异常通道数 nch={nch}")
    if body.size % nch != 0:
        raise ValueError("数据长度不能整除通道数，文件可能尚未写完")
    return fs, body.reshape(-1, nch)


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def same_phase_error(phase):
    return abs(wrap_deg(phase))


def opposite_phase_error(phase):
    return abs(abs(wrap_deg(phase)) - 180.0)


def circular_mean_deg(vals):
    vals = np.asarray(vals, dtype=float)
    a = np.deg2rad(vals)
    return float(np.degrees(np.angle(np.mean(np.exp(1j * a)))))


def circular_std_deg(vals):
    vals = list(vals)
    if len(vals) <= 1:
        return 0.0
    mu = circular_mean_deg(vals)
    d = np.array([wrap_deg(v - mu) for v in vals], dtype=float)
    return float(np.std(d, ddof=1))


def anti_fraction(U, D, mask):
    anti = np.sum(np.abs(U[mask] - D[mask]) ** 2)
    sym = np.sum(np.abs(U[mask] + D[mask]) ** 2)
    den = anti + sym
    return float(anti / den) if den > 0 else np.nan


def theory_target(branch, relation):
    if relation in ("E1-C", "E2-C"):
        if branch == "IC-I": return "0° 同相", 0.0
        if branch == "IC-II": return "中心节点", None
        if branch == "IC-III": return "180° 反相", 180.0
    if relation == "E1-E2":
        if branch == "IC-I": return "0° 同相", 0.0
        if branch == "IC-II": return "180° 反相", 180.0
        if branch == "IC-III": return "0° 同相", 0.0
    raise ValueError("未知理论目标")



def analyze_one(path, branch, info, pair):
    fs, d = read_dat(path)
    n = len(d)

    Aup = info["A上_CH"] - 1
    Adn = info["A下_CH"] - 1
    Bup = info["B上_CH"] - 1
    Bdn = info["B下_CH"] - 1

    if d.shape[1] <= max(Aup, Adn, Bup, Bdn):
        raise ValueError("文件通道数与文件名不匹配")

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lo, hi = WINDOWS[branch]
    mask = (freqs >= lo) & (freqs <= hi)

    if mask.sum() < 2:
        raise ValueError(f"{branch} 频窗内 FFT 点数不足")

    a = d[:, Aup] - d[:, Adn]
    b = d[:, Bup] - d[:, Bdn]

    A = np.fft.rfft(a)
    B = np.fft.rfft(b)

    EA = float(np.sum(np.abs(A[mask]) ** 2))
    EB = float(np.sum(np.abs(B[mask]) ** 2))

    if EA <= 0 or EB <= 0:
        raise ValueError(f"{branch} 频窗内能量为 0")

    z = np.sum(A[mask] * np.conj(B[mask])) / np.sqrt(EA * EB)
    phase = float(np.degrees(np.angle(z)))
    corr = float(abs(z))
    ratio_db = float(20.0 * np.log10(np.sqrt(EA / EB)))

    U_A = np.fft.rfft(d[:, Aup])
    D_A = np.fft.rfft(d[:, Adn])
    U_B = np.fft.rfft(d[:, Bup])
    D_B = np.fft.rfft(d[:, Bdn])

    etaA = anti_fraction(U_A, D_A, mask)
    etaB = anti_fraction(U_B, D_B, mask)

    target_text, target_phase = theory_target(branch, pair["关系"])

    center_end_db = np.nan
    endpoint_eta = np.nan
    center_eta = np.nan

    if pair["关系"] in ("E1-C", "E2-C"):
        center = pair["C"]
        if info["点A"] == center:
            Ecenter, Eend = EA, EB
            center_eta, endpoint_eta = etaA, etaB
        elif info["点B"] == center:
            Ecenter, Eend = EB, EA
            center_eta, endpoint_eta = etaB, etaA
        else:
            raise ValueError("端-中关系中没有识别到中心点")

        center_end_db = float(20.0 * np.log10(np.sqrt(Ecenter / Eend)))

    if target_phase is not None:
        if target_phase == 0.0:
            err = same_phase_error(phase)
        else:
            err = opposite_phase_error(phase)

        passed = (
            err <= PHASE_TOL_DEG
            and corr >= CORR_MIN
            and etaA >= ANTI_MIN
            and etaB >= ANTI_MIN
        )
        status = "通过" if passed else "复查"

    else:
        # IC-II 端-中：中心本来就是弱响应，中心 eta 只显示，不作为否决条件。
        err = np.nan
        if (
            np.isfinite(center_end_db)
            and center_end_db <= NODE_STRONG_DB
            and endpoint_eta >= ANTI_MIN
        ):
            status = "强节点支持"
        elif (
            np.isfinite(center_end_db)
            and center_end_db <= NODE_WEAK_DB
            and endpoint_eta >= ANTI_MIN
        ):
            status = "弱节点支持"
        else:
            status = "节点复查"

    return {
        "文件名": path.name,
        "重复编号": info["重复编号"],
        "trimer编号": pair["trimer编号"],
        "E1": pair["E1"],
        "C": pair["C"],
        "E2": pair["E2"],
        "关系": pair["关系"],
        "点A": info["点A"],
        "点B": info["点B"],
        "分支": branch,
        "理论目标": target_text,
        "相位差_A减B_deg": phase,
        "相位误差_deg": err,
        "复相关绝对值_C": corr,
        "A反对称能量分数": etaA,
        "B反对称能量分数": etaB,
        "幅值比_A除B_dB": ratio_db,
        "中心除端点_dB": center_end_db,
        "端点反对称能量分数": endpoint_eta,
        "中心反对称能量分数_仅参考": center_eta,
        "自动判断": status,
    }

def find_files():
    if RECURSIVE:
        fs = list(DATA_DIR.rglob("*.dat")) + list(DATA_DIR.rglob("*.DAT"))
    else:
        fs = list(DATA_DIR.glob("*.dat")) + list(DATA_DIR.glob("*.DAT"))
    fs = list({p.resolve(): p for p in fs}.values())
    good, ignored = [], []
    for p in fs:
        info = parse_filename(p)
        if info is None:
            ignored.append((p, "文件名格式不符合规则")); continue
        pair = identify_pair(info["点A"], info["点B"])
        if pair is None:
            ignored.append((p, "点对不属于已定义 trimer")); continue
        good.append((p, info, pair))
    good.sort(key=lambda x: (x[2]["trimer编号"], {"E1-C":0,"E2-C":1,"E1-E2":2}[x[2]["关系"]], x[1]["重复编号"], x[0].name))
    return good, ignored



def summarize_relation(rows, tid, relation, branch):
    rr = [
        r for r in rows
        if r["trimer编号"] == tid
        and r["关系"] == relation
        and r["分支"] == branch
    ]
    if not rr:
        return None

    phases = [r["相位差_A减B_deg"] for r in rr]
    mu = circular_mean_deg(phases)
    sd = circular_std_deg(phases)

    corrs = [r["复相关绝对值_C"] for r in rr]
    etaA = [r["A反对称能量分数"] for r in rr]
    etaB = [r["B反对称能量分数"] for r in rr]

    target_text, target_phase = theory_target(branch, relation)

    center_vals = [
        r["中心除端点_dB"] for r in rr
        if np.isfinite(r["中心除端点_dB"])
    ]
    endpoint_eta_vals = [
        r["端点反对称能量分数"] for r in rr
        if np.isfinite(r["端点反对称能量分数"])
    ]
    center_eta_vals = [
        r["中心反对称能量分数_仅参考"] for r in rr
        if np.isfinite(r["中心反对称能量分数_仅参考"])
    ]

    if target_phase is not None:
        err = same_phase_error(mu) if target_phase == 0.0 else opposite_phase_error(mu)
        passed = (
            err <= PHASE_TOL_DEG
            and np.mean(corrs) >= CORR_MIN
            and np.mean(etaA) >= ANTI_MIN
            and np.mean(etaB) >= ANTI_MIN
        )
        status = "通过" if passed else "复查"

    else:
        err = np.nan
        center_mean = float(np.mean(center_vals)) if center_vals else np.nan
        endpoint_eta_mean = float(np.mean(endpoint_eta_vals)) if endpoint_eta_vals else np.nan

        if (
            np.isfinite(center_mean)
            and center_mean <= NODE_STRONG_DB
            and endpoint_eta_mean >= ANTI_MIN
        ):
            status = "强节点支持"
        elif (
            np.isfinite(center_mean)
            and center_mean <= NODE_WEAK_DB
            and endpoint_eta_mean >= ANTI_MIN
        ):
            status = "弱节点支持"
        else:
            status = "节点复查"

    return {
        "trimer编号": tid,
        "关系": relation,
        "分支": branch,
        "重复次数": len(rr),
        "相位均值_deg": mu,
        "相位标准差_deg": sd,
        "理论目标": target_text,
        "相位误差_deg": err,
        "平均复相关绝对值_C": float(np.mean(corrs)),
        "平均A反对称能量分数": float(np.mean(etaA)),
        "平均B反对称能量分数": float(np.mean(etaB)),
        "中心除端点平均_dB": float(np.mean(center_vals)) if center_vals else np.nan,
        "端点eta平均": float(np.mean(endpoint_eta_vals)) if endpoint_eta_vals else np.nan,
        "中心eta平均_仅参考": float(np.mean(center_eta_vals)) if center_eta_vals else np.nan,
        "总体判断": status,
    }


def fingerprint_status(summary_rows, tid, branch):
    d = {
        (r["关系"], r["分支"]): r
        for r in summary_rows if r["trimer编号"] == tid
    }

    need = [
        ("E1-C", branch),
        ("E2-C", branch),
        ("E1-E2", branch),
    ]
    if not all(k in d for k in need):
        return "未完整"

    r1 = d[("E1-C", branch)]
    r2 = d[("E2-C", branch)]
    r3 = d[("E1-E2", branch)]

    if branch == "IC-I":
        ok = all(r["总体判断"] == "通过" for r in (r1, r2, r3))

    elif branch == "IC-II":
        node_ok = {"强节点支持", "弱节点支持"}
        ok = (
            r1["总体判断"] in node_ok
            and r2["总体判断"] in node_ok
            and r3["总体判断"] == "通过"
        )

    elif branch == "IC-III":
        ok = all(r["总体判断"] == "通过" for r in (r1, r2, r3))

    else:
        ok = False

    return "完整通过" if ok else "需要复查"

def save_csv(rows, path):
    if not rows: return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def analyze_all():
    if not DATA_DIR.exists():
        print("数据目录不存在：", DATA_DIR); return

    parsed, ignored = find_files()
    print("=" * 110)
    print("IC 本地检测｜COMSOL校准稳定版 v5")
    print("=" * 110)
    print("数据目录：", DATA_DIR)
    print("IC-I   : E1-C 同相；E2-C 同相；E1-E2 同相")
    print("IC-II  : E1-C / E2-C 中心受抑；E1-E2 反相")
    print("IC-III : E1-C 反相；E2-C 反相；E1-E2 同相")
    print(f"节点分级：<= {NODE_STRONG_DB:.1f} dB 强节点；{NODE_STRONG_DB:.1f}~{NODE_WEAK_DB:.1f} dB 弱节点；> {NODE_WEAK_DB:.1f} dB 复查")
    print("注意：IC-II 中心点 eta 只显示，不作为节点否决条件")
    print("=" * 110)

    if ignored:
        print("\n被忽略的文件：")
        for p, why in ignored:
            print(f"  - {p.name}: {why}")

    all_rows = []
    for p, info, pair in parsed:
        print(f"\n文件：{p.name}\n  T{pair['trimer编号']:02d}=({pair['E1']},{pair['C']},{pair['E2']}) | 关系={pair['关系']} | A={info['点A']} | B={info['点B']}")
        try:
            for branch in ("IC-I", "IC-II", "IC-III"):
                r = analyze_one(p, branch, info, pair)
                all_rows.append(r)
                if np.isfinite(r["相位误差_deg"]):
                    print(f"  {branch:7s} | {pair['关系']:5s} | 相位={r['相位差_A减B_deg']:+8.2f}° | 误差={r['相位误差_deg']:6.2f}° | |C|={r['复相关绝对值_C']:.4f} | ηA={100*r['A反对称能量分数']:.1f}% | ηB={100*r['B反对称能量分数']:.1f}% | {r['自动判断']}")
                else:
                    print(f"  {branch:7s} | {pair['关系']:5s} | 中心/端={r['中心除端点_dB']:+6.2f} dB | 端点η={100*r['端点反对称能量分数']:.1f}% | 中心η={100*r['中心反对称能量分数_仅参考']:.1f}%（仅参考） | {r['自动判断']}")
        except Exception as e:
            print("  ❌分析失败：", e)

    detail_csv = DATA_DIR / "IC_三关系_逐文件检测结果.csv"
    save_csv(all_rows, detail_csv)

    summary_rows = []
    tids = sorted(set(r["trimer编号"] for r in all_rows))
    for tid in tids:
        for relation in ("E1-C", "E2-C", "E1-E2"):
            for branch in ("IC-I", "IC-II", "IC-III"):
                s = summarize_relation(all_rows, tid, relation, branch)
                if s: summary_rows.append(s)

    summary_csv = DATA_DIR / "IC_三关系_重复测量汇总.csv"
    save_csv(summary_rows, summary_csv)

    print("\n\n" + "=" * 110)
    print("三关系重复测量汇总")
    print("=" * 110)
    for r in summary_rows:
        if np.isfinite(r["相位误差_deg"]):
            print(f"T{r['trimer编号']:02d} {r['关系']:5s} {r['分支']:7s} | n={r['重复次数']} | 相位={r['相位均值_deg']:+.2f}° ± {r['相位标准差_deg']:.2f}° | 理论={r['理论目标']} | 误差={r['相位误差_deg']:.2f}° | |C|={r['平均复相关绝对值_C']:.4f} | {r['总体判断']}")
        else:
            print(f"T{r['trimer编号']:02d} {r['关系']:5s} {r['分支']:7s} | n={r['重复次数']} | 中心/端={r['中心除端点平均_dB']:+.2f} dB | 端点η={100*r['端点eta平均']:.1f}% | 中心η={100*r['中心eta平均_仅参考']:.1f}%（仅参考） | {r['总体判断']}")

    print("\n" + "=" * 110)
    print("每个 trimer 的三分支 fingerprint")
    print("=" * 110)
    for tid in tids:
        e1, c, e2 = TRIMERS[tid - 1]
        print(f"\nT{tid:02d}=({e1},{c},{e2})")
        for branch in ("IC-I", "IC-II", "IC-III"):
            print(f"  {branch:7s} -> {fingerprint_status(summary_rows, tid, branch)}")

    print("\n结果已保存：")
    print("  ", detail_csv)
    print("  ", summary_csv)


if __name__ == "__main__":
    analyze_all()
