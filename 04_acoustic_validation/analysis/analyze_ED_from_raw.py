# -*- coding: utf-8 -*-
"""
通用 ED 本地检测 / 实时监测脚本
============================================================

【文件命名规则】
    17(14)-27(32)a.dat

含义：
    17      = 第一个测点编号 A
    (14)    = A点：上表面 CH1，下表面 CH4
    27      = 第二个测点编号 B
    (32)    = B点：上表面 CH3，下表面 CH2
    a       = 第1次重复测量

后续可写：
    17(14)-27(32)b.dat   第2次
    17(14)-27(32)c.dat   第3次

其他 ED 点完全同样：
    18(14)-28(32)a.dat
    42(14)-45(32)a.dat
    44(14)-47(32)a.dat
    ...

程序会从文件名自动读取：
    - 点号
    - 上/下表面对应通道
    - 重复测量编号
无需再改代码中的点号或通道。

【冻结频窗】
    ED-I  : 4928–4963 Hz -> 理论同相 0°
    ED-II : 5859–5899 Hz -> 理论反相 180°

【DAT格式】
    big-endian float64
    arr[0] = fs
    arr[1] = nch
    arr[2:] = 按通道交织的时域数据

【依赖】
    pip install numpy

注意：
    文件名中的通道号是 1 基编号（CH1~CH4），程序内部会自动转换成 Python 的 0 基下标。
"""

from pathlib import Path
import csv
import re
import time
import numpy as np

# ============================================================
# 1. 数据目录：以后如果移动文件夹，只修改这里
# ============================================================
DATA_DIR = Path(__file__).resolve().parent / "raw" / "ed"

# 是否连子文件夹一起搜索
RECURSIVE = True

# ============================================================
# 2. ED 冻结频率窗口
# ============================================================
WINDOWS = {
    "ED-I":  (4928.0, 4963.0),
    "ED-II": (5859.0, 5899.0),
}

# ============================================================
# 3. 现场快速质量筛查阈值
#    这些只是现场QC，不是论文中的物理临界值
# ============================================================
PHASE_TOL_DEG = 30.0
CORR_MIN = 0.90
ANTI_MIN = 0.80

# ============================================================
# 4. 文件名解析
#    例：17(14)-27(32)a.dat
# ============================================================
NAME_RE = re.compile(
    r"^(?P<A>\d+)\((?P<Aup>[1-9]\d*)(?P<Adn>[1-9]\d*)\)"
    r"-(?P<B>\d+)\((?P<Bup>[1-9]\d*)(?P<Bdn>[1-9]\d*)\)"
    r"(?P<rep>[A-Za-z]+|\d+)?$"
)

# 由于你的实际通道只有1~4，像(14)、(32)恰好两位；
# 上面的通用正则在多位通道号场景会歧义，所以实际优先用这个四通道正则。
NAME_RE_4CH = re.compile(
    r"^(?P<A>\d+)\((?P<Aup>[1-4])(?P<Adn>[1-4])\)"
    r"-(?P<B>\d+)\((?P<Bup>[1-4])(?P<Bdn>[1-4])\)"
    r"(?P<rep>[A-Za-z]+|\d+)?$",
    re.IGNORECASE
)


def parse_filename(path: Path):
    """
    从文件名自动得到：
    A点号、A上通道、A下通道、B点号、B上通道、B下通道、重复编号
    """
    stem = path.stem.strip()
    m = NAME_RE_4CH.match(stem)
    if not m:
        return None

    g = m.groupdict()
    result = {
        "点A": int(g["A"]),
        "A上_CH": int(g["Aup"]),
        "A下_CH": int(g["Adn"]),
        "点B": int(g["B"]),
        "B上_CH": int(g["Bup"]),
        "B下_CH": int(g["Bdn"]),
        "重复编号": (g["rep"] or "").lower(),
    }

    # 基本合法性检查
    chs = [
        result["A上_CH"], result["A下_CH"],
        result["B上_CH"], result["B下_CH"]
    ]
    if any(c < 1 or c > 4 for c in chs):
        raise ValueError(f"通道必须为 CH1~CH4：{path.name}")
    if len(set(chs)) != 4:
        print(f"⚠ 警告：{path.name} 的四个上/下通道并非全部唯一：{chs}")

    return result


def read_dat(path: Path):
    arr = np.fromfile(path, dtype=">f8")
    if arr.size < 10:
        raise ValueError("文件太短，可能尚未写完")

    fs = float(arr[0])
    nch = int(round(float(arr[1])))
    body = arr[2:]

    if nch <= 0:
        raise ValueError(f"异常通道数 nch={nch}")
    if body.size % nch != 0:
        raise ValueError("文件可能仍在写入，数据长度不能整除通道数")

    data = body.reshape(-1, nch)
    return fs, data


def wait_file_stable(path: Path, interval=0.8, checks=3):
    """实时监测时避免文件尚未写完就开始分析。"""
    last = -1
    stable = 0

    for _ in range(60):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            time.sleep(interval)
            continue

        if size == last and size > 16:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0

        last = size
        time.sleep(interval)

    return False


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def same_phase_error(phase):
    """离0°有多远。"""
    return abs(wrap_deg(phase))


def opposite_phase_error(phase):
    """离±180°有多远。"""
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
    """
    反对称能量比例：
        eta = |U-D|^2 / ( |U-D|^2 + |U+D|^2 )
    """
    anti = np.sum(np.abs(U[mask] - D[mask]) ** 2)
    sym  = np.sum(np.abs(U[mask] + D[mask]) ** 2)
    den = anti + sym
    return float(anti / den) if den > 0 else np.nan


def analyze_one(path: Path, branch: str, info: dict):
    fs, d = read_dat(path)
    n = len(d)

    # 文件名中的 CH1~CH4 -> Python 0~3
    Aup = info["A上_CH"] - 1
    Adn = info["A下_CH"] - 1
    Bup = info["B上_CH"] - 1
    Bdn = info["B下_CH"] - 1

    max_ch = max(Aup, Adn, Bup, Bdn)
    if d.shape[1] <= max_ch:
        raise ValueError(
            f"文件只有 {d.shape[1]} 通道，但文件名要求使用 CH{max_ch + 1}"
        )

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lo, hi = WINDOWS[branch]
    mask = (freqs >= lo) & (freqs <= hi)

    if mask.sum() < 2:
        raise ValueError(f"{branch} 频窗内FFT点数不足")

    # 局域反对称轨道
    a = d[:, Aup] - d[:, Adn]
    b = d[:, Bup] - d[:, Bdn]

    A = np.fft.rfft(a)
    B = np.fft.rfft(b)

    EA = float(np.sum(np.abs(A[mask]) ** 2))
    EB = float(np.sum(np.abs(B[mask]) ** 2))

    if EA <= 0 or EB <= 0:
        raise ValueError(f"{branch} 频窗内能量为0")

    # 同一文件内的归一化复交叉量
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

    if branch == "ED-I":
        target = "0° 同相"
        error = same_phase_error(phase)
    else:
        target = "180° 反相"
        error = opposite_phase_error(phase)

    passed = (
        error <= PHASE_TOL_DEG
        and corr >= CORR_MIN
        and etaA >= ANTI_MIN
        and etaB >= ANTI_MIN
    )

    return {
        "文件名": path.name,
        "重复编号": info["重复编号"],
        "点A": info["点A"],
        "A上_CH": info["A上_CH"],
        "A下_CH": info["A下_CH"],
        "点B": info["点B"],
        "B上_CH": info["B上_CH"],
        "B下_CH": info["B下_CH"],
        "分支": branch,
        "理论目标": target,
        "采样率_Hz": fs,
        "时长_s": n / fs,
        "相位差_A减B_deg": phase,
        "相位误差_deg": error,
        "幅值比_A除B_dB": ratio_db,
        "复相关绝对值_C": corr,
        "A反对称能量分数": etaA,
        "B反对称能量分数": etaB,
        "自动判断": "通过" if passed else "复查",
    }


def pair_key(info):
    """
    分组只按物理点号，不按通道和a/b/c。
    17(...)-27(...)a/b/c 会自动归入同一组。
    """
    return (info["点A"], info["点B"])


def rep_sort_key(rep):
    if rep is None:
        return (2, "")
    rep = str(rep).lower()
    if rep.isdigit():
        return (0, int(rep))
    return (1, rep)


def find_dat_files():
    if RECURSIVE:
        files = list(DATA_DIR.rglob("*.dat")) + list(DATA_DIR.rglob("*.DAT"))
    else:
        files = list(DATA_DIR.glob("*.dat")) + list(DATA_DIR.glob("*.DAT"))

    # 去重
    files = list({p.resolve(): p for p in files}.values())

    parsed = []
    ignored = []

    for p in files:
        info = parse_filename(p)
        if info is None:
            ignored.append(p)
        else:
            parsed.append((p, info))

    parsed.sort(
        key=lambda x: (
            x[1]["点A"],
            x[1]["点B"],
            rep_sort_key(x[1]["重复编号"]),
            x[0].name
        )
    )
    return parsed, ignored


def print_one_result(r):
    icon = "✅" if r["自动判断"] == "通过" else "⚠"
    rep = r["重复编号"] or "未标"

    print(
        f"  {r['分支']:6s} | "
        f"重复={rep:>3s} | "
        f"相位={r['相位差_A减B_deg']:+8.2f}° | "
        f"目标误差={r['相位误差_deg']:6.2f}° | "
        f"|C|={r['复相关绝对值_C']:.4f} | "
        f"ηA={100*r['A反对称能量分数']:.1f}% | "
        f"ηB={100*r['B反对称能量分数']:.1f}% | "
        f"{icon}{r['自动判断']}"
    )


def summarize_pair(rows, A, B):
    print("\n" + "-" * 92)
    print(f"ED {A}-{B} 重复测量汇总")
    print("-" * 92)

    summary_rows = []

    for branch in ("ED-I", "ED-II"):
        rr = [r for r in rows if r["分支"] == branch]
        if not rr:
            continue

        phases = [r["相位差_A减B_deg"] for r in rr]
        corrs  = [r["复相关绝对值_C"] for r in rr]
        etaA   = [r["A反对称能量分数"] for r in rr]
        etaB   = [r["B反对称能量分数"] for r in rr]
        dbs    = [r["幅值比_A除B_dB"] for r in rr]

        mu = circular_mean_deg(phases)
        sd = circular_std_deg(phases)

        if branch == "ED-I":
            target = "0°"
            err = same_phase_error(mu)
        else:
            target = "180°"
            err = opposite_phase_error(mu)

        final_pass = (
            err <= PHASE_TOL_DEG
            and np.mean(corrs) >= CORR_MIN
            and np.mean(etaA) >= ANTI_MIN
            and np.mean(etaB) >= ANTI_MIN
        )

        db_sd = float(np.std(dbs, ddof=1)) if len(dbs) > 1 else 0.0

        print(
            f"{branch}: n={len(rr)} | "
            f"相位均值={mu:+.2f}° ± {sd:.2f}° | "
            f"理论={target} | 误差={err:.2f}° | "
            f"平均|C|={np.mean(corrs):.4f} | "
            f"ηA={100*np.mean(etaA):.1f}% | "
            f"ηB={100*np.mean(etaB):.1f}% | "
            f"A/B={np.mean(dbs):+.2f}±{db_sd:.2f} dB "
            f"→ {'✅总体通过' if final_pass else '⚠总体复查'}"
        )

        summary_rows.append({
            "点A": A,
            "点B": B,
            "分支": branch,
            "重复次数": len(rr),
            "相位均值_deg": mu,
            "相位标准差_deg": sd,
            "理论目标": target,
            "目标误差_deg": err,
            "平均复相关绝对值_C": float(np.mean(corrs)),
            "平均A反对称能量分数": float(np.mean(etaA)),
            "平均B反对称能量分数": float(np.mean(etaB)),
            "平均幅值比_A除B_dB": float(np.mean(dbs)),
            "幅值比标准差_dB": db_sd,
            "总体判断": "通过" if final_pass else "复查",
        })

    return summary_rows


def save_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def analyze_all():
    if not DATA_DIR.exists():
        print("数据目录不存在：")
        print(DATA_DIR)
        return

    parsed, ignored = find_dat_files()

    print("=" * 92)
    print("通用 ED 本地检测")
    print("=" * 92)
    print("数据目录：", DATA_DIR)
    print("命名示例：17(14)-27(32)a.dat")
    print("解释：17上CH1/下CH4；27上CH3/下CH2；a=第1次")
    print("ED-I  4928–4963 Hz -> 同相 0°")
    print("ED-II 5859–5899 Hz -> 反相 180°")
    print("=" * 92)

    if ignored:
        print("\n以下 DAT 因文件名不符合规则而忽略：")
        for p in ignored:
            print("  -", p.name)

    if not parsed:
        print("\n没有找到符合命名规则的 DAT 文件。")
        return

    all_rows = []
    grouped = {}

    for p, info in parsed:
        key = pair_key(info)
        grouped.setdefault(key, [])

        print(
            f"\n文件：{p.name}\n"
            f"  解析：A={info['点A']} "
            f"(上CH{info['A上_CH']}/下CH{info['A下_CH']})，"
            f"B={info['点B']} "
            f"(上CH{info['B上_CH']}/下CH{info['B下_CH']})，"
            f"重复={info['重复编号'] or '未标'}"
        )

        try:
            for branch in ("ED-I", "ED-II"):
                r = analyze_one(p, branch, info)
                all_rows.append(r)
                grouped[key].append(r)
                print_one_result(r)
        except Exception as e:
            print("  ❌分析失败：", e)

    # 保存逐文件结果
    detail_csv = DATA_DIR / "ED_全部测点_逐文件检测结果.csv"
    save_csv(all_rows, detail_csv)

    # 每一对单独汇总
    summary_rows = []
    print("\n\n" + "=" * 92)
    print("全部 ED 点对汇总")
    print("=" * 92)

    for (A, B), rows in sorted(grouped.items()):
        summary_rows.extend(summarize_pair(rows, A, B))

    summary_csv = DATA_DIR / "ED_全部测点_重复测量汇总.csv"
    save_csv(summary_rows, summary_csv)

    print("\n结果已保存：")
    print("  ", detail_csv)
    print("  ", summary_csv)


def live_watch():
    """
    实时监测：
    以后把符合命名规则的新DAT放入目录即可。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    done = set()
    print("=" * 92)
    print("通用 ED 实时监测已启动")
    print("目录：", DATA_DIR)
    print("文件命名示例：17(14)-27(32)a.dat")
    print("Ctrl+C 结束。")
    print("=" * 92)

    try:
        while True:
            parsed, ignored = find_dat_files()

            for p, info in parsed:
                full = str(p.resolve())
                if full in done:
                    continue

                if not wait_file_stable(p):
                    continue

                print(
                    f"\n发现新文件：{p.name}\n"
                    f"解析：A={info['点A']} "
                    f"(上CH{info['A上_CH']}/下CH{info['A下_CH']})；"
                    f"B={info['点B']} "
                    f"(上CH{info['B上_CH']}/下CH{info['B下_CH']})；"
                    f"重复={info['重复编号'] or '未标'}"
                )

                try:
                    for branch in ("ED-I", "ED-II"):
                        r = analyze_one(p, branch, info)
                        print_one_result(r)

                    done.add(full)

                    # 每来一个新文件，都重新生成完整CSV与各组汇总
                    print("\n正在刷新总汇总...")
                    analyze_all()

                except Exception as e:
                    print("❌分析失败：", e)

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n已停止实时监测。")


if __name__ == "__main__":
    print("\n请选择模式：")
    print("1 = 分析当前目录中所有符合命名规则的ED数据")
    print("2 = 实时监测，新DAT保存后自动分析")
    mode = input("请输入 1 或 2，直接回车默认 1：").strip()

    if mode == "2":
        live_watch()
    else:
        analyze_all()
