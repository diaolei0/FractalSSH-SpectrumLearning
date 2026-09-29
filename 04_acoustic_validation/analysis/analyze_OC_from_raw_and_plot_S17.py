#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. S17 — OC spatial-support validation from original DAT recordings
完整单文件版：直接输出分图 + 完整四联组合图。

核心处理：
    原始 DAT
      -> H1 上/下表面传递函数
      -> 局域反对称声学轨道响应 Phi_i(f)
      -> OC 分支频窗积分平方响应
      -> 5235.9375 Hz 峰值频率响应
      -> OC/control 对比度
      -> Fig. S17(a-d)

重要说明
--------
1. 本程序不使用 pandas、scipy 或 PIL。
2. 强制使用 Matplotlib Agg 后端，避免 PyCharm 交互绘图后端间接调用 pandas。
3. 全部绘图字体固定为 Times New Roman。
4. 所有数据标记统一采用圆形，不使用正方形、三角形、菱形或星形。
5. Fig. S17(c) 是“峰值频率检验”，不是独立窄频带数据集。
6. 33/34 配对只用于 Fig. S17(a) 的完整几何展示，不进入 23 组统一定量统计。
7. 完整四联图直接由 Matplotlib 绘制，不再依赖 PIL 拼图。

标准 DAT 格式
------------
第一项 float64：采样频率 fs
第二项 float64：通道数 nch
其余：多通道交错数据

标准通道：
    CH2 = upper
    CH3 = lower
    CH4 = common reference
    CH1 = auxiliary

默认原始数据目录沿用此前可以运行的 S17 版本：
    D:\横向各项同性\新论文全部\补充材料原稿\实验数据\声源53
"""

from pathlib import Path
import argparse
import csv
import os
import sys
import math
import warnings

# -----------------------------------------------------------------------------
# PyCharm / pandas 兼容处理：必须在 import pyplot 之前设置
# -----------------------------------------------------------------------------
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import font_manager


# -----------------------------------------------------------------------------
# 固定分析参数
# -----------------------------------------------------------------------------

OC_WINDOW = (5197.3, 5239.1)       # Hz，FEM 定义的 OC 分支频窗
PEAK_TARGET = 5235.9375             # Hz，最接近共同响应峰的 FFT 频率点
SOURCE_SITE = 53
SPECIAL_PAIR = (33, 34)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "oc"

DEFAULT_OUTDIR_NAME = "output_FigS17_OC_complete"

# 24 组几何 OC/control 配对
ALL_PAIRS = [
    (3,1,'bottom'), (4,2,'bottom'),
    (5,6,'left'), (7,8,'left'),
    (14,13,'right'), (16,15,'right'),
    (23,21,'bottom'), (24,22,'bottom'),
    (31,29,'bottom'), (32,30,'bottom'),
    (33,34,'left'), (35,36,'left'),
    (66,65,'right'), (68,67,'right'),
    (69,71,'top'), (70,72,'top'),
    (77,79,'top'), (78,80,'top'),
    (85,86,'left'), (87,88,'left'),
    (94,93,'right'), (96,95,'right'),
    (97,99,'top'), (98,100,'top'),
]

ANALYSIS_PAIRS = [
    p for p in ALL_PAIRS
    if (p[0], p[1]) != SPECIAL_PAIR
]

REQUIRED_SITES = sorted(
    {s for oc, ctrl, _ in ANALYSIS_PAIRS for s in (oc, ctrl)}
)

# Legacy regression constants are intentionally omitted from the public release.
REFERENCE_CONTRASTS = {}

# 单文件内置 G(2) 1–100 坐标；旁边有 CSV 时优先读取 CSV
EMBEDDED_COORDS = {
    1: (-14.500000, 14.500000),
    2: (14.500000, 14.500000),
    3: (-14.500000, -14.500000),
    4: (14.500000, -14.500000),
    5: (-72.500000, 72.500000),
    6: (-43.500000, 72.500000),
    7: (-72.500000, 43.500000),
    8: (-43.500000, 43.500000),
    9: (-14.500000, 72.500000),
    10: (14.500000, 72.500000),
    11: (-14.500000, 43.500000),
    12: (14.500000, 43.500000),
    13: (43.500000, 72.500000),
    14: (72.500000, 72.500000),
    15: (43.500000, 43.500000),
    16: (72.500000, 43.500000),
    17: (-14.500000, 130.500000),
    18: (14.500000, 130.500000),
    19: (-14.500000, 101.500000),
    20: (14.500000, 101.500000),
    21: (-188.500000, 188.500000),
    22: (-159.500000, 188.500000),
    23: (-188.500000, 159.500000),
    24: (-159.500000, 159.500000),
    25: (-14.500000, 188.500000),
    26: (14.500000, 188.500000),
    27: (-14.500000, 159.500000),
    28: (14.500000, 159.500000),
    29: (159.500000, 188.500000),
    30: (188.500000, 188.500000),
    31: (159.500000, 159.500000),
    32: (188.500000, 159.500000),
    33: (-246.500000, 246.500000),
    34: (-217.500000, 246.500000),
    35: (-246.500000, 217.500000),
    36: (-217.500000, 217.500000),
    37: (-188.500000, 246.500000),
    38: (-159.500000, 246.500000),
    39: (-188.500000, 217.500000),
    40: (-159.500000, 217.500000),
    41: (-130.500000, 246.500000),
    42: (-101.500000, 246.500000),
    43: (-130.500000, 217.500000),
    44: (-101.500000, 217.500000),
    45: (-72.500000, 246.500000),
    46: (-43.500000, 246.500000),
    47: (-72.500000, 217.500000),
    48: (-43.500000, 217.500000),
    49: (-14.500000, 246.500000),
    50: (14.500000, 246.500000),
    51: (-14.500000, 217.500000),
    52: (14.500000, 217.500000),
    53: (43.500000, 246.500000),
    54: (72.500000, 246.500000),
    55: (43.500000, 217.500000),
    56: (72.500000, 217.500000),
    57: (101.500000, 246.500000),
    58: (130.500000, 246.500000),
    59: (101.500000, 217.500000),
    60: (130.500000, 217.500000),
    61: (159.500000, 246.500000),
    62: (188.500000, 246.500000),
    63: (159.500000, 217.500000),
    64: (188.500000, 217.500000),
    65: (217.500000, 246.500000),
    66: (246.500000, 246.500000),
    67: (217.500000, 217.500000),
    68: (246.500000, 217.500000),
    69: (-188.500000, 304.500000),
    70: (-159.500000, 304.500000),
    71: (-188.500000, 275.500000),
    72: (-159.500000, 275.500000),
    73: (-14.500000, 304.500000),
    74: (14.500000, 304.500000),
    75: (-14.500000, 275.500000),
    76: (14.500000, 275.500000),
    77: (159.500000, 304.500000),
    78: (188.500000, 304.500000),
    79: (159.500000, 275.500000),
    80: (188.500000, 275.500000),
    81: (-14.500000, 362.500000),
    82: (14.500000, 362.500000),
    83: (-14.500000, 333.500000),
    84: (14.500000, 333.500000),
    85: (-72.500000, 420.500000),
    86: (-43.500000, 420.500000),
    87: (-72.500000, 391.500000),
    88: (-43.500000, 391.500000),
    89: (-14.500000, 420.500000),
    90: (14.500000, 420.500000),
    91: (-14.500000, 391.500000),
    92: (14.500000, 391.500000),
    93: (43.500000, 420.500000),
    94: (72.500000, 420.500000),
    95: (43.500000, 391.500000),
    96: (72.500000, 391.500000),
    97: (-14.500000, 478.500000),
    98: (14.500000, 478.500000),
    99: (-14.500000, 449.500000),
    100: (14.500000, 449.500000)
}

# 期刊图配色
RED = "#C0392B"
BLUE = "#2C6EAF"
GOLD = "#D9A72E"
DARK = "#222222"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D9E5EE"
STEM_GRAY = "#B8B8B8"


# -----------------------------------------------------------------------------
# 字体与坐标
# -----------------------------------------------------------------------------

def configure_font():
    """固定使用 Times New Roman。"""
    font_name = "Times New Roman"
    try:
        path = font_manager.findfont(font_name, fallback_to_default=False)
        font_name = font_manager.FontProperties(fname=path).get_name()
    except Exception:
        # Windows 正常情况下存在 Times New Roman。
        # 即使 findfont 当前无法解析，也仍然明确请求该字体。
        font_name = "Times New Roman"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    print("绘图字体：Times New Roman")
    return font_name


def load_coords():
    """
    优先读取脚本旁边的 G2_100_cavity_site_map.csv。
    若不存在，则自动使用代码内置坐标。
    """
    csv_path = Path(__file__).with_name("G2_100_cavity_site_map.csv")
    if csv_path.is_file():
        try:
            out = {}
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    out[int(row["site_id"])] = (
                        float(row["x_mm"]),
                        float(row["y_mm"])
                    )
            if len(out) == 100:
                print(f"读取外部坐标：{csv_path}")
                return out
        except Exception as exc:
            print(f"外部坐标读取失败，将使用内置坐标：{exc}")

    print("自动使用代码内置的 G(2) 1–100 号坐标。")
    return dict(EMBEDDED_COORDS)


# -----------------------------------------------------------------------------
# DAT 文件读取
# -----------------------------------------------------------------------------

def plausible_header(fs, nch):
    if not np.isfinite(fs) or not np.isfinite(nch):
        return False
    if not (1000.0 <= fs <= 2_000_000.0):
        return False
    nch_i = int(round(float(nch)))
    if not (1 <= nch_i <= 64):
        return False
    return abs(float(nch) - nch_i) <= 1e-8


def detect_dat_endian(path):
    candidates = []
    for endian in (">", "<"):
        hdr = np.fromfile(path, dtype=endian + "f8", count=2)
        if hdr.size == 2 and plausible_header(hdr[0], hdr[1]):
            candidates.append(
                (endian, float(hdr[0]), int(round(float(hdr[1]))))
            )

    if not candidates:
        raise ValueError(
            f"无法解析 DAT 文件头：{path}\n"
            "预期前两个 float64 分别为 fs 和 nch。"
        )

    if len(candidates) > 1:
        for item in candidates:
            if item[0] == ">":
                return item

    return candidates[0]


def read_original_dat(path, trim_start_s=0.0, trim_end_s=0.0):
    endian, fs, nch = detect_dat_endian(path)
    arr = np.fromfile(path, dtype=endian + "f8")
    if arr.size < 2:
        raise ValueError(f"文件过短：{path}")

    payload = arr[2:]
    rem = payload.size % nch
    if rem != 0:
        raise ValueError(
            f"{path.name} 的数据长度不能被 nch={nch} 整除，余数={rem}。"
        )

    data = payload.reshape(-1, nch)
    n_original = data.shape[0]

    i0 = int(round(trim_start_s * fs))
    i1 = n_original - int(round(trim_end_s * fs))
    if i0 < 0 or i1 <= i0 or i1 > n_original:
        raise ValueError(f"{path.name} 的裁剪参数无效。")

    data = data[i0:i1]
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name} 含非有限数值。")

    meta = {
        "file": str(path),
        "endian": "big" if endian == ">" else "little",
        "fs_hz": fs,
        "n_channels": nch,
        "n_samples_original": n_original,
        "n_samples_used": data.shape[0],
        "duration_used_s": data.shape[0] / fs,
    }
    return fs, data, meta


def locate_numeric_site_file(data_dir, site, recursive=True):
    """
    查找类似 3.dat / 003.dat 的标准数字文件名。
    """
    patterns = [f"{site}.dat", f"{site:02d}.dat", f"{site:03d}.dat"]

    matches = []
    for name in patterns:
        if recursive:
            matches.extend(data_dir.rglob(name))
        else:
            p = data_dir / name
            if p.is_file():
                matches.append(p)

    # 去重
    uniq = []
    seen = set()
    for p in matches:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)

    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) == 0:
        raise FileNotFoundError(
            f"未找到 site {site} 的标准 DAT 文件。\n"
            f"数据目录：{data_dir}"
        )

    raise RuntimeError(
        f"site {site} 找到多个候选文件：\n" +
        "\n".join(str(p) for p in uniq)
    )


# -----------------------------------------------------------------------------
# H1 与反对称声学轨道响应
# -----------------------------------------------------------------------------

def make_window(name, n):
    name = name.lower()
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name in ("rect", "boxcar", "none"):
        return np.ones(n)
    raise ValueError(f"不支持的窗函数：{name}")


def h1_upper_lower(
    data,
    fs,
    up_ch_1based=2,
    down_ch_1based=3,
    ref_ch_1based=4,
    nperseg=32768,
    overlap=0.5,
    nfft=None,
    window="hann",
    detrend_mean=True,
):
    """
    H1(f) = <Y X*> / <X X*>
    先在多个 Hann 分段上平均互谱/自谱，再作比值。
    """
    if nfft is None:
        nfft = nperseg
    if nfft < nperseg:
        raise ValueError("nfft 必须 >= nperseg。")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap 必须满足 0 <= overlap < 1。")

    n = data.shape[0]
    nch = data.shape[1]

    idx_up = up_ch_1based - 1
    idx_dn = down_ch_1based - 1
    idx_ref = ref_ch_1based - 1

    for label, idx in (("upper", idx_up), ("lower", idx_dn), ("reference", idx_ref)):
        if not (0 <= idx < nch):
            raise ValueError(
                f"{label} 通道 CH{idx+1} 超出文件通道数 {nch}。"
            )

    if n < nperseg:
        raise ValueError(
            f"记录仅 {n} 点，小于 nperseg={nperseg}。"
        )

    noverlap = int(round(nperseg * overlap))
    step = nperseg - noverlap
    starts = np.arange(0, n - nperseg + 1, step, dtype=int)
    if starts.size == 0:
        raise ValueError("没有完整频谱分段。")

    w = make_window(window, nperseg)

    nfreq = nfft // 2 + 1
    sxx = np.zeros(nfreq, dtype=np.float64)
    syx_up = np.zeros(nfreq, dtype=np.complex128)
    syx_dn = np.zeros(nfreq, dtype=np.complex128)

    for start in starts:
        seg = data[
            start:start+nperseg,
            [idx_up, idx_dn, idx_ref]
        ].astype(np.float64, copy=True)

        if detrend_mean:
            seg -= seg.mean(axis=0, keepdims=True)

        seg *= w[:, None]
        F = np.fft.rfft(seg, n=nfft, axis=0)

        Yu = F[:, 0]
        Yd = F[:, 1]
        X = F[:, 2]

        sxx += np.real(X * np.conj(X))
        syx_up += Yu * np.conj(X)
        syx_dn += Yd * np.conj(X)

    nseg = float(starts.size)
    sxx /= nseg
    syx_up /= nseg
    syx_dn /= nseg

    ref_floor = np.finfo(float).eps * max(float(np.max(sxx)), 1.0)
    valid = sxx > ref_floor

    H_up = np.full(nfreq, np.nan + 1j*np.nan, dtype=np.complex128)
    H_dn = np.full(nfreq, np.nan + 1j*np.nan, dtype=np.complex128)

    H_up[valid] = syx_up[valid] / sxx[valid]
    H_dn[valid] = syx_dn[valid] / sxx[valid]

    freq = np.fft.rfftfreq(nfft, d=1.0/fs)

    meta = {
        "nperseg": nperseg,
        "noverlap": noverlap,
        "overlap_fraction": overlap,
        "nfft": nfft,
        "frequency_resolution_hz": fs / nfft,
        "n_segments": int(starts.size),
        "window": window,
        "detrend_mean": bool(detrend_mean),
        "reference_valid_fraction": float(np.mean(valid)),
    }

    return freq, H_up, H_dn, meta


def orbital_response_from_dat(
    path,
    up_ch,
    down_ch,
    ref_ch,
    nperseg,
    overlap,
    nfft,
    window,
    detrend_mean,
    trim_start_s,
    trim_end_s,
):
    fs, data, dat_meta = read_original_dat(
        path,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s
    )

    freq, H_up, H_dn, h1_meta = h1_upper_lower(
        data=data,
        fs=fs,
        up_ch_1based=up_ch,
        down_ch_1based=down_ch,
        ref_ch_1based=ref_ch,
        nperseg=nperseg,
        overlap=overlap,
        nfft=nfft,
        window=window,
        detrend_mean=detrend_mean,
    )

    # 与论文 Eq. (S7.2) 一致
    phi = 0.5 * (H_up - H_dn)
    return freq, phi, {**dat_meta, **h1_meta}


def band_integrated_squared_response(freq, phi, f1, f2):
    """
    频带积分平方响应指标：
        R_i(W) = sum |Phi_i(f)|^2 * df

    注意：这是响应指标，不称为热力学声学能量。
    """
    finite = np.isfinite(phi.real) & np.isfinite(phi.imag)
    mask = (freq >= f1) & (freq <= f2) & finite

    if not np.any(mask):
        raise ValueError(
            f"频率区间 [{f1}, {f2}] Hz 内没有有效 FFT 点。"
        )

    df = float(freq[1] - freq[0])
    value = float(np.sum(np.abs(phi[mask])**2) * df)
    return value, int(mask.sum())


def peak_frequency_squared_response(freq, phi, target_hz=PEAK_TARGET):
    """
    峰值频率检验：
    只使用最接近 target_hz 的单个 FFT bin。

    对比度中公共的 df 因子会抵消，因此直接使用 |Phi|^2。
    """
    idx = int(np.argmin(np.abs(freq - target_hz)))
    if not np.isfinite(phi[idx].real) or not np.isfinite(phi[idx].imag):
        raise ValueError(
            f"峰值频率点 {freq[idx]:.6f} Hz 的 Phi 无效。"
        )
    return float(np.abs(phi[idx])**2), idx, float(freq[idx])


def pair_contrast_db(r_oc, r_ctrl):
    if r_oc <= 0 or r_ctrl <= 0:
        raise ValueError(
            f"遇到非正响应：OC={r_oc}, control={r_ctrl}"
        )
    return float(10.0 * np.log10(r_oc / r_ctrl))


# -----------------------------------------------------------------------------
# 原始数据处理
# -----------------------------------------------------------------------------

def process_raw_data(args):
    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"实验数据目录不存在：{data_dir}"
        )

    phi_cache = {}
    manifest_rows = []
    common_freq = None
    expected_fs = None

    print("=" * 96)
    print("Fig. S17｜OC 原始 DAT 空间支撑验证")
    print("=" * 96)
    print(f"实验数据目录：\n{data_dir}")
    print(f"需要统一处理的标准位点：{len(REQUIRED_SITES)} 个")
    print()

    for k, site in enumerate(REQUIRED_SITES, 1):
        path = locate_numeric_site_file(
            data_dir,
            site,
            recursive=not args.no_recursive_search
        )

        freq, phi, meta = orbital_response_from_dat(
            path=path,
            up_ch=args.up_ch,
            down_ch=args.down_ch,
            ref_ch=args.ref_ch,
            nperseg=args.nperseg,
            overlap=args.overlap,
            nfft=args.nfft,
            window=args.window,
            detrend_mean=not args.no_detrend,
            trim_start_s=args.trim_start,
            trim_end_s=args.trim_end,
        )

        if common_freq is None:
            common_freq = freq
            expected_fs = meta["fs_hz"]
        else:
            if not np.isclose(
                meta["fs_hz"], expected_fs,
                rtol=0, atol=1e-9
            ):
                raise ValueError(
                    f"site {site} 的 fs 不一致："
                    f"{meta['fs_hz']} vs {expected_fs} Hz"
                )

            if (
                common_freq.shape != freq.shape
                or not np.allclose(
                    common_freq, freq,
                    rtol=0, atol=1e-12
                )
            ):
                raise ValueError(
                    f"site {site} 的频率轴与其他文件不一致。"
                )

        phi_cache[site] = phi
        manifest_rows.append({"site_id": site, **meta})

        print(
            f"[{k:02d}/{len(REQUIRED_SITES)}] "
            f"site {site:>3} | {path.name} | "
            f"segments={meta['n_segments']} | "
            f"df={meta['frequency_resolution_hz']:.6f} Hz"
        )

    peak_idx = int(
        np.argmin(np.abs(common_freq - PEAK_TARGET))
    )
    peak_bin = float(common_freq[peak_idx])

    print()
    print(
        f"OC 分支频窗："
        f"{OC_WINDOW[0]:.1f}–{OC_WINDOW[1]:.1f} Hz"
    )
    print(
        f"峰值频率检验：目标 {PEAK_TARGET:.4f} Hz，"
        f"实际 FFT bin = {peak_bin:.4f} Hz"
    )

    site_metrics = {}

    for site in REQUIRED_SITES:
        phi = phi_cache[site]

        r_branch, nbin_branch = band_integrated_squared_response(
            common_freq,
            phi,
            OC_WINDOW[0],
            OC_WINDOW[1]
        )

        r_peak, idx, actual_peak = peak_frequency_squared_response(
            common_freq,
            phi,
            PEAK_TARGET
        )

        site_metrics[site] = {
            "R_branch": r_branch,
            "R_peak": r_peak,
            "branch_bins": nbin_branch,
            "peak_bin_index": idx,
            "peak_bin_hz": actual_peak,
        }

    pair_rows = []

    for oc, ctrl, direction in ANALYSIS_PAIRS:
        d_branch = pair_contrast_db(
            site_metrics[oc]["R_branch"],
            site_metrics[ctrl]["R_branch"]
        )

        d_peak = pair_contrast_db(
            site_metrics[oc]["R_peak"],
            site_metrics[ctrl]["R_peak"]
        )

        pair_rows.append({
            "oc_site": oc,
            "control_site": ctrl,
            "direction": direction,
            "pair": f"{oc}/{ctrl}",
            "R_oc_branch": site_metrics[oc]["R_branch"],
            "R_control_branch": site_metrics[ctrl]["R_branch"],
            "delta_branch_db": d_branch,
            "R_oc_peak": site_metrics[oc]["R_peak"],
            "R_control_peak": site_metrics[ctrl]["R_peak"],
            "delta_peak_db": d_peak,
            "sign_agreement": np.sign(d_branch) == np.sign(d_peak),
        })

    info = {
        "fs_hz": float(expected_fs),
        "frequency_resolution_hz": float(
            common_freq[1] - common_freq[0]
        ),
        "peak_bin_hz": peak_bin,
        "peak_bin_index": peak_idx,
    }

    return (
        pair_rows,
        site_metrics,
        manifest_rows,
        common_freq,
        phi_cache,
        info,
    )


# -----------------------------------------------------------------------------
# 统计工具
# -----------------------------------------------------------------------------

def values(rows, key):
    return np.asarray(
        [float(r[key]) for r in rows],
        dtype=float
    )


def median(rows, key):
    return float(np.median(values(rows, key)))


def positive_count(rows, key):
    return int(np.sum(values(rows, key) > 0))


def sign_agreement_count(rows):
    return int(sum(bool(r["sign_agreement"]) for r in rows))


# -----------------------------------------------------------------------------
# 绘图
# -----------------------------------------------------------------------------

def add_panel_label(ax, label):
    ax.text(
        -0.10, 1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        clip_on=False
    )


def draw_panel_a(ax, coords):
    all_sites = sorted(coords)
    xs = [coords[s][0] for s in all_sites]
    ys = [coords[s][1] for s in all_sites]

    oc_sites = [p[0] for p in ALL_PAIRS]
    ctrl_sites = [p[1] for p in ALL_PAIRS]

    ax.scatter(
        xs, ys,
        s=10,
        marker="o",
        facecolors=LIGHT_GRAY,
        edgecolors="#A8C5D8",
        linewidths=0.25,
        zorder=1
    )

    for oc, ctrl, _ in ALL_PAIRS:
        x1, y1 = coords[oc]
        x2, y2 = coords[ctrl]
        ls = "--" if (oc, ctrl) == SPECIAL_PAIR else "-"
        ax.plot(
            [x1, x2], [y1, y2],
            lw=0.65,
            ls=ls,
            alpha=0.55,
            color=MID_GRAY,
            zorder=2
        )

    ax.scatter(
        [coords[s][0] for s in oc_sites],
        [coords[s][1] for s in oc_sites],
        s=32,
        marker="o",
        facecolors=RED,
        edgecolors="black",
        linewidths=0.35,
        zorder=4,
        label="Predicted OC site"
    )

    # 控制位点也用圆形：空心蓝边圆
    ax.scatter(
        [coords[s][0] for s in ctrl_sites],
        [coords[s][1] for s in ctrl_sites],
        s=31,
        marker="o",
        facecolors="white",
        edgecolors=BLUE,
        linewidths=1.0,
        zorder=4,
        label="Inward control"
    )

    # 声源同样用圆形，通过金色填充和尺寸区分
    sx, sy = coords[SOURCE_SITE]
    ax.scatter(
        [sx], [sy],
        s=52,
        marker="o",
        facecolors=GOLD,
        edgecolors="black",
        linewidths=0.55,
        zorder=5,
        label="Fixed source (53)"
    )

    ax.annotate(
        "53",
        (sx, sy),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7.0
    )

    xm = 0.5 * (coords[33][0] + coords[34][0])
    ym = 0.5 * (coords[33][1] + coords[34][1])

    ax.annotate(
        "33/34*",
        xy=(xm, ym),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.7,
        arrowprops=dict(
            arrowstyle="-",
            lw=0.50,
            color=MID_GRAY,
            shrinkA=1,
            shrinkB=1
        )
    )

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xc = 0.5 * (x_min + x_max)
    yc = 0.5 * (y_min + y_max)
    span = max(x_max - x_min, y_max - y_min)
    half = 0.62 * span

    ax.set_xlim(xc-half, xc+half)
    ax.set_ylim(yc-half, yc+half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        borderaxespad=0,
        handletextpad=0.42,
        labelspacing=0.30,
        fontsize=6.5
    )

    add_panel_label(ax, "(a)")


def draw_contrast_panel(
    ax,
    rows,
    value_key,
    panel_label,
):
    vals = values(rows, value_key)
    x = np.arange(len(rows))
    posmask = vals > 0

    ax.vlines(
        x, 0, vals,
        linewidth=0.75,
        color=STEM_GRAY,
        zorder=1
    )

    ax.scatter(
        x[posmask], vals[posmask],
        s=28,
        marker="o",
        facecolors=RED,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
        label="OC > control"
    )

    ax.scatter(
        x[~posmask], vals[~posmask],
        s=28,
        marker="o",
        facecolors=BLUE,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
        label="OC < control"
    )

    med = float(np.median(vals))
    npos = int(np.sum(vals > 0))

    ax.axhline(0, lw=0.8, color=DARK)
    ax.axhline(
        med,
        lw=0.85,
        ls="--",
        color=MID_GRAY
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["pair"] for r in rows],
        rotation=90,
        ha="center",
        va="top"
    )

    ax.set_ylabel(r"$\Delta_k$ (dB)")
    ax.set_xlim(-0.6, len(rows)-0.4)
    ax.set_ylim(
        min(-22, float(vals.min()) - 2.5),
        max(31, float(vals.max()) + 2.5)
    )

    ax.text(
        0.985, 0.96,
        f"{npos}/{len(rows)} pairs > 0 dB\n"
        f"median = {med:+.2f} dB",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.1
    )

    ax.legend(
        loc="lower left",
        frameon=False,
        borderaxespad=0.25,
        handletextpad=0.4,
        labelspacing=0.25
    )

    add_panel_label(ax, panel_label)


def draw_panel_d(ax, rows):
    x = values(rows, "delta_branch_db")
    y = values(rows, "delta_peak_db")

    lim = max(
        31.0,
        float(np.max(np.abs(np.r_[x, y]))) + 2.5
    )

    ax.axhline(0, lw=0.75, color="black")
    ax.axvline(0, lw=0.75, color="black")
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        ls="--",
        lw=0.75,
        color=MID_GRAY,
        zorder=1
    )

    # 所有点统一圆形，黑色空心
    ax.scatter(
        x, y,
        s=36,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=0.9,
        zorder=3
    )

    discordant = [
        r for r in rows
        if not bool(r["sign_agreement"])
    ]

    for row in discordant:
        xx = float(row["delta_branch_db"])
        yy = float(row["delta_peak_db"])

        ax.scatter(
            [xx], [yy],
            s=82,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.35,
            zorder=4
        )

        ax.annotate(
            row["pair"],
            (xx, yy),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=7.0
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("OC branch-integrated contrast (dB)")
    ax.set_ylabel("Peak-frequency contrast (dB)")

    nagree = sign_agreement_count(rows)

    ax.text(
        0.03, 0.96,
        f"{nagree}/{len(rows)} pairs retain the same sign",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1
    )

    add_panel_label(ax, "(d)")


def save_figure_all(fig, stem, dpi=600):
    stem = Path(stem)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.03
    )
    try:
        fig.savefig(
            stem.with_suffix(".tiff"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03
        )
    except Exception as exc:
        print(f"TIFF 保存失败（PNG/PDF 不受影响）：{exc}")


def save_individual_panels(coords, rows, outdir):
    # (a)
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    draw_panel_a(ax, coords)
    save_figure_all(fig, outdir / "S17a_OC_spatial_map")
    plt.close(fig)

    # (b)
    fig, ax = plt.subplots(figsize=(5.1, 3.05))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.95)
    draw_contrast_panel(
        ax, rows,
        "delta_branch_db",
        "(b)"
    )
    save_figure_all(fig, outdir / "S17b_OC_branch_integrated_contrast")
    plt.close(fig)

    # (c)
    fig, ax = plt.subplots(figsize=(5.1, 3.05))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.95)
    draw_contrast_panel(
        ax, rows,
        "delta_peak_db",
        "(c)"
    )
    save_figure_all(fig, outdir / "S17c_OC_peak_frequency_contrast")
    plt.close(fig)

    # (d)
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.95)
    draw_panel_d(ax, rows)
    save_figure_all(fig, outdir / "S17d_OC_branch_vs_peak_consistency")
    plt.close(fig)


def save_complete_figure(coords, rows, outdir):
    """
    直接用 Matplotlib 生成完整 Fig. S17。
    不再使用 PIL 拼接分图，因此必定产生一张完整四联图。
    """
    fig = plt.figure(figsize=(7.2, 6.0))

    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[0.92, 1.38],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.975,
        wspace=0.28,
        hspace=0.38
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a, coords)
    draw_contrast_panel(
        ax_b,
        rows,
        "delta_branch_db",
        "(b)"
    )
    draw_panel_d(ax_d, rows)
    draw_contrast_panel(
        ax_c,
        rows,
        "delta_peak_db",
        "(c)"
    )

    stem = outdir / "Fig_S17_OC_spatial_support_validation_COMPLETE"
    save_figure_all(fig, stem)
    plt.close(fig)

    return stem


# -----------------------------------------------------------------------------
# CSV 与报告
# -----------------------------------------------------------------------------

def write_csv(path, rows, fieldnames=None):
    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with Path(path).open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    rows,
    site_metrics,
    manifest_rows,
    info,
    args,
    outdir
):
    write_csv(
        outdir / "Fig_S17_OC_pair_metrics_from_raw.csv",
        rows
    )

    site_rows = []
    for site in sorted(site_metrics):
        site_rows.append(
            {"site_id": site, **site_metrics[site]}
        )

    write_csv(
        outdir / "Fig_S17_OC_site_response_metrics.csv",
        site_rows
    )

    write_csv(
        outdir / "Fig_S17_OC_raw_file_manifest.csv",
        manifest_rows
    )

    npos_branch = positive_count(
        rows, "delta_branch_db"
    )
    npos_peak = positive_count(
        rows, "delta_peak_db"
    )
    med_branch = median(
        rows, "delta_branch_db"
    )
    med_peak = median(
        rows, "delta_peak_db"
    )
    nagree = sign_agreement_count(rows)

    lines = [
        "Fig. S17 OC raw-data analysis report",
        "=" * 52,
        f"Raw data directory: {args.data_dir}",
        f"OC branch interval: {OC_WINDOW[0]:.1f}-{OC_WINDOW[1]:.1f} Hz",
        f"Peak-frequency target: {PEAK_TARGET:.4f} Hz",
        f"Actual FFT peak bin: {info['peak_bin_hz']:.4f} Hz",
        f"Sampling frequency: {info['fs_hz']:.3f} Hz",
        f"Frequency resolution: {info['frequency_resolution_hz']:.6f} Hz",
        "",
        f"Branch-integrated contrast: {npos_branch}/{len(rows)} > 0 dB",
        f"Branch-integrated median: {med_branch:+.3f} dB",
        f"Peak-frequency contrast: {npos_peak}/{len(rows)} > 0 dB",
        f"Peak-frequency median: {med_peak:+.3f} dB",
        f"Sign agreement: {nagree}/{len(rows)}",
        "",
        "Pair 33/34 is displayed in panel (a) but excluded from the",
        "common quantitative statistics because site 33 used a different",
        "multichannel acquisition configuration.",
        "",
        "The peak-frequency result is a complementary single-bin check,",
        "not an independent narrow-band dataset.",
    ]

    (outdir / "Fig_S17_OC_analysis_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


def compare_reference(rows):
    diffs_branch = []
    diffs_peak = []

    for row in rows:
        key = row["pair"]
        if key not in REFERENCE_CONTRASTS:
            continue

        rb, rp = REFERENCE_CONTRASTS[key]
        diffs_branch.append(
            float(row["delta_branch_db"]) - rb
        )
        diffs_peak.append(
            float(row["delta_peak_db"]) - rp
        )

    if diffs_branch:
        rmse_b = float(
            np.sqrt(np.mean(np.asarray(diffs_branch)**2))
        )
        rmse_p = float(
            np.sqrt(np.mean(np.asarray(diffs_peak)**2))
        )
        print()
        print("与此前 Fig. S17 结果的回归核对：")
        print(f"  分支积分 RMSE = {rmse_b:.4f} dB")
        print(f"  峰值频率 RMSE = {rmse_p:.4f} dB")


# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "从原始 DAT 重新计算并直接输出完整四联 Fig. S17。"
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name(
            DEFAULT_OUTDIR_NAME
        )
    )

    parser.add_argument("--up-ch", type=int, default=2)
    parser.add_argument("--down-ch", type=int, default=3)
    parser.add_argument("--ref-ch", type=int, default=4)

    parser.add_argument(
        "--nperseg",
        type=int,
        default=32768
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=32768
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--window",
        choices=["hann", "hamming", "rect"],
        default="hann"
    )

    parser.add_argument(
        "--no-detrend",
        action="store_true"
    )
    parser.add_argument(
        "--trim-start",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--trim-end",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--no-recursive-search",
        action="store_true"
    )
    parser.add_argument(
        "--compare-reference",
        action="store_true"
    )

    args = parser.parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    configure_font()
    coords = load_coords()

    (
        rows,
        site_metrics,
        manifest_rows,
        freq,
        phi_cache,
        info,
    ) = process_raw_data(args)

    npos_branch = positive_count(
        rows, "delta_branch_db"
    )
    npos_peak = positive_count(
        rows, "delta_peak_db"
    )
    med_branch = median(
        rows, "delta_branch_db"
    )
    med_peak = median(
        rows, "delta_peak_db"
    )
    nagree = sign_agreement_count(rows)

    print()
    print("=" * 96)
    print("S17 真实数据统计")
    print("=" * 96)
    print(
        f"完整 OC 分支频窗："
        f"{npos_branch}/{len(rows)} 为正，"
        f"中位数={med_branch:+.3f} dB"
    )
    print(
        f"5235.9375 Hz 峰值频率："
        f"{npos_peak}/{len(rows)} 为正，"
        f"中位数={med_peak:+.3f} dB"
    )
    print(
        f"两种频率选择符号一致："
        f"{nagree}/{len(rows)}"
    )

    if args.compare_reference:
        compare_reference(rows)

    # 先输出单独分图
    save_individual_panels(
        coords,
        rows,
        args.outdir
    )

    # 再直接输出一张完整四联图
    complete_stem = save_complete_figure(
        coords,
        rows,
        args.outdir
    )

    write_outputs(
        rows,
        site_metrics,
        manifest_rows,
        info,
        args,
        args.outdir
    )

    print()
    print("=" * 96)
    print("完成。")
    print("=" * 96)
    print(f"输出目录：{args.outdir}")
    print()
    print("完整四联图：")
    print(f"  {complete_stem.with_suffix('.png')}")
    print(f"  {complete_stem.with_suffix('.pdf')}")
    print(f"  {complete_stem.with_suffix('.tiff')}")
    print()
    print("同时保留 S17a / S17b / S17c / S17d 单独分图。")
    print("程序不使用 pandas、PIL 或 scipy。")
    print("所有数据标记均为圆形。")
    print("字体：Times New Roman。")
    print("绘图后端：Agg（直接保存文件，不弹出 PyCharm 绘图窗口）。")


if __name__ == "__main__":
    main()
