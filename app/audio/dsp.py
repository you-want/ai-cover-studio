"""纯 numpy/scipy 的音频 DSP 工具：重采样、滤波、RBJ 均衡器、
WSOLA 变速不变调、变调不变速、基频/音色分析、响度归一。

不依赖 librosa/torch，保证核心链路在任何环境（含离线）可用。
"""
from __future__ import annotations

import numpy as np
from fractions import Fraction
from scipy.signal import butter, lfilter, resample_poly, sosfilt


# ---------------------------------------------------------------- 基础变换

def to_mono(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        return data
    return data.mean(axis=0).astype(np.float32)


def to_stereo(data: np.ndarray) -> np.ndarray:
    """统一为 [2, T] 立体声。"""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        return np.stack([data, data])
    if data.shape[0] == 1:
        return np.repeat(data, 2, axis=0)
    return data


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """多晶相位重采样，支持 [T] 或 [C, T]。"""
    if int(sr_in) == int(sr_out):
        return np.asarray(data, dtype=np.float32)
    f = Fraction(int(sr_out), int(sr_in))
    axis = -1 if np.asarray(data).ndim > 1 else 0
    out = resample_poly(np.asarray(data, dtype=np.float64), f.numerator, f.denominator, axis=axis)
    return out.astype(np.float32)


# ---------------------------------------------------------------- 滤波/均衡

def _apply_sos(data: np.ndarray, sos) -> np.ndarray:
    axis = -1 if np.asarray(data).ndim > 1 else 0
    out = sosfilt(sos, np.asarray(data, dtype=np.float64), axis=axis)
    return out.astype(np.float32)


def _apply_sos_filtfilt(data: np.ndarray, sos) -> np.ndarray:
    """零相位滤波（正反各一次）。对超短输入自动退化为普通滤波。"""
    x = np.asarray(data, dtype=np.float64)
    axis = -1 if x.ndim > 1 else 0
    try:
        from scipy.signal import sosfiltfilt
        out = sosfiltfilt(sos, x, axis=axis)
    except Exception:
        out = sosfilt(sos, x, axis=axis)
    return out.astype(np.float32)


def lowpass(data: np.ndarray, sr: int, freq: float, order: int = 4) -> np.ndarray:
    return _apply_sos_filtfilt(data, butter(order, freq, btype="low", fs=sr, output="sos"))


def highpass(data: np.ndarray, sr: int, freq: float, order: int = 4) -> np.ndarray:
    return _apply_sos_filtfilt(data, butter(order, freq, btype="high", fs=sr, output="sos"))


def _rbj_biquad(kind: str, sr: int, freq: float, gain_db: float = 0.0, q: float = 0.707):
    """RBJ Audio EQ Cookbook 双二阶滤波器系数。

    支持 highshelf / lowshelf / peaking（lowshelf 为 RBJ "Low-shelf 90dB/oct" 配方）。
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * float(freq) / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    sqA = 2 * np.sqrt(A) * alpha
    if kind == "highshelf":
        b = [A * ((A + 1) + (A - 1) * cw + sqA),
             -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sqA)]
        a = [(A + 1) - (A - 1) * cw + sqA,
             2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - sqA]
    elif kind == "lowshelf":
        # RBJ low shelf：a1 = -2((A-1) + (A+1)cos)，此前误用 highshelf 的
        # a1 形式导致非 0dB 增益时滤波器不稳定（极点半径 >1）
        b = [A * ((A + 1) - (A - 1) * cw + sqA),
             2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - sqA)]
        a = [(A + 1) + (A - 1) * cw + sqA,
             -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - sqA]
    elif kind == "peaking":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        raise ValueError(f"不支持的滤波器类型: {kind}")
    b = np.array(b, dtype=np.float64) / a[0]
    a = np.array(a, dtype=np.float64) / a[0]
    return b, a


def biquad(data: np.ndarray, b, a) -> np.ndarray:
    axis = -1 if np.asarray(data).ndim > 1 else 0
    out = lfilter(b, a, np.asarray(data, dtype=np.float32), axis=axis)
    return out.astype(np.float32)


# ---------------------------------------------------------------- WSOLA 变速/变调

def _best_offset(mono: np.ndarray, tail: np.ndarray, ideal: int, lo: int, hi: int) -> int:
    """在 [lo, hi] 中寻找与 tail 最匹配的起点（粗搜 + 细搜）。"""
    L = len(tail)
    step = max(1, (hi - lo) // 96)
    best_s, best_v = ideal, -np.inf
    for s in range(lo, hi + 1, step):
        v = float(np.dot(tail, mono[s:s + L]))
        if v > best_v:
            best_v, best_s = v, s
    for s in range(max(lo, best_s - step), min(hi, best_s + step) + 1):
        v = float(np.dot(tail, mono[s:s + L]))
        if v > best_v:
            best_v, best_s = v, s
    return best_s


def time_stretch(data: np.ndarray, speed: float, sr: int) -> np.ndarray:
    """WSOLA 变速不变调。

    speed > 1 表示加速（输出时长 = 输入时长 / speed），音高不变。
    支持 [T] 或 [C, T]（一维输入返回一维）；多声道共用同一对齐序列，避免声道间相位撕裂。
    """
    speed = float(speed)
    y = np.asarray(data, dtype=np.float32)
    was_1d = y.ndim == 1
    if was_1d:
        y = y[None, :]
    if abs(speed - 1.0) < 1e-4:
        return y.reshape(-1) if was_1d else y.copy()
    n_ch, n = y.shape
    if n < 1024 or n_ch < 1:
        return y.reshape(-1) if was_1d else y.copy()

    frame = 1 << int(np.log2(max(0.046 * sr, 512)))   # ~2048 @44.1kHz
    hop = frame // 2
    search = 1 << int(np.log2(max(0.006 * sr, 64)))
    win = np.hanning(frame).astype(np.float32)

    out_total = max(int(round(n / speed)), frame)
    out = np.zeros((n_ch, out_total + frame * 2), dtype=np.float32)
    wsum = np.zeros((1, out_total + frame * 2), dtype=np.float32)

    mono = y.mean(axis=0)
    in_pos = 0.0
    out_pos = 0
    prev_tail: np.ndarray | None = None
    start = 0
    while out_pos < out_total and start + frame < n:
        ideal = int(in_pos)
        if prev_tail is None:
            start = max(0, min(ideal, n - frame))
        else:
            lo = max(0, ideal - search)
            hi = min(n - frame, ideal + search)
            start = _best_offset(mono, prev_tail, ideal, lo, hi)
        if start + frame > n:
            break
        out[:, out_pos:out_pos + frame] += y[:, start:start + frame] * win
        wsum[:, out_pos:out_pos + frame] += win
        prev_tail = mono[start + hop:start + frame].copy()
        out_pos += hop
        in_pos += hop * speed
    out /= np.maximum(wsum, 1e-6)
    out = out[:, :out_total]
    return out.reshape(-1) if was_1d else out


def pitch_shift(data: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """变调不变速（WSOLA + 重采样组合）。n_steps 为半音数，可为小数。

    原理：先 WSOLA 拉伸（升调拉长 s 倍 / 降调压短），再重采样恢复原时长（音高随缩放改变）。
    输出长度与输入完全一致。
    """
    if abs(float(n_steps)) < 1e-3:
        y = np.asarray(data, dtype=np.float32)
        return y.copy()
    s = 2.0 ** (float(n_steps) / 12.0)
    stretched = time_stretch(data, 1.0 / s, sr)        # 时长 × s，音高不变
    f = Fraction(1.0 / s).limit_denominator(100000)
    axis = -1 if np.asarray(stretched).ndim > 1 else 0
    out = resample_poly(np.asarray(stretched, dtype=np.float64), f.numerator, f.denominator, axis=axis)
    out = out.astype(np.float32)
    # 消除重采样取整误差，保证输出长度与输入一致
    out = pad_or_trim(out, np.asarray(data).shape[-1])
    return out


# ---------------------------------------------------------------- 声学分析

def median_f0(data: np.ndarray, sr: int, fmin: float = 65.0, fmax: float = 900.0,
              frame_s: float = 0.05, hop_s: float = 0.025,
              min_voiced: int = 6) -> float | None:
    """基于 FFT 自相关 + 抛物线细化的中位基频估计。

    返回 None 表示 voiced 帧不足（静音/噪声/过短）。
    """
    y = to_mono(data)
    frame = int(frame_s * sr)
    hop = int(hop_s * sr)
    if len(y) < frame * 2:
        return None
    lo_lag = max(1, int(sr / fmax))
    hi_lag = min(frame - 1, int(sr / fmin))
    if hi_lag <= lo_lag + 2:
        return None
    rms_all = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    nfft = 1 << int(2 * frame - 1).bit_length()
    f0s: list[float] = []
    for i in range(0, len(y) - frame, hop):
        x = y[i:i + frame]
        if np.sqrt(np.mean(x ** 2) + 1e-12) < 0.5 * rms_all:
            continue
        x = x - x.mean()
        spec = np.fft.rfft(x, nfft)
        ac = np.fft.irfft(spec * np.conj(spec))[:frame]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo_lag:hi_lag]
        k = int(np.argmax(seg))
        if float(seg[k]) < 0.30:
            continue
        lag = lo_lag + k
        if 0 < k < len(seg) - 1:
            a_, b_, c_ = float(seg[k - 1]), float(seg[k]), float(seg[k + 1])
            denom = a_ - 2 * b_ + c_
            if abs(denom) > 1e-9:
                lag += 0.5 * (a_ - c_) / denom
        f0 = sr / max(lag, 1.0)
        if fmin <= f0 <= fmax:
            f0s.append(f0)
    if len(f0s) < min_voiced:
        return None
    return float(np.median(f0s))


def spectral_centroid(data: np.ndarray, sr: int, max_seconds: float = 20.0) -> float | None:
    """频谱质心（音色亮暗的粗略度量），Hz。"""
    y = to_mono(data)
    step = int(max_seconds * sr)
    if len(y) > step:
        y = y[:step]
    if len(y) < 256 or float(np.sqrt(np.mean(y ** 2))) < 1e-4:
        return None
    x = y - y.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)).astype(np.float32)))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total = float(spec.sum())
    if total <= 0:
        return None
    return float((freqs * spec).sum() / total)


def hf_energy_ratio(data: np.ndarray, sr: int, freq: float = 5000.0,
                    max_seconds: float = 30.0) -> float | None:
    """>freq 频段的能量占比（0~1），衡量人声"闷不闷"；静音返回 None。"""
    y = to_mono(data)
    step = int(max_seconds * sr)
    if len(y) > step:
        y = y[:step]
    if len(y) < 1024:
        return None
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)).astype(np.float32)))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    total = float(spec.sum())
    if total <= 1e-9:
        return None
    return float(spec[freqs > freq].sum() / total)


# ---------------------------------------------------------------- LUFS 响度（ITU-R BS.1770-4）

# K 加权两级滤波器设计常数（ITU-R BS.1770-4 规定的 shelving + high-pass）
_K_SHELF_F0 = 1681.9744509555319
_K_SHELF_GAIN_DB = 3.999843853973347
_K_SHELF_Q = 0.7071752369554196
_K_HP_F0 = 38.13547087602444
_K_HP_Q = 0.5003270373238773


def _k_weight(data: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 K 加权：高频搁架（模拟头部声学）+ 高通（Rlb）。
    输入 [C, T] 或 [T]，输出同形状（float64，便于后续累计均方）。"""
    y = np.asarray(data, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    b_sh, a_sh = _rbj_biquad("highshelf", sr, _K_SHELF_F0, gain_db=_K_SHELF_GAIN_DB, q=_K_SHELF_Q)
    w0 = 2 * np.pi * _K_HP_F0 / sr
    alpha = np.sin(w0) / (2 * _K_HP_Q)
    cw = np.cos(w0)
    b_hp = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]) / (1 + alpha)
    a_hp = np.array([1 + alpha, -2 * cw, 1 - alpha]) / (1 + alpha)
    out = np.empty_like(y)
    for c in range(y.shape[0]):
        t = lfilter(b_sh, a_sh, y[c])
        out[c] = lfilter(b_hp, a_hp, t)
    return out


def _block_mean_square(k_weighted: np.ndarray, sr: int,
                       block_s: float = 0.4, hop_s: float = 0.1) -> np.ndarray:
    """按 400ms 块 / 100ms 步进（75% 重叠）计算各声道均方能量，返回 [n_blocks, C]。"""
    n_ch, n = k_weighted.shape
    block = int(block_s * sr)
    hop = int(hop_s * sr)
    if n < block:
        return np.empty((0, n_ch), dtype=np.float64)
    sq = k_weighted ** 2
    cs = np.zeros((n_ch, n + 1), dtype=np.float64)
    np.cumsum(sq, axis=1, out=cs[:, 1:])
    starts = np.arange(0, n - block + 1, hop)
    # ms[b, c] = (cs[b+block] - cs[b]) / block
    ms = (cs[:, starts + block] - cs[:, starts]).T / block
    return ms


def loudness_lufs(data: np.ndarray, sr: int) -> float | None:
    """综合响度 LUFS（ITU-R BS.1770-4，含绝对 -70LU 与相对 -10LU 门限）。

    data 为 [C, T] 或 [T]；返回 None 表示整段静音（无法测量）。
    """
    kw = _k_weight(data, sr)
    ms = _block_mean_square(kw, sr)
    if ms.shape[0] == 0:
        return None
    ch_weight = np.ones(ms.shape[1], dtype=np.float64)  # L/R 权重 1.0
    block_ms = ms @ ch_weight
    j = -0.691 + 10.0 * np.log10(np.maximum(block_ms, 1e-24))
    # 绝对门限：丢弃低于 -70 LUFS 的块
    keep = j >= -70.0
    if not keep.any():
        return None
    l_ungated = -0.691 + 10.0 * np.log10(float(block_ms[keep].mean()))
    # 相对门限：再丢弃低于（未门限响度 - 10 LU）的块
    keep2 = keep & (j >= l_ungated - 10.0)
    if not keep2.any():
        return l_ungated
    return -0.691 + 10.0 * np.log10(float(block_ms[keep2].mean()))


def normalize_lufs(data: np.ndarray, target_lufs: float = -14.0, sr: int = 44100,
                   max_peak: float = 0.97) -> np.ndarray:
    """响度归一到目标 LUFS；静音段直接返回原数据，峰值超限时回退缩放。"""
    y = np.asarray(data, dtype=np.float32)
    l = loudness_lufs(y, sr)
    if l is None:
        return y
    y = y * (10 ** ((target_lufs - l) / 20.0))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > max_peak:
        y = y * (max_peak / peak)
    return y.astype(np.float32)


# ---------------------------------------------------------------- 限幅与淡入淡出

def soft_limit(data: np.ndarray, sr: int, ceiling: float = 0.97,
               lookahead_s: float = 0.005, release_s: float = 0.08) -> np.ndarray:
    """前瞻软限幅器：瞬时攻击 + 指数释放，把峰值压到 ceiling 以内且不产生硬削波失真。

    原理：以 5ms 块取峰值包络（含一块前瞻），得到目标增益曲线后，
    用"一阶低通 + 取小"实现快攻击/慢释放的增益平滑。
    """
    y = np.asarray(data, dtype=np.float32)
    if y.size == 0:
        return y
    was_1d = y.ndim == 1
    if was_1d:
        y = y[None, :]
    peak = np.max(np.abs(y), axis=0)
    if float(peak.max()) <= ceiling:
        return y.reshape(-1) if was_1d else y

    n = peak.size
    block = max(1, int(lookahead_s * sr))
    pad = (-n) % block
    pb = np.pad(peak, (0, pad), mode="constant").reshape(-1, block).max(axis=1)
    # 前瞻一块：每个块的增益受下一个块的峰值约束，保证攻击"提前"生效
    env_b = np.empty_like(pb)
    env_b[:-1] = np.maximum(pb[:-1], pb[1:])
    env_b[-1] = pb[-1]
    env = np.repeat(env_b, block)[:n]
    g_t = np.minimum(1.0, ceiling / np.maximum(env, 1e-12)).astype(np.float64)

    # 释放平滑：线性一阶低通只放慢"增益回升"，配合 min 实现瞬时压低。
    # 用稳态初值初始化滤波器，避免曲子开头增益从 0 缓慢爬升造成假淡入
    coef = float(np.exp(-1.0 / max(release_s * sr, 1.0)))
    from scipy.signal import lfilter_zi
    zi = lfilter_zi([1 - coef], [1.0, -coef]) * g_t[0]
    g_smooth = lfilter([1 - coef], [1.0, -coef], g_t, zi=zi)[0]
    g = np.minimum(g_t, g_smooth).astype(np.float32)
    out = y * g[None, :]
    return out.reshape(-1) if was_1d else out


def fade(data: np.ndarray, sr: int, in_s: float = 0.015, out_s: float = 0.03) -> np.ndarray:
    """首尾淡入淡出，消除处理链在边界处的咔哒声。支持 [T] 或 [C, T]。"""
    y = np.asarray(data, dtype=np.float32)
    n = y.shape[-1]
    n_in = min(n // 2, max(1, int(in_s * sr)))
    n_out = min(n // 2, max(1, int(out_s * sr)))
    ramp_in = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    y = y.copy()
    y[..., :n_in] *= ramp_in
    y[..., n - n_out:] *= ramp_out
    return y


# ---------------------------------------------------------------- 响度与长度

def pad_or_trim(data: np.ndarray, length: int) -> np.ndarray:
    if data.shape[-1] >= length:
        return data[..., :length]
    pad = length - data.shape[-1]
    return np.pad(data, [(0, 0)] * (data.ndim - 1) + [(0, pad)])


def rms_db(data: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2)))
    return 20.0 * np.log10(r + 1e-12)


def normalize_loudness(data: np.ndarray, target_db: float = -14.0, max_peak: float = 0.97) -> np.ndarray:
    y = np.asarray(data, dtype=np.float32)
    gain = target_db - rms_db(y)
    y = y * (10 ** (gain / 20.0))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > max_peak:
        y = y * (max_peak / peak)
    return y.astype(np.float32)


def peak_normalize(data: np.ndarray, peak: float = 0.9) -> np.ndarray:
    y = np.asarray(data, dtype=np.float32)
    p = float(np.max(np.abs(y))) if y.size else 0.0
    if p > 1e-9:
        y = y * (float(peak) / p)
    return y.astype(np.float32)
