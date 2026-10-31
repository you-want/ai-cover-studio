"""混音与母带处理：人声活动检测、静音段噪声门、电平自动对齐、混音、LUFS 母带。"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .dsp import (_rbj_biquad, biquad, highpass, loudness_lufs, normalize_lufs,  # noqa: F401
                   pad_or_trim, soft_limit, to_mono, to_stereo)
from scipy.signal import fftconvolve


def mix_tracks(vocals: np.ndarray, instrumental: np.ndarray,
               vocal_gain_db: float = 0.0, instrumental_gain_db: float = 0.0) -> np.ndarray:
    """长度对齐后线性相加。输入 [C, T]，输出长度取两者较大值。"""
    n = max(vocals.shape[-1], instrumental.shape[-1])
    v = pad_or_trim(np.asarray(vocals, dtype=np.float32), n) * (10 ** (vocal_gain_db / 20.0))
    i = pad_or_trim(np.asarray(instrumental, dtype=np.float32), n) * (10 ** (instrumental_gain_db / 20.0))
    return (i + v).astype(np.float32)


def master(data: np.ndarray, target_db: float = -14.0, limit: float = 0.97,
           sr: int = 44100) -> np.ndarray:
    """母带："开阔度"EQ → BS.1770 LUFS 响度归一（target_db 即目标 LUFS）→ 软限幅。

    人声经过 OpenVoice 等模型转换后 2.5~5kHz 亮度带普遍比原唱暗 7~9dB，
    这里在整条母带上 +3dB 高频搁架提升整体亮度。相比旧版 RMS 估算，
    LUFS 按 K 加权与人耳感知响度对齐，不同歌之间响度更一致；
    峰值用前瞻软限幅控制，避免归一后硬削波。
    """
    y = np.asarray(data, dtype=np.float32)
    y = to_stereo(y)
    b, a = _rbj_biquad("highshelf", sr, 2500.0, gain_db=3.0)
    y = biquad(y, b, a)
    y = normalize_lufs(y, target_lufs=target_db, sr=sr, max_peak=1.0)
    return soft_limit(y, sr, ceiling=limit)


# ---------------------------------------------------------------- 人声活动与电平对齐

def vocal_activity_mask(vocals: np.ndarray, sr: int,
                        frame_s: float = 0.04, hop_s: float = 0.01,
                        floor_db: float = -55.0, rel_db: float = -38.0,
                        min_on_s: float = 0.12, bridge_s: float = 0.25,
                        attack_s: float = 0.02, release_s: float = 0.12) -> np.ndarray:
    """检测"正在演唱"的片段，返回逐样本增益 [0,1]（[T]，单声道时间轴）。

    - 帧能量超过 max(地板, 峰值-rel_db) 视为活跃帧；
    - 去除 <min_on_s 的毛刺，桥接 <bridge_s 的句间停顿；
    - 边缘按 attack/release 斜坡过渡，避免硬切产生咔哒声。
    """
    x = to_mono(vocals)
    n = x.size
    frame = max(1, int(frame_s * sr))
    hop = max(1, int(hop_s * sr))
    if n < frame:
        return np.ones(n, dtype=np.float32)
    n_frames = 1 + (n - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    thr = max(floor_db, float(db.max()) + rel_db)
    active = db > thr

    from scipy.ndimage import binary_closing, binary_opening
    active = binary_opening(active, np.ones(max(1, int(min_on_s / hop_s))))
    active = binary_closing(active, np.ones(max(1, int(bridge_s / hop_s))))

    mask = np.zeros(n, dtype=np.float32)
    for i, a in enumerate(active):
        if a:
            mask[i * hop:i * hop + frame] = 1.0
    # 起音斜坡（onset 之后 rise）；收音用反转后的同函数（offset 之前 fall）
    def _ramp_after_onsets(m: np.ndarray, ramp_n: int) -> np.ndarray:
        idxs = np.arange(m.size)
        starts = np.where(np.diff(m, prepend=0.0) > 0.5, idxs, -1)
        starts = np.maximum.accumulate(starts)
        return np.clip((idxs - starts) / max(ramp_n, 1), 0.0, 1.0)

    g = _ramp_after_onsets(mask, int(attack_s * sr))
    g *= _ramp_after_onsets(mask[::-1], int(release_s * sr))[::-1]
    return (g * mask).astype(np.float32)


def gate_to_source(converted: np.ndarray, source: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """静音段噪声门：原唱人声没在唱的位置，把转换后人声压到静音。

    清理音色转换模型在句间产生的残响/噪声/电流声。返回 (处理后信号, 抑制比例)，
    抑制比例 = 被压低区域占总时长的百分比（写入任务日志）。
    """
    c = np.asarray(converted, dtype=np.float32)
    src = np.asarray(source, dtype=np.float32)
    n = max(c.shape[-1], src.shape[-1])
    c = pad_or_trim(c, n)
    src = pad_or_trim(src, n)
    gain = vocal_activity_mask(src, sr)
    gated = c * gain[None, :] if c.ndim > 1 else c * gain
    frac = float(np.mean(gain < 0.5) * 100.0)
    return gated.astype(np.float32), frac


def auto_vocal_gain(src_vocals: np.ndarray, converted: np.ndarray, sr: int,
                    max_gain_db: float = 9.0) -> float | None:
    """计算把转换后人声响度对齐到原唱人声的增益（dB）。

    只在"原唱正在演唱"的片段测响度（LUFS），避免静音段拉低测量值；
    返回 None 表示任一轨无法测量（如纯伴奏、静音）。
    """
    src = np.asarray(src_vocals, dtype=np.float32)
    conv = np.asarray(converted, dtype=np.float32)
    n = max(src.shape[-1], conv.shape[-1])
    src = pad_or_trim(src, n)
    conv = pad_or_trim(conv, n)
    gain_mask = vocal_activity_mask(src, sr)
    mono_src = to_mono(src) * gain_mask
    mono_conv = to_mono(conv) * gain_mask
    l_src = loudness_lufs(mono_src, sr)
    l_conv = loudness_lufs(mono_conv, sr)
    if l_src is None or l_conv is None:
        return None
    g = float(np.clip(l_src - l_conv, -max_gain_db, max_gain_db))
    return g if abs(g) > 0.15 else 0.0


def control_vocal_peaks(vocals: np.ndarray, sr: int, ceiling: float = 0.88) -> tuple[np.ndarray, float]:
    """人声轨独立峰值控制（软限幅），防止电平对齐/增益后人声超程。

    转换模型输出经响度对齐后瞬时峰值可能超过 1.0：
    - 写 WAV（PCM16）时会硬削波（可闻"破音"）；
    - 混音后母带限幅器要在大幅超程上工作，产生泵感失真。
    先对人声单独软限幅（快攻击/慢释放，只碰瞬时峰值），混音时母带
    限幅器几乎不再介入。返回 (处理后人声, 实际限幅量 dB)。
    """
    y = np.asarray(vocals, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= ceiling:
        return y, 0.0
    limited = soft_limit(y, sr, ceiling=ceiling)
    reduction = 20.0 * np.log10(max(peak, 1e-12) / max(float(np.max(np.abs(limited))), 1e-12))
    return limited, float(reduction)


# ---------------------------------------------------------------- 人声总线

def _compress(x: np.ndarray, sr: int, threshold_db: float = -18.0,
              ratio: float = 3.0, attack_s: float = 0.015,
              release_s: float = 0.18, max_gr_db: float = 12.0) -> np.ndarray:
    """单声道对数域前馈压缩器（密度与稳定性；纯 numpy/scipy）。"""
    from scipy.signal import lfilter
    from scipy.ndimage import maximum_filter1d
    r = float(np.exp(-1.0 / (release_s * sr)))  # 慢收
    rect = np.abs(np.asarray(x, dtype=np.float32))
    # 峰值保持检测：2ms 滑窗取真峰（低通会平均掉瞬态，导致压缩器"看不见"峰），
    # 上升瞬时跟随、下降走慢释放
    peaks = maximum_filter1d(rect, size=max(3, int(0.0025 * sr)))
    env = np.maximum(peaks, lfilter([1 - r], [1, -r], peaks))
    env_db = 20.0 * np.log10(env + 1e-10)
    over = np.maximum(env_db - threshold_db, 0.0)
    gr = np.minimum(over * (1.0 - 1.0 / ratio), max_gr_db)
    # 增益下降瞬时咬合（取大），恢复走慢速滤波——顺序反了会放过瞬态、压扁持续段
    gr_eff = np.maximum(gr, lfilter([1 - r], [1, -r], gr))
    gain = 10.0 ** (-gr_eff / 20.0)
    return (np.asarray(x, dtype=np.float32) * gain).astype(np.float32)


def _plate_reverb(x: np.ndarray, sr: int, rt60_s: float = 1.0,
                  wet: float = 0.16, predelay_s: float = 0.02) -> np.ndarray:
    """合成板式混响（指数衰减噪声 IR + 高频衰减），给人声"进歌"的空间感。"""
    from scipy.signal import lfilter
    rng = np.random.default_rng(1234)
    n = int(rt60_s * sr)
    t = np.arange(n) / sr
    decay = 10.0 ** (-3.0 * t / rt60_s)  # rt60 指数衰减
    bright = rng.standard_normal(n).astype(np.float64)
    dark = lfilter(*_rbj_biquad("lowshelf", sr, 4000.0, gain_db=-9.0), bright)
    ir = (0.55 * bright + 0.45 * dark) * decay
    ramp = max(1, int(0.004 * sr))
    ir[:ramp] *= np.linspace(0, 1, ramp)  # 起振平滑
    ir /= np.sqrt(np.sum(ir ** 2) + 1e-12)
    ir = np.concatenate([np.zeros(int(predelay_s * sr)), ir])
    wet_sig = fftconvolve(x, ir[None, :], mode="full", axes=-1)[..., : x.shape[-1]]
    return (x * (1.0 - wet) + wet_sig * wet).astype(np.float32)


def vocal_bus(vocals: np.ndarray, sr: int) -> np.ndarray:
    """人声总线：低切去浊 → 压缩增密 → 去刺 → 板式混响粘合。

    转换模型输出普遍低频浑浊、动态松散、且去混响后干得像卡拉OK——
    这条链是商业人声制作的极简版，把人声"做进"伴奏里。
    """
    y = np.asarray(vocals, dtype=np.float32)
    # 1) 低切 70Hz（去 rumble）+ 280Hz 浊频 -2dB
    y = highpass(y, sr, 70.0)
    b, a = _rbj_biquad("peaking", sr, 280.0, gain_db=-2.0, q=0.9)
    y = biquad(y, b, a)
    # 2) 压缩（2:1 @ -13dB）拉回密度（按声道独立，立体声像不塌陷）——
    #    参数在真实 SVC 人声上标定：crest 7.0 → 4.5，与商业人声密度一致
    out = np.empty_like(y)
    for c in range(y.shape[0]):
        out[c] = _compress(y[c], sr, threshold_db=-13.0, ratio=2.0)
    y = out
    # 3) 3.5kHz 轻微去刺
    b, a = _rbj_biquad("peaking", sr, 3500.0, gain_db=-1.2, q=1.2)
    y = biquad(y, b, a)
    # 4) 板式混响粘合进混音
    return _plate_reverb(y, sr)


__all__ = ["mix_tracks", "master", "vocal_activity_mask", "gate_to_source",
           "auto_vocal_gain", "control_vocal_peaks", "vocal_bus"]
