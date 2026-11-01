"""副歌定位：找全曲"最像高潮"的片段（能量峰值滑动窗口 + 中后段位置先验）。"""
from __future__ import annotations

import numpy as np


def find_chorus(data: np.ndarray, sr: int, clip_s: float = 40.0) -> tuple[float, float]:
    """返回高潮片段的 (start_s, end_s)。data: [C, T] 或 [T]。"""
    mono = data.mean(axis=0)
    hop = max(1, sr // 2)  # 0.5 秒一帧
    n_frames = max(1, (len(mono) - hop) // hop)
    idx = np.arange(hop)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(mono[idx] ** 2, axis=1) + 1e-12)
    k = max(1, 6)  # 3 秒平滑
    energy = np.convolve(rms, np.ones(k) / k, mode="same")

    win = max(1, int(clip_s / 0.5))
    if n_frames <= win:  # 歌比片段还短，全曲即高潮
        return 0.0, len(mono) / sr
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    sums = csum[win:] - csum[:-win]
    pos = np.linspace(0.0, 1.0, len(sums))
    weight = 0.65 + 0.35 * np.clip(1.0 - np.abs(pos - 0.6) / 0.45, 0.0, 1.0)
    best = int(np.argmax(sums * weight))
    start_s = best * 0.5
    return start_s, start_s + clip_s
