"""中心声道提取：经典卡拉OK式人声/伴奏分离（演示级质量，零依赖、离线秒级）。

立体声混音中人声通常位于正中央，伴奏乐器多分布在两侧：
  人声   ≈ 中置通道 (L+R)/2 的中频段
  伴奏   ≈ 边带 (L-R)/2 + 中置的低频(贝斯/底鼓)与高频(镲片/气声)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio.dsp import highpass, lowpass, peak_normalize
from ..audio.io import load_wav, save_wav
from .base import BaseSeparator, ProgressCB, SeparationResult


class CenterExtractSeparator(BaseSeparator):
    name = "center"
    label = "中心声道提取（快速兜底）"
    kind = "dsp"
    description = ("经典卡拉OK式分离，纯 DSP 零依赖、秒级完成。"
                   "质量为演示级：宽立体声歌曲效果尚可，单声道/人声居中混音的歌曲效果有限。")

    def separate(self, song_wav: Path, out_dir: Path,
                 progress: ProgressCB = lambda p, m: None) -> SeparationResult:
        progress(0.1, "读取音频")
        data, sr = load_wav(song_wav)
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
        L, R = data[0], data[1]

        mid = (L + R) / 2.0
        side = (L - R) / 2.0

        progress(0.4, "频带切分")
        low = lowpass(mid, sr, 160.0)
        high = highpass(mid, sr, 7200.0)
        band = mid - low - high

        progress(0.75, "生成两轨")
        vocals = peak_normalize(np.stack([band, band]), 0.95)
        instrumental = peak_normalize(np.stack([low + high + side, low + high - side]), 0.95)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        v = save_wav(out_dir / "vocals.wav", vocals, sr)
        i = save_wav(out_dir / "no_vocals.wav", instrumental, sr)
        progress(1.0, "分离完成")
        return SeparationResult(vocals=v, instrumental=i)
