"""演示级声音引擎（永远可用，零依赖）。

"音域适配"：把原唱人声的整体音高适配到用户的中位基频（音域），
并按用户音色的亮暗特征做 EQ 微调。这不是真正的音色克隆，
但零模型下载、秒级完成，作为兜底与效果对照。

用户也可以在界面上手动微调半音数，实现男女声调对调等玩法。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio.dsp import (_rbj_biquad, biquad, fade, median_f0, peak_normalize,
                         pitch_shift, spectral_centroid)
from ..audio.io import load_wav, save_wav
from .base import BaseVoiceEngine, ProgressCB, VoiceProfile


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PitchFitVoiceEngine(BaseVoiceEngine):
    name = "pitchfit"
    label = "音域适配（演示引擎）"
    kind = "demo"
    description = ("离线 DSP：把原唱人声音高适配到你的音域，并按你的音色亮暗做 EQ 微调。"
                   "零模型下载、秒级完成；效果为演示级，用于快速试听与兜底。")

    def convert(self, source_vocal: Path, profile: VoiceProfile, out_wav: Path,
                pitch_shift_semitones: float = 0.0,
                progress: ProgressCB = lambda p, m: None) -> dict:
        progress(0.1, "读取人声")
        data, sr = load_wav(source_vocal)
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)

        progress(0.25, "估计原唱音域")
        f0_src = median_f0(data, sr)
        notes: dict[str, str] = {}
        n_steps = 0.0
        if profile.f0_median and f0_src:
            raw = 12.0 * np.log2(profile.f0_median / f0_src)
            n_steps = float(_clamp(raw, -7.0, 7.0))
            if abs(raw) > 7.0:
                notes["fit_clamped"] = (f"你的音域与原唱差距较大（{raw:+.1f} 半音），"
                                        f"已限制在 ±7 半音内（实际 {n_steps:+.1f}）")
            notes["pitch_fit"] = (f"原唱中位基频 {f0_src:.0f}Hz → 你的 {profile.f0_median:.0f}Hz，"
                                  f"自动适配 {n_steps:+.1f} 半音")
        if abs(pitch_shift_semitones) > 1e-3:
            notes["manual_pitch"] = f"手动音高微调 {pitch_shift_semitones:+.1f} 半音"
        n_steps += float(pitch_shift_semitones)

        progress(0.45, "音高适配")
        y = pitch_shift(data, sr, n_steps) if abs(n_steps) > 1e-3 else data

        progress(0.75, "音色微调")
        c_src = spectral_centroid(data, sr)
        if profile.centroid and c_src:
            tilt = 6.0 * np.log2(max(profile.centroid, 1.0) / max(c_src, 1.0))
            tilt = float(_clamp(tilt, -6.0, 6.0))
            if abs(tilt) > 0.3:
                b, a = _rbj_biquad("highshelf", sr, 2500.0, gain_db=tilt)
                y = biquad(y, b, a)
                notes["timbre_eq"] = f"按你的音色亮暗做高频搁架均衡 {tilt:+.1f}dB"

        y = peak_normalize(y, 0.9)
        y = fade(y, sr, in_s=0.015, out_s=0.05)
        progress(1.0, "转换完成")
        save_wav(out_wav, y, sr)
        return notes
