"""DSP 层测试：变调不变速、变速不变长、基频估计、LUFS 响度、限幅与淡入淡出。"""
import numpy as np
import pytest

from conftest import SR, dominant_freq, tone
from app.audio.dsp import (fade, loudness_lufs, median_f0, pitch_shift,
                           soft_limit, time_stretch)


def test_median_f0_basic():
    f0 = median_f0(tone(1.5, 220.0), SR)
    assert f0 is not None
    assert abs(f0 - 220.0) / 220.0 < 0.05


def test_median_f0_noise_returns_none():
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 2) * 0.3).astype(np.float32)
    assert median_f0(noise, SR) is None


@pytest.mark.parametrize("steps,base", [(+7, 220.0), (-5, 440.0), (+3.5, 330.0)])
def test_pitch_shift_frequency(steps, base):
    y = tone(1.5, base)
    out = pitch_shift(y, SR, steps)
    expected = base * (2 ** (steps / 12.0))
    got = dominant_freq(out)
    assert abs(got - expected) / expected < 0.03, f"expect {expected:.1f}Hz, got {got:.1f}Hz"


def test_pitch_shift_preserves_length():
    y = tone(2.0, 220.0)
    out = pitch_shift(y, SR, -4)
    assert abs(len(out) - len(y)) / len(y) < 0.01


def test_time_stretch_shortens():
    y = tone(2.0, 300.0)
    out = time_stretch(y, 1.5, SR)
    assert abs(len(out) - len(y) / 1.5) / len(y) < 0.02
    # 音高应基本不变
    assert abs(dominant_freq(out) - 300.0) / 300.0 < 0.03


# ---------------------------------------------------------------- LUFS / 限幅 / 淡入淡出

def test_lufs_linearity_and_absolute():
    sr = SR
    t = np.arange(sr * 3) / sr
    sine = np.sin(2 * np.pi * 997 * t).astype(np.float32)
    st = np.stack([sine, sine])
    l0 = loudness_lufs(st, sr)
    l1 = loudness_lufs(st * 0.5, sr)
    assert abs((l0 - l1) - 6.0) < 0.1            # -6dB 电平变化 → 6 LU 读数差
    # 997Hz 立体声 -20dBFS 峰值 → 约 -20.3 LUFS（与 ffmpeg ebur128 实测一致）
    assert abs(loudness_lufs(st * 0.1, sr) - (-20.3)) < 0.5


def test_lufs_silence_returns_none():
    assert loudness_lufs(np.zeros((2, SR)), SR) is None
    assert loudness_lufs(np.zeros(SR), SR) is None


def test_soft_limit_caps_peak_and_passthrough():
    y = tone(1.0, 440.0, amp=0.5)[None, :]
    assert np.allclose(soft_limit(y, SR, ceiling=0.97), y)   # 未超限直通
    sp = y.copy()
    sp[0, SR // 2] = 5.0                                     # 注入超高峰值
    out = soft_limit(sp, SR, ceiling=0.97)
    assert np.abs(out).max() <= 0.9701   # float32 量化容差
    # 远离尖峰处采样保持不变（限幅只作用于局部）
    assert abs(out[0, 100] - y[0, 100]) < 1e-6


def test_fade_edges():
    y = np.ones((2, SR), dtype=np.float32)
    out = fade(y, SR, in_s=0.01, out_s=0.02)
    assert out[0, 0] == 0.0 and out[0, -1] == 0.0
    assert out[0, SR // 2] == 1.0
