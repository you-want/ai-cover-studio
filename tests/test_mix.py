"""混音与母带测试。"""
import numpy as np

from conftest import SR, tone
from app.audio.dsp import loudness_lufs, rms_db
from app.audio.mix import (auto_vocal_gain, control_vocal_peaks, gate_to_source,
                           master, mix_tracks, pad_or_trim, vocal_activity_mask)


def test_mix_aligns_length():
    v = tone(2.0, 440.0)
    i = tone(3.0, 220.0)
    out = mix_tracks(np.stack([v, v]), np.stack([i, i]))  # 立体声 [2, T]
    assert out.shape == (2, 3 * SR)


def test_mix_gain():
    v = tone(1.0, 440.0)[None, :]
    i = np.zeros((1, SR), dtype=np.float32)
    out = mix_tracks(v, i, vocal_gain_db=6.0)
    ref = mix_tracks(v, i, vocal_gain_db=0.0)
    ratio_db = rms_db(out) - rms_db(ref)
    assert abs(ratio_db - 6.0) < 0.3


def test_master_lufs_and_peak():
    y = (tone(2.0, 300.0, amp=0.05))[None, :]
    out = master(y, target_db=-14.0)
    assert np.abs(out).max() <= 0.98
    # 母带归一到 -14 LUFS（BS.1770），而不是 RMS
    assert abs(loudness_lufs(out, SR) - (-14.0)) < 0.5


def test_vocal_activity_mask():
    v = np.zeros((2, SR * 4), dtype=np.float32)
    v[:, SR:2 * SR] = tone(1.0, 300.0, amp=0.5)[None, :]
    m = vocal_activity_mask(v, SR)
    assert m[:SR // 2].max() < 0.01                              # 深处静音段不活跃
    active = m[SR + SR // 5:2 * SR - SR // 5]
    assert active.min() > 0.9                                    # 演唱段保持活跃


def test_gate_to_source_cleans_silence():
    src = np.zeros((2, SR * 4), dtype=np.float32)
    src[:, SR:2 * SR] = tone(1.0, 300.0, amp=0.5)[None, :]
    rng = np.random.default_rng(0)
    conv = src * 0.5 + (0.02 * rng.standard_normal((2, SR * 4))).astype(np.float32)
    gated, pct = gate_to_source(conv, src, SR)
    assert pct > 50                                              # 大部分时长被抑制
    assert np.abs(gated[:, :SR // 2]).max() < 0.002              # 静音中段几乎清零
    assert np.abs(gated[:, SR + SR // 10]).max() > 0.1           # 演唱段信号保留


def test_auto_vocal_gain():
    st = np.stack([tone(2.0, 300.0, amp=0.5)] * 2)
    g = auto_vocal_gain(st, st * 0.5, SR)
    assert abs(g - 6.0) < 0.6                                    # -6dB 差 → +6dB 增益
    g2 = auto_vocal_gain(st, st * 0.25, SR)
    assert abs(g2 - 9.0) < 0.6                                   # 差距过大时限幅 ±9dB
    assert auto_vocal_gain(st, np.zeros_like(st), SR) is None    # 静音无法测量


def test_control_vocal_peaks():
    st = np.stack([tone(2.0, 300.0, amp=0.8)] * 2)
    out, db = control_vocal_peaks(st, SR, ceiling=0.5)
    assert float(np.abs(out).max()) <= 0.5001      # 峰值被压到 ceiling
    assert db > 4.0                                # 0.8 → 0.5 约 -4.1dB
    # 未超程直通
    same, db0 = control_vocal_peaks(st, SR, ceiling=0.9)
    assert np.allclose(same, st) and db0 == 0.0


def test_pad_or_trim():
    y = np.ones((1, 100), dtype=np.float32)
    assert pad_or_trim(y, 150).shape[-1] == 150
    assert pad_or_trim(y, 50).shape[-1] == 50
