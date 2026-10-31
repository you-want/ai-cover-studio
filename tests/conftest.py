"""测试公共夹具：使用独立临时数据目录，并生成合成音频。"""
import os
import sys
import tempfile
from pathlib import Path

# 必须在导入 app 之前设置：让 config 指向测试专用目录
_TMP = tempfile.mkdtemp(prefix="vc_test_")
os.environ["VC_DATA_DIR"] = _TMP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import soundfile as sf

SR = 44100


def tone(dur: float, freq: float, sr: int = SR, vibrato_cents: float = 25.0,
         harmonics=(1.0, 0.5, 0.25), amp: float = 0.5) -> np.ndarray:
    """带颤音与泛音的"人声化"音色。"""
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = freq * (2 ** ((vibrato_cents / 1200.0) * np.sin(2 * np.pi * 5.0 * t)))
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros(n, dtype=np.float64)
    for i, a in enumerate(harmonics, start=1):
        y += a * np.sin(i * phase)
    y /= max(1e-9, np.abs(y).max())
    return (y * amp).astype(np.float32)


def make_sample_wav_bytes(seconds: float = 8.0, freq: float = 200.0) -> bytes:
    """模拟用户录音（低音域男声感）。"""
    import io
    buf = io.BytesIO()
    sf.write(buf, tone(seconds, freq, harmonics=(1.0, 0.4, 0.2)), SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def make_song_wav_bytes(seconds: float = 4.0, melody_freq: float = 440.0) -> bytes:
    """合成一首测试"歌曲"：中置人声旋律 + 宽展伴奏噪声 + 中置低频贝斯。"""
    import io
    rng = np.random.default_rng(42)
    n = int(SR * seconds)
    vocal = tone(seconds, melody_freq, amp=0.7)
    bass = tone(seconds, 80.0, harmonics=(1.0,), amp=0.4)
    side = (0.3 * rng.standard_normal(n)).astype(np.float32)
    center = vocal + bass
    left = center + side
    right = center - side
    buf = io.BytesIO()
    sf.write(buf, np.stack([left, right], axis=1), SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def dominant_freq(y: np.ndarray, sr: int = SR) -> float:
    """主频（幅度最大谱峰）。"""
    x = np.asarray(y, dtype=np.float64).flatten()
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(freqs[int(np.argmax(spec))])


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    with TestClient(app) as c:
        yield c
