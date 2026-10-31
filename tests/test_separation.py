"""人声分离测试（中心声道兜底分离器）。"""
import numpy as np

from conftest import SR, dominant_freq, tone
from app.audio.dsp import to_stereo
from app.audio.io import load_wav, save_wav
from app.separation import resolve_separator
from app.separation.center_extract import CenterExtractSeparator


def _band_energy(y, sr, freq, width=8.0):
    x = np.asarray(y, dtype=np.float64).flatten()
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = (freqs >= freq - width) & (freqs <= freq + width)
    return float(spec[mask].sum())


def test_center_extract_separates(tmp_path):
    song = tmp_path / "song.wav"
    n = int(SR * 3.0)
    vocal = tone(3.0, 440.0, amp=0.7)
    bass = tone(3.0, 80.0, harmonics=(1.0,), amp=0.4)
    rng = np.random.default_rng(1)
    side = (0.3 * rng.standard_normal(n)).astype(np.float32)
    center = vocal + bass
    save_wav(song, np.stack([center + side, center - side]), SR)

    res = CenterExtractSeparator().separate(song, tmp_path / "stems")
    voc, sr1 = load_wav(res.vocals)
    inst, sr2 = load_wav(res.instrumental)
    assert sr1 == sr2 == SR

    e_voc_melody = _band_energy(voc, SR, 440.0)
    e_inst_melody = _band_energy(inst, SR, 440.0)
    # 旋律应主要留在人声轨
    assert e_voc_melody > 3.0 * max(e_inst_melody, 1e-12)

    e_voc_bass = _band_energy(voc, SR, 80.0)
    e_inst_bass = _band_energy(inst, SR, 80.0)
    # 低频贝斯应主要留在伴奏轨
    assert e_inst_bass > 3.0 * max(e_voc_bass, 1e-12)


def test_resolve_separator_auto():
    try:
        import demucs  # noqa: F401
        installed = True
    except Exception:
        installed = False
    sep, _note = resolve_separator("auto")
    assert sep.name == ("demucs" if installed else "center")
