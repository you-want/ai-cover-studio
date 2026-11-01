"""智能匹配测试：音域/亮度推荐逻辑（分离步骤 mock，不跑真实模型）。"""
import numpy as np
import pytest

from conftest import SR, tone
from app import analysis
from app.audio.io import save_wav


def _mk_song(tmp_path, f0=300.0):
    """合成一首"歌"：中置人声旋律 + 伴奏噪声。"""
    n = int(SR * 8)
    t = np.arange(n) / SR
    vocal = 0.5 * np.sin(2 * np.pi * f0 * t)
    rng = np.random.default_rng(7)
    side = 0.2 * rng.standard_normal(n)
    left = vocal + side
    right = vocal - side
    p = tmp_path / f"song_{f0}.mp3.wav"
    save_wav(p, np.stack([left, right]), SR)
    return p


def _cand(vid, name, f0, centroid, tmp_path):
    p = tmp_path / f"{vid}.wav"
    save_wav(p, np.stack([tone(3.0, f0)] * 2), SR)
    return {"id": vid, "name": name, "path": str(p), "duration": 10.0}


@pytest.fixture()
def fake_separation(monkeypatch):
    """把分离步骤替换为返回固定人声（300Hz），不跑 Demucs。"""
    def fake_separate(self, excerpt, workdir, progress=None):
        from app.separation.base import SeparationResult
        v = workdir / "vocals.wav"
        save_wav(v, np.stack([tone(8.0, 300.0, amp=0.4)] * 2), SR)
        i = workdir / "no_vocals.wav"
        save_wav(i, (0.1 * np.random.default_rng(1).standard_normal(int(SR * 8))).astype(np.float32), SR)
        return SeparationResult(vocals=v, instrumental=i)
    monkeypatch.setattr("app.separation.demucs_sep.DemucsSeparator.separate", fake_separate)


def test_analyze_matches_register(tmp_path, monkeypatch, fake_separation):
    """原唱 300Hz 时应推荐音域最近的候选（280Hz），而不是 500Hz 的。"""
    song = _mk_song(tmp_path, 300.0)
    cands = [
        _cand("a", "低音候选", 500.0, 800.0, tmp_path),
        _cand("b", "匹配候选", 280.0, 700.0, tmp_path),
    ]
    rec = analysis.analyze_song(song, cands)
    assert rec["voice_id"] == "b"
    assert rec["engine"] in ("seedvc", "openvoice", "pitchfit")
    assert any("音域差" in r for r in rec["reasons"])
    assert rec["register_semitones"] < 3.0


def test_analyze_skips_long_stems(tmp_path, monkeypatch, fake_separation):
    """过长的样本（整曲 stem）不参与推荐。"""
    song = _mk_song(tmp_path, 300.0)
    cands = [_cand("stem", "vocals", 300.0, 700.0, tmp_path)]
    cands[0]["duration"] = 216.0
    rec = analysis.analyze_song(song, cands)
    assert rec["voice_id"] is None


def test_semitone_distance():
    assert abs(analysis._semitone_distance(440, 220) - 12.0) < 0.01
    assert analysis._semitone_distance(300, 300) == 0.0
