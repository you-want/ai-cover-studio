"""任务层测试：分离缓存、回退日志去噪、取消语义。"""
import numpy as np
import pytest

from conftest import SR, tone
from app import config
from app.separation import cache as sep_cache
from app.tasks import JobCancelled, _execute, _short_reason


def _write_wav(path, amp=0.5):
    from app.audio.io import save_wav
    save_wav(path, np.stack([tone(1.0, 300.0, amp=amp)] * 2), SR)
    return path


def test_song_hash_by_content(tmp_path):
    a = _write_wav(tmp_path / "a.wav")
    b = tmp_path / "b.wav"
    b.write_bytes(a.read_bytes())          # 同内容不同文件名
    c = _write_wav(tmp_path / "c.wav", amp=0.9)
    assert sep_cache.song_hash(a) == sep_cache.song_hash(b)
    assert sep_cache.song_hash(a) != sep_cache.song_hash(c)


def test_separation_cache_roundtrip(tmp_path):
    song = _write_wav(tmp_path / "song.wav")
    stems = tmp_path / "stems1"
    stems.mkdir()
    v, i = _write_wav(stems / "vocals.wav"), _write_wav(stems / "no_vocals.wav")

    assert sep_cache.get_cached(song, "demucs-htdemucs") is None
    sep_cache.store(v, i, song, "demucs-htdemucs", meta={"separator": "demucs"})
    cached = sep_cache.get_cached(song, "demucs-htdemucs")
    assert cached is not None
    assert (cached[0].name, cached[1].name) == ("vocals.wav", "no_vocals.wav")

    # 恢复到另一个任务目录：内容一致且自包含
    stems2 = tmp_path / "stems2"
    v2, i2 = sep_cache.restore_into(cached[0], cached[1], stems2)
    assert v2.read_bytes() == v.read_bytes()
    assert i2.read_bytes() == i.read_bytes()

    # 不同模型键不互相命中
    assert sep_cache.get_cached(song, "demucs-htdemucs_ft") is None


def test_cache_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_MAX_GB", 0.0)  # 任何写入都触发淘汰
    song = _write_wav(tmp_path / "s.wav")
    stems = tmp_path / "st"
    stems.mkdir()
    sep_cache.store(_write_wav(stems / "v.wav"), _write_wav(stems / "i.wav"),
                    song, "demucs-x")
    # 超限后条目被淘汰（保留最近一个也放不下 → 全清）
    assert sep_cache.get_cached(song, "demucs-x") is None


def test_short_reason_one_line():
    e = RuntimeError("Seed-VC 推理失败:\n  File \"x.py\", line 1\n    boom()\nModuleNotFoundError: dac")
    s = _short_reason(e)
    assert "\n" not in s and "  " not in s
    assert s.endswith("ModuleNotFoundError: dac")


def test_short_reason_truncates():
    s = _short_reason(RuntimeError("x" * 1000))
    assert len(s) <= 241 and s.endswith("…")


def test_execute_cancelled_at_checkpoint(tmp_path):
    """should_cancel=True 时任务在预处理阶段即抛 JobCancelled，不产生任何结果。"""
    song = _write_wav(tmp_path / "song.wav")
    sample = _write_wav(tmp_path / "ref.wav")
    notes: list[str] = []

    def progress(stage, p):
        pass

    with pytest.raises(JobCancelled):
        _execute(song_path=song, ref_wav=sample, engine_name="pitchfit",
                 separator_name="center", options={}, job_dir=tmp_path / "job",
                 progress=progress, notes=notes, should_cancel=lambda: True)
    assert not (tmp_path / "job" / "cover.wav").exists()
