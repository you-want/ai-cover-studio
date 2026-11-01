"""歌曲智能分析：音域 / 亮度 / 能量 → 推荐最合适的参考音色与一键设置。

给小白的"懒人模式"：拿到歌不知道选什么音色、怎么调——上传后一键分析，
自动匹配最接近原唱音域与音色亮度的参考音色，并给出推荐理由。
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from . import config
from .audio.chorus import find_chorus
from .audio.dsp import median_f0, spectral_centroid
from .audio.io import decode_to_wav, load_wav, save_wav

_PROFILE_CACHE = "voice_profiles.json"


def _semitone_distance(f0_a: float, f0_b: float) -> float:
    return abs(12.0 * math.log2(max(f0_a, 1e-6) / max(f0_b, 1e-6)))


def _excerpt(song_wav: Path, sr: int, clip_s: float) -> Path:
    """取副歌附近 clip_s 秒的分析片段。"""
    data, _ = load_wav(song_wav)
    start, end = find_chorus(data, sr, clip_s)
    clip = data[..., int(start * sr):min(int(end * sr), data.shape[-1])]
    h = hashlib.sha1(clip.tobytes()).hexdigest()[:12]
    out = config.DATA_DIR / "tmp" / f"analyze_{h}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_wav(out, clip, sr)
    return out


def _separate_excerpt(excerpt: Path, workdir: Path) -> Path:
    """分离分析片段（走分离缓存；45 秒片段约 1 分钟）。"""
    from .separation import cache as sep_cache
    from .separation.demucs_sep import DemucsSeparator

    key = "demucs-htdemucs"
    cached = sep_cache.get_cached(excerpt, key)
    if cached:
        return cached[0]
    result = DemucsSeparator().separate(excerpt, workdir, progress=lambda p, m: None)
    sep_cache.store(result.vocals, result.instrumental, excerpt, key,
                    meta={"purpose": "analysis"})
    return result.vocals


def _sample_profile(ref_path: Path) -> dict:
    """参考音色档案：中位基频 + 频谱质心（JSON 缓存，按路径+mtime 失效）。"""
    cache_file = config.DATA_DIR / _PROFILE_CACHE
    cache = {}
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    key = f"{ref_path}:{int(ref_path.stat().st_mtime)}"
    if key not in cache:
        try:
            data, sr = load_wav(ref_path)
            cache[key] = {"f0": median_f0(data, sr), "centroid": spectral_centroid(data, sr)}
        except Exception:
            cache[key] = {"f0": None, "centroid": None}
        try:
            cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return cache[key]


def analyze_song(song_path: Path, candidates: list[dict]) -> dict:
    """分析歌曲并推荐音色。candidates 为声音样本列表（store.list_samples() 结果）。"""
    song_path = Path(song_path)
    workdir = config.DATA_DIR / "tmp" / f"analyze_work_{song_path.stem[:16]}"
    song_wav = workdir / "song.wav"
    decode_to_wav(song_path, song_wav, config.TARGET_SR)
    excerpt = _excerpt(song_wav, config.TARGET_SR, 45.0)
    vocal = _separate_excerpt(excerpt, workdir)

    vdata, vsr = load_wav(vocal)
    f0_song = median_f0(vdata, vsr)
    c_song = spectral_centroid(vdata, vsr)

    scored = []
    for cand in candidates:
        if (cand.get("duration") or 0) > 90:  # 过长的是整曲 stem，不是参考音色
            continue
        prof = _sample_profile(Path(cand["path"]))
        if not prof.get("f0") or not f0_song:
            continue
        reg = _semitone_distance(prof["f0"], f0_song)
        bri = abs(math.log2(max(prof.get("centroid") or 1, 1) / max(c_song or 1, 1)))
        scored.append({"cand": cand, "centroid": prof.get("centroid"), "f0": prof["f0"],
                       "reg": reg, "bri": bri, "total": reg + 0.5 * min(bri, 3.0)})
    scored.sort(key=lambda x: x["total"])

    # 推荐引擎：本地最优可用者（云端付费引擎永不自动推荐）
    engine = "pitchfit"
    from .voice import iter_engines
    for e in iter_engines():
        if e.kind == "api":
            continue
        ok, _ = e.available()
        if ok:
            engine = e.name
            break

    if not scored:
        shutil.rmtree(workdir, ignore_errors=True)
        return {"voice_id": None, "voice_name": None, "engine": engine,
                "f0_song": round(f0_song, 1) if f0_song else None,
                "reasons": ["未能测定原唱音域（人声太少或过轻），请手动选择音色"]}

    top = scored[0]
    reasons = [f"原唱人声音域约 {f0_song:.0f} Hz，推荐「{top['cand']['name']}」"
               f"（{top['f0']:.0f} Hz），音域差仅 {top['reg']:.1f} 半音"]
    if c_song and top["centroid"]:
        ratio = top["centroid"] / c_song
        tone = "更明亮" if ratio > 1.1 else ("更柔和" if ratio < 0.9 else "亮度接近")
        reasons.append(f"音色亮度与原唱{tone}，整体听感自然")
    reasons.append("引擎已按当前环境自动选优，混音参数保持智能默认"
                   "（电平对齐/静音段清理/人声总线全开）")

    shutil.rmtree(workdir, ignore_errors=True)
    return {
        "voice_id": top["cand"]["id"],
        "voice_name": top["cand"]["name"],
        "engine": engine,
        "f0_song": round(f0_song, 1),
        "f0_voice": round(top["f0"], 1),
        "register_semitones": round(top["reg"], 1),
        "reasons": reasons,
    }
