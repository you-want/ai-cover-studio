"""分离结果缓存：同一首歌（按内容哈希）+ 同一模型只分离一次。

调参重跑（改增益/换引擎/开关混音选项）时跳过最慢的分离步骤，
直接复用缓存里的人声/伴奏轨。缓存按 LRU 容量上限自动淘汰。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from .. import config


def song_hash(song_path: Path) -> str:
    """歌曲文件内容 sha256 前 16 位（与文件名/格式无关，重命名不失效）。"""
    h = hashlib.sha256()
    with open(song_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_key(song_path: Path, model_key: str) -> str:
    return f"{song_hash(song_path)}-{model_key}"


def cache_dir(song_path: Path, model_key: str) -> Path:
    return config.CACHE_DIR / "separation" / cache_key(song_path, model_key)


def get_cached(song_path: Path, model_key: str) -> tuple[Path, Path] | None:
    """返回 (vocals, instrumental)，未命中返回 None。"""
    d = cache_dir(song_path, model_key)
    if (d / "vocals.wav").exists() and (d / "no_vocals.wav").exists():
        return d / "vocals.wav", d / "no_vocals.wav"
    return None


def store(vocals: Path, instrumental: Path, song_path: Path, model_key: str,
          meta: dict | None = None) -> None:
    """把一次成功分离的两轨存入缓存（失败静默，不影响主流程）。"""
    try:
        d = cache_dir(song_path, model_key)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vocals, d / "vocals.wav")
        shutil.copy2(instrumental, d / "no_vocals.wav")
        meta = dict(meta or {})
        meta.update({"cached_at": time.time(), "model_key": model_key})
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        _evict()
    except Exception:
        pass


def restore_into(vocals: Path, instrumental: Path, stems_dir: Path) -> tuple[Path, Path]:
    """把缓存的两轨放入任务目录（优先硬链接，跨设备退回复制）。

    任务目录保持自包含：删除任务/清理缓存互不影响（硬链接两边各持一个引用）。
    """
    stems_dir = Path(stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for src, name in ((vocals, "vocals.wav"), (instrumental, "no_vocals.wav")):
        dst = stems_dir / name
        if dst.exists():
            dst.unlink()
        try:
            os_link(str(src), str(dst))
        except OSError:
            shutil.copy2(src, dst)
        out.append(dst)
    return out[0], out[1]


def _evict() -> None:
    """缓存超过容量上限时按最近使用时间淘汰整个条目。"""
    root = config.CACHE_DIR / "separation"
    if not root.exists():
        return
    entries = [e for e in root.iterdir() if e.is_dir()]
    total = sum(f.stat().st_size for e in entries for f in e.rglob("*") if f.is_file())
    limit = int(config.CACHE_MAX_GB * (1 << 30))
    if total <= limit:
        return
    for e in sorted(entries, key=lambda p: p.stat().st_mtime):
        if total <= limit:
            break
        total -= sum(f.stat().st_size for f in e.rglob("*") if f.is_file())
        shutil.rmtree(e, ignore_errors=True)


def os_link(src: str, dst: str) -> None:
    import os
    os.link(src, dst)
