"""音频读写与格式转换：soundfile 为主，ffmpeg 兜底处理非 wav 格式。"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import config
from .dsp import resample, to_stereo


class FFmpegNotFound(RuntimeError):
    pass


def find_ffmpeg() -> str | None:
    """优先系统 ffmpeg，其次 imageio-ffmpeg 自带的静态二进制。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_ffmpeg(args: list[str], timeout: int = 900) -> None:
    exe = find_ffmpeg()
    if not exe:
        raise FFmpegNotFound("未找到 ffmpeg：请安装（brew install ffmpeg）或 pip install imageio-ffmpeg")
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 执行失败: {(proc.stderr or '').strip()[-500:]}")


def load_wav(path) -> tuple[np.ndarray, int]:
    """读取 wav/aiff/flac 等 soundfile 支持的格式 → ([C, T] float32, sr)。"""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return data.T.astype(np.float32), int(sr)


def save_wav(path, data: np.ndarray, sr: int, subtype: str = "PCM_16") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 1:
        data = data[None, :]
    sf.write(str(path), data.T, int(sr), subtype=subtype)
    return path


def wav_duration(path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def decode_to_wav(src, dst, sr: int = config.TARGET_SR) -> Path:
    """任意支持格式 → 指定采样率的双声道 wav。soundfile 可读的直接读，否则走 ffmpeg。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext in config.SUPPORTED_SF_EXT:
        try:
            data, in_sr = load_wav(src)
            data = to_stereo(data)
            if in_sr != sr:
                data = resample(data, in_sr, sr)
            return save_wav(dst, data, sr)
        except Exception:
            pass  # soundfile 读不了（如加密/损坏），落回 ffmpeg
    run_ffmpeg(["-i", str(src), "-vn", "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])
    return dst


def export_mp3(wav_path, mp3_path, bitrate: str = "320k") -> Path | None:
    """导出 mp3；ffmpeg 不可用时返回 None（保留 wav 结果）。"""
    if find_ffmpeg() is None:
        return None
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)])
    return mp3_path
