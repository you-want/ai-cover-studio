"""副歌自动定位与截取：把整首歌里"最像高潮"的片段剪出来，用于快速试听翻唱。

原理：逐秒计算能量（RMS），平滑后用滑动窗口求和找能量峰值窗口；
流行编曲中副歌通常是全曲能量最高、乐器最满的段落，再对位置施加
"偏向中后段"的先验权重（避开前奏/间奏的纯器乐高能段）。

用法:
  .venv/bin/python scripts/cut_chorus.py <song.mp3> [--len 40] [--out clip.wav]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.audio.io import decode_to_wav, load_wav, save_wav  # noqa: E402


def find_chorus(data: np.ndarray, sr: int, clip_s: float = 40.0) -> tuple[float, float]:
    """返回高潮片段的 (start_s, end_s)。"""
    mono = data.mean(axis=0)
    hop = max(1, sr // 2)  # 0.5 秒一帧
    n_frames = max(1, (len(mono) - hop) // hop)
    idx = np.arange(hop)[None, :] + hop * np.arange(n_frames)[:, None]
    rms = np.sqrt(np.mean(mono[idx] ** 2, axis=1) + 1e-12)
    # 平滑（3 秒窗）
    k = max(1, 6)
    kernel = np.ones(k) / k
    energy = np.convolve(rms, kernel, mode="same")

    win = max(1, int(clip_s / 0.5))
    if n_frames <= win:  # 歌比片段还短，全曲即高潮
        return 0.0, len(mono) / sr
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    sums = csum[win:] - csum[:-win]  # 每个起点的窗口能量
    # 位置先验：偏向中后段（0.25~0.85 区间给满权重，避开纯前奏）
    pos = np.linspace(0.0, 1.0, len(sums))
    weight = 0.65 + 0.35 * np.clip(1.0 - np.abs(pos - 0.6) / 0.45, 0.0, 1.0)
    best = int(np.argmax(sums * weight))
    start_s = best * 0.5
    return start_s, start_s + clip_s


def main() -> int:
    ap = argparse.ArgumentParser(description="截取歌曲高潮片段")
    ap.add_argument("song", help="歌曲文件")
    ap.add_argument("--len", type=float, default=40.0, help="片段长度（秒）")
    ap.add_argument("--out", default="", help="输出 wav 路径（默认与歌同名 _chorus.wav）")
    args = ap.parse_args()

    src = Path(args.song)
    tmp = Path("/tmp/_chorus_decode.wav")
    decode_to_wav(src, tmp, 44100)
    data, sr = load_wav(tmp)
    start, end = find_chorus(data, sr, args.len)
    start_i, end_i = int(start * sr), min(int(end * sr), data.shape[-1])
    clip = data[..., start_i:end_i]
    out = Path(args.out) if args.out else src.with_name(src.stem + "_chorus.wav")
    save_wav(out, clip, sr)
    print(f"高潮片段: {start:.1f}s → {end:.1f}s（{end - start:.0f} 秒）→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
