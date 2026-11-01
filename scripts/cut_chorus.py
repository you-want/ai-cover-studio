"""副歌自动定位与截取：把整首歌里"最像高潮"的片段剪出来，用于快速试听翻唱。

用法:
  .venv/bin/python scripts/cut_chorus.py <song.mp3> [--len 40] [--out clip.wav]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.chorus import find_chorus  # noqa: E402
from app.audio.io import decode_to_wav, load_wav, save_wav  # noqa: E402


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
