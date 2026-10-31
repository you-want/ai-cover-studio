"""命令行工具：不启动 Web 也能生成翻唱。

用法:
  python -m app.cli engines
  python -m app.cli cover --song 歌曲文件 --sample 声音样本.wav [--engine auto] [--separator auto]
                          [--vocal-gain 0] [--pitch 0] [--out 输出目录]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .tasks import _execute


def cmd_engines(_args):
    from .separation import separator_status
    from .voice import engine_status
    print("声音引擎（按优先级）:")
    for e in engine_status():
        mark = "✓" if e["available"] else "✗"
        reason = f"  —— {e['reason']}" if e["reason"] else ""
        print(f"  [{mark}] {e['name']:<10} {e['label']}{reason}")
    print("人声分离器:")
    for s in separator_status():
        mark = "✓" if s["available"] else "✗"
        reason = f"  —— {s['reason']}" if s["reason"] else ""
        print(f"  [{mark}] {s['name']:<10} {s['label']}{reason}")


def cmd_cover(args):
    job_dir = Path(args.out).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    last_stage = [""]

    def progress(stage: str, p: float):
        line = f"\r[{stage}] {int(p * 100):3d}%"
        if stage != last_stage[0]:
            sys.stdout.write("\n")
        last_stage[0] = stage
        sys.stdout.write(line)
        sys.stdout.flush()
        if p >= 1.0:
            sys.stdout.write("\n")

    result = _execute(song_path=Path(args.song), ref_wav=Path(args.sample),
                      engine_name=args.engine, separator_name=args.separator,
                      options={"vocal_gain_db": args.vocal_gain,
                               "instrumental_gain_db": args.instrumental_gain,
                               "pitch_shift_semitones": args.pitch,
                               "tau": args.tau,
                               "auto_balance": not args.no_auto_balance,
                               "denoise_gate": not args.no_denoise_gate},
                      job_dir=job_dir, progress=progress, notes=notes)
    print()
    for n in notes:
        print("·", n)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="ai-cover-studio", description="声替 CoverMe 命令行")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("engines", help="查看引擎与分离器可用状态").set_defaults(func=cmd_engines)

    p = sub.add_parser("cover", help="生成一首翻唱")
    p.add_argument("--song", required=True, help="原歌曲文件路径")
    p.add_argument("--sample", required=True, help="用户声音样本 wav 路径（≥10 秒）")
    p.add_argument("--engine", default=config.DEFAULT_ENGINE, help="auto/seedvc/openvoice/pitchfit")
    p.add_argument("--separator", default=config.DEFAULT_SEPARATOR, help="auto/demucs/center")
    p.add_argument("--vocal-gain", type=float, default=0.0, help="人声音量 dB（-12~12）")
    p.add_argument("--instrumental-gain", type=float, default=0.0, help="伴奏音量 dB（-12~12）")
    p.add_argument("--pitch", type=float, default=0.0, help="音高微调半音数（-6~6）")
    p.add_argument("--tau", type=float, default=0.3,
                   help="OpenVoice 音色浓度 tau（0.05~0.9，越小音色越浓）")
    p.add_argument("--no-auto-balance", action="store_true",
                   help="关闭人声电平自动对齐（默认开启）")
    p.add_argument("--no-denoise-gate", action="store_true",
                   help="关闭静音段噪声门清理（默认开启）")
    p.add_argument("--out", default="output", help="输出目录")
    p.set_defaults(func=cmd_cover)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
