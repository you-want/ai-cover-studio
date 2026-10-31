"""一键端到端演示：生成演示素材 → 走完整翻唱流水线 → 打印结果路径。

用法:
  .venv/bin/python scripts/demo.py                # 自动选择可用引擎/分离器
  .venv/bin/python scripts/demo.py --say          # 用 macOS TTS 生成真实感声音样本
  .venv/bin/python scripts/demo.py --engine pitchfit --separator center
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.tasks import _execute  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--separator", default="auto")
    ap.add_argument("--say", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "demo" / "output"))
    args = ap.parse_args()

    demo = ROOT / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    song = demo / "demo_song.wav"
    ref = demo / "demo_ref.wav"
    if not song.exists() or not ref.exists() or args.say:
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "make_demo_song.py"), "--out", str(demo)]
        if args.say:
            cmd.append("--say")
        subprocess.run(cmd, check=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    last = [""]

    def progress(stage, p):
        if stage != last[0]:
            sys.stdout.write("\n")
            last[0] = stage
        sys.stdout.write(f"\r[{stage}] {int(p * 100):3d}%")
        sys.stdout.flush()
        if p >= 1.0:
            sys.stdout.write("\n")

    print(f"引擎: {args.engine}  分离器: {args.separator}")
    result = _execute(song_path=song, ref_wav=ref, engine_name=args.engine,
                      separator_name=args.separator, options={},
                      job_dir=out_dir, progress=progress, notes=notes)
    print()
    for n in notes:
        print("·", n)
    print("\n完成！产出文件：")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
