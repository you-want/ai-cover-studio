"""生成演示素材：
- demo_song.wav  合成一首"歌曲"（旋律"人声" + 和弦垫 + 鼓点，中置人声/宽展伴奏）
- demo_ref.wav   模拟"用户声音样本"（低音域音色）

加 --say 可用 macOS 自带 TTS（say 命令）生成更真实的干声样本。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.audio.io import decode_to_wav  # noqa: E402

SR = 44100
rng = np.random.default_rng(7)


def tone(dur, freq, harmonics=(1.0, 0.5, 0.25), amp=0.5, vibrato_cents=30.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = freq * (2 ** ((vibrato_cents / 1200.0) * np.sin(2 * np.pi * 5.5 * t)))
    phase = 2 * np.pi * np.cumsum(f) / SR
    y = np.zeros(n)
    for i, a in enumerate(harmonics, start=1):
        y += a * np.sin(i * phase)
    y /= max(1e-9, np.abs(y).max())
    return y * amp


def midi(n_):
    return 440.0 * 2 ** ((n_ - 69) / 12)


def build_song(out_path: Path, note_sec=0.5):
    melody = [69, 71, 73, 74, 76, 74, 73, 71] * 4           # A4 附近的简单旋律
    vocal = np.concatenate([tone(note_sec * 0.92, midi(m)) for m in melody])
    n = len(vocal)
    # 和弦垫（伴奏，中置）
    pad = np.zeros(n)
    roots = [48, 53, 55, 53] * 8                             # C3 F3 G3 F3
    for i, root in enumerate(roots):
        s = int(i * note_sec * 2 * SR)
        e = min(int((i + 1) * note_sec * 2 * SR), n)
        if s >= n:
            break
        for iv in (0, 4, 7):
            pad[s:e] += 0.22 * tone((e - s) / SR, midi(root + iv), harmonics=(1.0, 0.35))
    # 鼓点（底鼓/军鼓，中置）
    drums = np.zeros(n)
    L = SR // 10
    for k in range(int(n / (SR * 0.5)) + 1):
        c = int(k * 0.5 * SR)
        if c + L >= n:
            break
        env = np.exp(-np.linspace(0, 7, L))
        f0 = 80.0 if k % 2 == 0 else 190.0
        drums[c:c + L] += (0.5 if k % 2 == 0 else 0.3) * env * np.sin(2 * np.pi * f0 * np.arange(L) / SR)
    # 宽展"乐器"噪声（边带）
    side = (0.10 * rng.standard_normal(n))

    center = 0.85 * vocal + 0.9 * pad + 0.6 * drums
    left = center + side
    right = center - side
    peak = max(np.abs(left).max(), np.abs(right).max())
    out = np.stack([left, right], axis=1) / peak * 0.9
    sf.write(str(out_path), out, SR, subtype="PCM_16")
    print(f"✓ 演示歌曲: {out_path}（{n / SR:.1f} 秒）")


def build_ref(out_path: Path, use_say: bool):
    if use_say:
        text = ("大家好，欢迎来到 AI 翻唱工坊。这是一段用于克隆音色的录音样本。"
                "希望我的声音可以唱出我喜欢的歌，一起来试试吧！")
        aiff = out_path.with_suffix(".aiff")
        try:
            import subprocess
            subprocess.run(["say", "-o", str(aiff), text], check=True, timeout=120)
            decode_to_wav(aiff, out_path, SR)
            aiff.unlink(missing_ok=True)
            print(f"✓ 声音样本(macOS TTS): {out_path}")
            return
        except Exception as e:
            print(f"say 不可用（{e}），回退为合成音色")
    y = np.concatenate([tone(1.2, midi(m), harmonics=(1.0, 0.45, 0.2), amp=0.55, vibrato_cents=15)
                        for m in [45, 48, 50, 52, 50, 48, 45, 43, 45, 48, 50, 45]])  # 低音域
    sf.write(str(out_path), y, SR, subtype="PCM_16")
    print(f"✓ 声音样本(合成): {out_path}（{len(y) / SR:.1f} 秒）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "demo"))
    ap.add_argument("--say", action="store_true", help="用 macOS say 生成真实感干声样本")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    build_song(out / "demo_song.wav")
    build_ref(out / "demo_ref.wav", args.say)
