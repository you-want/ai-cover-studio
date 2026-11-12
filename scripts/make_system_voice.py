"""生成系统音色：对现有参考音色做"老年化"DSP，注册进系统音色库（kind='builtin'）。

做法：以仓库内干净的参考人声为底，套用老年嗓音特征（全部用 app.audio.dsp 现成原语）：
  1) 降基频（pitch_shift，老年声更低）
  2) 嘎裂声（creaky voice）：低频带 6.5Hz 慢速调幅，近似老年喉部粗糙摩擦
  3) 气声（breathiness）：叠加轻微带通噪声
  4) 颤抖（tremor）：5Hz 慢速幅度波动，老年声带控制力下降
  5) 高频衰减：老年声高频能量变薄（lowpass）
  6) 响度归一 + 软限幅 + 淡入淡出
随机种子固定，结果可复现；参数不满意改 preset 数值重跑即可（DSP 秒级完成）。

用法:
  python -m scripts.make_system_voice                      # 生成默认 preset（老头 + 老太太）
  python -m scripts.make_system_voice --preset old-man     # 只生成指定 preset
  python -m scripts.make_system_voice --source data/samples/xxx/ref.wav \
      --name "系统音色·自定义" --semitones -4 --creaky 0.4 --breath 0.2
"""
from __future__ import annotations

import argparse
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.audio.dsp import (  # noqa: E402
    fade, lowpass, median_f0, normalize_loudness, pitch_shift, soft_limit, to_mono, to_stereo,
)
from app.audio.io import load_wav, save_wav, wav_duration  # noqa: E402

# 默认 preset：以仓库内现成参考音色为底
PRESETS = {
    # 源：参考音色·清亮男声（f0≈230Hz）→ 降 6 个半音（≈150Hz）+ 温和嘎裂 = 农民老头的苍老感
    # v2（实测复盘）：v1 的 creaky=0.55（6.5Hz 调幅）+ tremor=0.05（5Hz 颤抖）节奏落在
    # "抽泣"频段，SVC 把它搬进歌声后听感是明显哭腔；rolloff=5000 则导致转换输出发闷
    # （>5kHz 能量 7.7% vs 原唱 25.4%）。v2 放缓调幅、放宽高频。
    "old-man": dict(
        source="613b7e4620ca4623", name="系统音色·农民老头",
        semitones=-6.0, creaky=0.30, breath=0.12, tremor=0.02, rolloff=6500, seed=7,
    ),
    # 源：参考音色·温暖女中音（f0≈196Hz）→ 降 3 个半音（≈165Hz）+ 气声偏重 = 老太太的沧桑感
    # v2：与 old-man 同理放缓调幅、放宽高频
    "old-lady": dict(
        source="47062276128303a0", name="系统音色·老太太",
        semitones=-3.0, creaky=0.22, breath=0.16, tremor=0.02, rolloff=6800, seed=11,
    ),
}


def age_vox(x: np.ndarray, sr: int, *, semitones: float = 0.0, creaky: float = 0.0,
            breath: float = 0.0, tremor: float = 0.0, rolloff: float | None = None,
            seed: int = 0) -> np.ndarray:
    """老年化 DSP 链。x 为 [C, T] float32，输出同形状。"""
    rng = np.random.default_rng(seed)
    y = to_stereo(x)
    if abs(semitones) > 1e-3:
        y = to_stereo(pitch_shift(y, sr, semitones))
    t = np.arange(y.shape[-1], dtype=np.float64) / sr

    if creaky > 0:
        # 嘎裂声：低频带 6.5Hz 调幅 → 在基频两侧产生 6.5Hz 边带，听感为"喉部摩擦/气颤"
        low = lowpass(y, sr, 400.0)
        am = (0.5 + 0.5 * np.sin(2 * np.pi * 6.5 * t + rng.uniform(0, 2 * np.pi)))[None, :]
        y = y - low + low * (1.0 - creaky + creaky * am)

    if breath > 0:
        # 气声：白噪声限到 3.5kHz 以内，幅度按 voice/noise RMS 比标定（breath=0.2 约 -14dB）
        noise = lowpass(rng.standard_normal(y.shape).astype(np.float32), sr, 3500.0)
        rms_sig = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-9
        rms_n = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))) + 1e-9
        y = y + noise * (breath * rms_sig / rms_n)

    if tremor > 0:
        amp = (1.0 + tremor * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 2 * np.pi)))[None, :]
        y = y * amp

    if rolloff:
        y = lowpass(y, sr, float(rolloff))

    y = normalize_loudness(y, target_db=-14.0)
    y = soft_limit(y, sr, ceiling=0.95)
    y = fade(y, sr, in_s=0.02, out_s=0.05)
    return y.astype(np.float32)


def register(dst_wav: Path, name: str, register_db: bool = True) -> str:
    """写入系统音色库目录并登记（kind='builtin'），返回样本 id。"""
    sid = secrets.token_hex(8)
    ref = config.SAMPLES_DIR / sid / "ref.wav"
    ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(dst_wav), str(ref))
    dur = wav_duration(ref)
    if dur < 8:
        shutil.rmtree(config.SAMPLES_DIR / sid, ignore_errors=True)
        raise SystemExit(f"样本太短（{dur:.1f}s），需 10~30 秒参考人声")
    if register_db:
        from app.store import Store
        Store(config.DB_PATH).add_sample(sid, name, ref, dur, kind="builtin")
    print(f"样本 id: {sid}  →  {ref.relative_to(ROOT)}（{dur:.1f}s）")
    return sid


def build_preset(preset: str, args) -> None:
    p = dict(PRESETS[preset])
    p.update({k: v for k, v in dict(
        semitones=args.semitones, creaky=args.creaky, breath=args.breath,
        tremor=args.tremor, rolloff=args.rolloff, seed=args.seed,
        source=args.source, name=args.name).items() if v is not None})
    if p.get("source"):
        src_path = Path(p["source"]) if not str(p["source"]).startswith(
            "data/samples/") else ROOT / str(p["source"])
        src = Path(args.source) if args.source else config.SAMPLES_DIR / p["source"] / "ref.wav"
        if not src.exists():
            raise SystemExit(f"源样本不存在: {src}")
        x, sr = load_wav(src)
    else:
        x, sr = load_wav(config.SAMPLES_DIR / "9dc3d553a92e2d38" / "ref.wav")  # 兜底：烟嗓试验样本
    if sr != config.TARGET_SR:
        from app.audio.dsp import resample
        x = resample(x, sr, config.TARGET_SR)
        sr = config.TARGET_SR

    out = age_vox(x, sr, semitones=p["semitones"], creaky=p["creaky"], breath=p["breath"],
                  tremor=p["tremor"], rolloff=p["rolloff"], seed=p["seed"])
    f0_before = median_f0(x, sr)
    f0_after = median_f0(out, sr)
    preview = config.DATA_DIR / "tmp" / f"system_voice_{preset}.wav"
    save_wav(preview, out, sr)
    print(f"[{preset}] {p['name']}: 基频 {f0_before and round(f0_before, 1) or '?'}Hz → "
          f"{f0_after and round(f0_after, 1) or '?'}Hz, "
          f"参数 semitones={p['semitones']} creaky={p['creaky']} breath={p['breath']} "
          f"tremor={p['tremor']} rolloff={p['rolloff']}")
    print(f"试听: {preview}")
    if not args.no_register:
        register(preview, p["name"])


def main():
    ap = argparse.ArgumentParser(description="生成老年化系统音色并注册进系统音色库")
    ap.add_argument("--preset", default="both", choices=["old-man", "old-lady", "both"])
    ap.add_argument("--source", help="覆盖参考音色源（data/samples/<id>/ref.wav 或绝对路径）")
    ap.add_argument("--name", help="覆盖显示名")
    ap.add_argument("--semitones", type=float, help="降/升调半音数（默认 preset 值）")
    ap.add_argument("--creaky", type=float, help="嘎裂程度 0~1")
    ap.add_argument("--breath", type=float, help="气声程度 0~1")
    ap.add_argument("--tremor", type=float, help="颤抖深度 0~0.2")
    ap.add_argument("--rolloff", type=float, help="高频截止 Hz")
    ap.add_argument("--seed", type=int, help="随机种子")
    ap.add_argument("--no-register", action="store_true", help="只出试听文件，不入库")
    args = ap.parse_args()

    if args.preset == "both":
        build_preset("old-man", args)
        build_preset("old-lady", args)
    else:
        build_preset(args.preset, args)


if __name__ == "__main__":
    main()
