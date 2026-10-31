"""Seed-VC 声音转换引擎（实验性，零样本）。

Seed-VC 对"把原唱人声换成别人音色"的迁移效果是目前开源方案里最好的之一。
此封装为实验性：依赖 seed-vc 仓库检出 + Python 包（dac/transformers 等），
通过其官方 inference 脚本以子进程方式调用；未安装时引擎显示不可用，不影响其他功能。

目录约定：
  models/seedvc/repo/        seed-vc 仓库（含 inference.py）
  models/seedvc/repo/checkpoints/   权重（首次运行自动下载约 2GB，走 HF 镜像）
"""
from __future__ import annotations

import importlib.util
import os
import re
import select
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .. import config
from ..audio.dsp import peak_normalize, pitch_shift, resample, to_stereo
from ..audio.io import load_wav, save_wav
from .base import BaseVoiceEngine, ProgressCB, VoiceProfile

# inference.py 推理路径必需的包（gradio/resemblyzer 只服务于它自带的 Web UI）
_REQUIRED_PACKAGES = ("dac", "transformers")

# tqdm 进度条形如 " 20%|██ | 6/30 [00:41<02:46"；SVC 采样循环还会输出 "SAMPLE 6/30"
_TQDM_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
_SAMPLE_RE = re.compile(r"SAMPLE\s+(\d+)\s*/\s*(\d+)")


def _tqdm_fraction(tail: str) -> float | None:
    # 取缓冲尾部"最后一个"匹配（缓冲里残留着更早的旧进度，取第一个会卡死不动）
    ms = _SAMPLE_RE.findall(tail) or _TQDM_RE.findall(tail)
    if not ms:
        return None
    n, total = int(ms[-1][0]), int(ms[-1][1])
    if total <= 0:
        return None
    return max(0.0, min(1.0, n / total))


def _torch_env() -> dict:
    """子进程环境：限制 torch 线程数，避免与其他推理进程争抢 CPU。"""
    env = dict(os.environ)
    n = str(min(6, os.cpu_count() or 4))
    env.setdefault("OMP_NUM_THREADS", n)
    env.setdefault("MKL_NUM_THREADS", n)
    return env


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class SeedVCEngine(BaseVoiceEngine):
    name = "seedvc"
    label = "Seed-VC · 声音转换（实验）"
    kind = "zero-shot"
    description = ("零样本声音转换：迁移音色的同时最大程度保留演唱细节，对歌曲效果最佳。"
                   "实验性支持，需要 seed-vc 仓库与依赖包（bash setup.sh --full 尝试安装）；"
                   "首次使用自动下载约 2GB 权重，CPU 推理较慢（约数分钟/曲）。")

    def _repo_dir(self) -> Path:
        return config.SEEDVC_DIR / "repo"

    def _ready_flag(self) -> Path:
        return config.SEEDVC_DIR / "ready.flag"

    def _checkpoint_dir(self) -> Path:
        # Seed-VC 官方 inference.py 在 <repo>/checkpoints/ 下自动下载/读取权重
        return self._repo_dir() / "checkpoints"

    def _svc_ready(self) -> bool:
        """SVC 歌唱模式权重（f0-44k DiT + 配置）是否已就位。"""
        ckpt = self._checkpoint_dir()
        return (any(ckpt.glob("models--Plachta--Seed-VC/snapshots/*/DiT_seed_v2_uvit_whisper_base_f0_44k*.pth"))
                and any(ckpt.glob("models--Plachta--Seed-VC/snapshots/*/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")))

    def available(self) -> tuple[bool, str]:
        if not self._repo_dir().exists():
            return False, "未安装 seed-vc 仓库（bash setup.sh --full 尝试安装）"
        missing = [p for p in _REQUIRED_PACKAGES if importlib.util.find_spec(p) is None]
        if missing:
            return False, (f"缺少依赖包 {'/'.join(missing)}："
                           "运行 .venv/bin/pip install descript-audio-codec transformers 安装")
        ckpt = self._checkpoint_dir()
        # 自动下载模式：只要仓库与依赖在，首次运行 inference.py 会自动拉取权重，
        # 这里不强制要求权重已就位，只提示尚未预热
        if not ckpt.exists() and not self._ready_flag().exists():
            return True, "模型权重未下载（首次使用自动下载约 2GB）"
        return True, ""

    def build_profile(self, ref_wav: Path, cache_dir: Path,
                      progress: ProgressCB = lambda p, m: None) -> VoiceProfile:
        progress(0.5, "准备参考音频")
        # 子进程以 repo 为工作目录，参考音频必须用绝对路径
        ref_wav = Path(ref_wav).resolve()
        profile = VoiceProfile(sample_id=ref_wav.parent.name, ref_wav=ref_wav)
        profile.extra["ref_wav"] = str(ref_wav)
        progress(1.0, "声音分析完成")
        return profile

    def convert(self, source_vocal: Path, profile: VoiceProfile, out_wav: Path,
                pitch_shift_semitones: float = 0.0,
                progress: ProgressCB = lambda p, m: None) -> dict:
        progress(0.1, "启动 Seed-VC 推理（首次运行需下载权重）")
        repo = self._repo_dir()
        device = _pick_device()
        # 子进程 cwd 是 repo 目录：源音频/输出目录都用绝对路径，避免相对路径失效
        source_vocal = Path(source_vocal).resolve()
        out_wav = Path(out_wav).resolve()
        # SVC 歌唱模式：F0 条件化保留旋律 + 自动画到目标音域，比 VC 模式更适合唱歌
        use_svc = config.SEEDVC_SVC and self._svc_ready()
        svc_semi_shift: int | None = None
        # 每次运行用独立临时目录，避免并发任务互相踩踏
        out_dir = Path(out_wav).parent / f"_seedvc_out_{uuid.uuid4().hex[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(repo / "inference.py"),
               "--source", str(source_vocal),
               "--target", str(profile.extra["ref_wav"]),
               "--output", str(out_dir),
               "--diffusion-steps", str(config.SEEDVC_DIFFUSION_STEPS)]
        if use_svc:
            cmd += ["--f0-condition", "true", "--auto-f0-adjust", "true"]
            # SVC 模式下音高微调交给原生的 semi-tone-shift（变调不变速，质量优于后处理 WSOLA）
            if abs(pitch_shift_semitones) > 0.4:
                svc_semi_shift = int(round(pitch_shift_semitones))
                cmd += ["--semi-tone-shift", str(svc_semi_shift)]
                pitch_shift_semitones = 0.0  # 已由 SVC 原生处理
        # CPU/MPS 上 fp16 会崩或精度异常，只在 CUDA 上启用
        if device != "cuda":
            cmd += ["--fp16", "false"]
        # inference.py / hf_utils.py 全部使用相对路径（./checkpoints），
        # 必须以仓库目录为工作目录，权重才会落在 repo/checkpoints。
        # 用 Popen + select 流式读取 tqdm 输出，把扩散步数映射为实时进度；
        # 双看门狗：30 分钟无输出视为卡死，4 小时硬上限（长歌 30 步可能 2~3 小时）
        proc = subprocess.Popen(cmd, cwd=str(repo),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=_torch_env())
        fd = proc.stdout.fileno()  # type: ignore[union-attr]
        tail = b""
        last_frac = -1.0
        last_output = time.time()
        hard_deadline = time.time() + 4 * 3600
        while True:
            if time.time() > hard_deadline:
                proc.kill()
                raise RuntimeError("Seed-VC 推理超过 4 小时硬上限，已终止")
            ready, _, _ = select.select([fd], [], [], 60)
            if not ready:
                if time.time() - last_output > 1800:
                    proc.kill()
                    raise RuntimeError("Seed-VC 推理 30 分钟无输出，已按卡死终止")
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            last_output = time.time()
            tail = (tail + chunk)[-8192:]
            frac = _tqdm_fraction(tail.decode("utf-8", "ignore"))
            if frac is not None and frac > last_frac:
                last_frac = frac
                progress(0.1 + 0.72 * frac, f"扩散推理 {int(frac * 100)}%")
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Seed-VC 推理失败: {tail.decode('utf-8', 'ignore').strip()[-600:]}")
        produced = sorted(out_dir.glob("*.wav"))
        if not produced:
            raise RuntimeError(f"Seed-VC 未产生输出: {out_dir}")

        progress(0.85, "后处理")
        data, sr = load_wav(produced[0])
        data = to_stereo(data)
        if sr != config.TARGET_SR:
            data = resample(data, sr, config.TARGET_SR)
        if abs(pitch_shift_semitones) > 1e-3:
            data = pitch_shift(data, config.TARGET_SR, pitch_shift_semitones)
        data = peak_normalize(data, 0.9)
        save_wav(out_wav, data, config.TARGET_SR)
        for p in produced:
            p.unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        progress(1.0, "转换完成")
        mode = ("歌唱模式 SVC（F0 条件化，保留旋律与唱腔）" if use_svc else "声音转换 VC 模式")
        extra = f"，音域适配 {svc_semi_shift:+d} 半音" if use_svc and svc_semi_shift is not None else ""
        return {"engine": (f"已使用 Seed-VC {mode}"
                           f"（diffusion-steps={config.SEEDVC_DIFFUSION_STEPS}{extra}）")}
