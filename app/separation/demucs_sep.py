"""Demucs v4 分离（高质量），通过官方 CLI 子进程调用，模型自动下载到项目 models/ 目录。"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .. import config
from .base import BaseSeparator, ProgressCB, SeparationResult


def pick_device() -> str:
    if config.DEVICE != "auto":   # 显式覆盖（VC_DEVICE=cpu/mps/cuda），调试/回退用
        return config.DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class DemucsSeparator(BaseSeparator):
    name = "demucs"
    label = "Demucs v4（高质量）"
    kind = "model"
    description = ("Meta 开源的混合 Transformer 音乐源分离模型，人声/伴奏质量高。"
                   "首次使用自动下载模型（约 80MB，走 hf-mirror 无关，使用 Facebook 源）。")

    def __init__(self, model: str | None = None):
        self.model = model or config.DEMUCS_MODEL

    def available(self) -> tuple[bool, str]:
        try:
            import demucs  # noqa: F401
        except Exception:
            return False, "未安装 demucs：运行 bash setup.sh --full 或 .venv/bin/pip install demucs"
        return True, ""

    def separate(self, song_wav: Path, out_dir: Path,
                 progress: ProgressCB = lambda p, m: None) -> SeparationResult:
        progress(0.05, "准备输入")
        out_dir = Path(out_dir)
        tmp = out_dir / "_tmp"
        out_root = out_dir / "_demucs_out"
        tmp.mkdir(parents=True, exist_ok=True)
        # 复制为固定文件名，避免中文/空格/重名等文件名问题
        track = tmp / "input.wav"
        shutil.copy(str(song_wav), str(track))

        env = dict(os.environ)
        env.setdefault("TORCH_HOME", str(config.MODELS_DIR / "torch"))
        cmd = [sys.executable, "-m", "demucs", "-n", self.model,
               "--two-stems=vocals", "-d", pick_device(),
               "-o", str(out_root), str(track)]
        progress(0.1, "Demucs 推理中（CPU 可能需要 1~3 分钟）")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  timeout=config.DEMUCS_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Demucs 分离超时（>{config.DEMUCS_TIMEOUT // 60} 分钟），已终止；"
                "可重试或改用快速兜底分离器")
        if proc.returncode != 0:
            raise RuntimeError(f"Demucs 分离失败: {(proc.stderr or proc.stdout or '').strip()[-600:]}")

        stem_dir = out_root / self.model / "input"
        src_vocals = stem_dir / "vocals.wav"
        src_inst = stem_dir / "no_vocals.wav"
        if not src_vocals.exists() or not src_inst.exists():
            raise RuntimeError(f"Demucs 输出缺失: {stem_dir}")

        progress(0.9, "整理输出")
        out_dir.mkdir(parents=True, exist_ok=True)
        v = out_dir / "vocals.wav"
        i = out_dir / "no_vocals.wav"
        shutil.move(str(src_vocals), str(v))
        shutil.move(str(src_inst), str(i))
        shutil.rmtree(out_root, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        progress(1.0, "分离完成")
        return SeparationResult(vocals=v, instrumental=i)
