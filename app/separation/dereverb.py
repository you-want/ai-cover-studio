"""人声去混响/去回声（UVR-DeEcho-DeReverb 模型，经 audio-separator 调用）。

AI 翻唱社区（Jammable/kits.ai 同款流水线）的关键一步：Demucs 等分离出的
人声自带房间混响，混响"尾巴"会让声音转换模型产生拖影和毛刺；
先烘干再转换，清晰度显著提升。纯 CPU 推理，模型约 65MB。
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from .base import ProgressCB

_MODEL = "UVR-DeEcho-DeReverb.pth"


def _model_path() -> Path:
    return config.AUDIO_SEPARATOR_DIR / _MODEL


def available() -> tuple[bool, str]:
    try:
        import audio_separator  # noqa: F401
    except Exception as e:
        return False, f"audio-separator 包不可用（{type(e).__name__}: {e}）"
    if not _model_path().exists():
        return True, "去混响模型未下载（首次使用自动下载约 65MB）"
    return True, ""


def dereverb(vocals_wav: Path, out_dir: Path,
             progress: ProgressCB = lambda p, m: None) -> Path:
    """对分离出的人声做去混响，返回"干声"wav 路径。失败由调用方兜底。

    在独立子进程中运行：VR 模型 + torch 的内存用完即随子进程释放，
    不会在常驻服务进程里堆积（与 SVC 等大模型并行时避免内存爆掉）。
    """
    import subprocess
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress(0.2, "加载去混响模型")
    proc = subprocess.run(
        [sys.executable, "-m", "app.separation.dereverb",
         str(Path(vocals_wav).resolve()), str(out_dir)],
        capture_output=True, text=True, cwd=str(config.BASE_DIR))
    progress(0.85, "整理输出")
    target = out_dir / "vocals_dry.wav"
    if proc.returncode != 0 or not target.exists():
        raise RuntimeError(f"去混响失败: {(proc.stderr or proc.stdout or '').strip()[-500:]}")
    progress(1.0, "去混响完成")
    return target


if __name__ == "__main__":
    # 子进程入口：dereverb <input.wav> <out_dir>
    import sys

    _in, _out = Path(sys.argv[1]), Path(sys.argv[2])
    _out.mkdir(parents=True, exist_ok=True)
    from audio_separator.separator import Separator

    _sep = Separator(model_file_dir=str(config.AUDIO_SEPARATOR_DIR),
                     output_dir=str(_out), output_format="WAV")
    _sep.load_model(model_filename=_MODEL)
    _outputs = _sep.separate(str(_in))
    if not _outputs:
        raise SystemExit("去混响未产生输出")
    _name = Path(_outputs[0]).name
    import shutil
    _dry = next((c for c in (_out / _name, Path.cwd() / _name, Path(_name)) if c.exists()), None)
    if _dry is None:
        raise SystemExit(f"去混响输出未找到: {_outputs[0]}")
    _target = _out / "vocals_dry.wav"
    if _dry.resolve() != _target.resolve():
        shutil.move(str(_dry), str(_target))
    print(f"OK {_target}")
