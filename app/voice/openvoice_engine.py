"""OpenVoice v2 音色克隆引擎（零样本，本地推理）。

依赖（可选，setup.sh --full 自动安装）：
  - openvoice 包（GitHub myshell-ai/OpenVoice）
  - checkpoint_v2/converter 权重（scripts/download_openvoice.py 从 hf-mirror 下载）

工作方式：从用户录音提取音色向量（Speaker Embedding），保留旋律与节奏，
把原唱人声的音色替换为用户音色 —— 真正意义上的"音色克隆翻唱"。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from .. import config
from .base import BaseVoiceEngine, ProgressCB, VoiceProfile

_LOCK = threading.Lock()
_STATE: dict = {"converter": None}


def _pick_device() -> str:
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


def _converter_dir() -> Path:
    return config.OPENVOICE_DIR / "OpenVoiceV2" / "converter"


class OpenVoiceEngine(BaseVoiceEngine):
    name = "openvoice"
    label = "OpenVoice v2 · 音色克隆"
    kind = "zero-shot"
    description = ("零样本音色迁移：从你的录音提取音色向量，保留旋律与节奏，"
                   "把原唱人声转换成你的音色。CPU 单曲约需几十秒。"
                   "需要安装 openvoice 包与模型权重（bash setup.sh --full）。")

    def available(self) -> tuple[bool, str]:
        try:
            from openvoice.api import ToneColorConverter  # noqa: F401
        except Exception as e:
            return False, f"openvoice 包不可用（{type(e).__name__}: {e}），可运行 bash setup.sh --full 安装"
        d = _converter_dir()
        if not (d / "checkpoint.pth").exists() or not (d / "config.json").exists():
            return False, f"缺少模型权重（{d}），可运行 bash setup.sh --full 下载"
        return True, ""

    def _get_converter(self):
        with _LOCK:
            if _STATE["converter"] is None:
                from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

                class _NoWatermarkConverter(ToneColorConverter):
                    """绕过上游两个问题：
                    1. enable_watermark kwarg 未被消费即传给基类 → TypeError；
                    2. wavmark 水印模型需联网下载且经镜像不稳定。
                    本地个人工具不需要水印，直接置空（add_watermark 会自动直通）。
                    """

                    def __init__(self, config_path, device):
                        OpenVoiceBaseClass.__init__(self, config_path, device=device)
                        self.watermark_model = None
                        self.version = getattr(self.hps, "_version_", "v1")

                d = _converter_dir()
                conv = _NoWatermarkConverter(str(d / "config.json"), device=_pick_device())
                conv.load_ckpt(str(d / "checkpoint.pth"))
                _STATE["converter"] = conv
            return _STATE["converter"]

    def _extract_se(self, wav_path: Path, converter, se_save_path: str | None = None):
        """直接对整段音频提取音色向量（Speaker Embedding）。

        不走官方 se_extractor：其 vad=False 分支也会调用 whisper 做分段，
        需联网下载大模型；对 10~30 秒干净样本整段提取效果等价且快得多。
        """
        return converter.extract_se(str(wav_path), se_save_path=se_save_path)

    def build_profile(self, ref_wav: Path, cache_dir: Path,
                      progress: ProgressCB = lambda p, m: None) -> VoiceProfile:
        progress(0.15, "加载音色模型")
        profile = VoiceProfile(sample_id=Path(ref_wav).parent.name, ref_wav=Path(ref_wav))
        converter = self._get_converter()
        progress(0.45, "提取你的音色向量")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        se_path = cache_dir / "target_se.pt"
        se = self._extract_se(Path(ref_wav), converter, se_save_path=str(se_path))
        profile.extra["se_path"] = str(se_path)
        if se is None:
            # 兜底：某些版本 extract_se 不返回值时从保存文件读取
            import torch
            se = torch.load(se_path, map_location="cpu")
        profile.extra["tau"] = 0.3
        progress(1.0, "声音分析完成")
        return profile

    def convert(self, source_vocal: Path, profile: VoiceProfile, out_wav: Path,
                pitch_shift_semitones: float = 0.0,
                progress: ProgressCB = lambda p, m: None) -> dict:
        import torch
        progress(0.15, "加载音色模型")
        converter = self._get_converter()
        progress(0.35, "分析源人声音色")
        src_se = self._extract_se(Path(source_vocal), converter)
        tgt_se = torch.load(profile.extra["se_path"], map_location="cpu")
        if hasattr(converter, "device"):
            tgt_se = tgt_se.to(converter.device)

        progress(0.55, "音色转换（OpenVoice）")
        # 临时文件带唯一后缀，避免并发任务（VC_MAX_CONCURRENT_JOBS>1 或多开 CLI）
        # 读写同一路径互相覆盖
        out_raw = Path(out_wav).parent / f"_openvoice_raw_{os.getpid()}_{threading.get_ident()}.wav"
        tau = float(profile.extra.get("tau", 0.3))
        converter.convert(audio_src_path=str(source_vocal), src_se=src_se, tgt_se=tgt_se,
                          output_path=str(out_raw), tau=tau)

        progress(0.85, "后处理")
        from ..audio.dsp import (_rbj_biquad, biquad, normalize_loudness,
                                 pitch_shift, resample, to_stereo)
        from ..audio.io import load_wav, save_wav
        data, sr = load_wav(out_raw)
        data = to_stereo(data)
        if sr != config.TARGET_SR:
            data = resample(data, sr, config.TARGET_SR)
        if abs(pitch_shift_semitones) > 1e-3:
            data = pitch_shift(data, config.TARGET_SR, pitch_shift_semitones)
        # 成品"闷"的两点修正（相对纯 peak_normalize）：
        # 1) RMS 归一到 -18dB：峰值归一会按全曲最响瞬间对齐，整体响度偏低，
        #    人声混进伴奏后听感发闷发平；normalize_loudness 内部仍带峰值上限 0.9
        data = normalize_loudness(data, target_db=-18.0, max_peak=0.9)
        # 2) 高频搁架 +3dB：OpenVoice 输出 2.5~5kHz 亮度带通常比原人声暗 7~9dB，
        #    补一点高频给出口度和清晰度。低频不再额外补偿（实测 OpenVoice 输出
        #    低频充足，旧版低搁架实现有系数错误，已移除）
        sr_out = config.TARGET_SR
        b, a = _rbj_biquad("highshelf", sr_out, 2500.0, gain_db=3.0)
        data = biquad(data, b, a)
        save_wav(out_wav, data, sr_out)
        Path(out_raw).unlink(missing_ok=True)
        progress(1.0, "转换完成")
        return {"engine": f"已使用 OpenVoice v2 音色克隆（tau={tau:.2f}）"}
