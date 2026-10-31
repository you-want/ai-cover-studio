"""声音引擎基类与用户声音档案。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# 进度回调：progress(0.0~1.0 阶段内进度, 消息)
ProgressCB = Callable[[float, str], None]


@dataclass
class VoiceProfile:
    """从用户声音样本中提取的档案：通用声学特征 + 引擎私有缓存（如音色向量）。"""
    sample_id: str = ""
    ref_wav: Path | None = None
    f0_median: float | None = None    # 中位基频 Hz（音域高低）
    centroid: float | None = None     # 频谱质心 Hz（音色亮暗）
    extra: dict[str, Any] = field(default_factory=dict)


class BaseVoiceEngine:
    """声音引擎接口。

    convert() 把"原唱人声"转换成"用户音色"的歌声，返回一个 notes 字典
    描述本次转换做了什么（写入任务日志，展示给用户）。
    """

    name: str = "base"
    label: str = "基类"
    kind: str = "base"          # zero-shot / training / demo
    description: str = ""

    def available(self) -> tuple[bool, str]:
        return True, ""

    def build_profile(self, ref_wav: Path, cache_dir: Path,
                      progress: ProgressCB = lambda p, m: None) -> VoiceProfile:
        """默认实现：提取基础声学特征（基频、频谱质心）。"""
        from ..audio.dsp import median_f0, spectral_centroid
        from ..audio.io import load_wav
        progress(0.3, "分析参考录音")
        data, sr = load_wav(ref_wav)
        profile = VoiceProfile(
            sample_id=Path(ref_wav).parent.name,
            ref_wav=Path(ref_wav),
            f0_median=median_f0(data, sr),
            centroid=spectral_centroid(data, sr),
        )
        progress(1.0, "声音分析完成")
        return profile

    def convert(self, source_vocal: Path, profile: VoiceProfile, out_wav: Path,
                pitch_shift_semitones: float = 0.0,
                progress: ProgressCB = lambda p, m: None) -> dict:
        raise NotImplementedError
