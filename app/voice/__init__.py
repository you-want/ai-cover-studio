"""声音克隆/转换引擎模块。"""
from .base import BaseVoiceEngine, VoiceProfile
from .cloud_engine import CloudEngine
from .dsp_engine import PitchFitVoiceEngine
from .openvoice_engine import OpenVoiceEngine
from .seedvc_engine import SeedVCEngine


def iter_engines():
    """按质量优先级排序：Seed-VC(歌唱转换最佳) > OpenVoice(音色克隆) > 云端API(按量付费) > 演示引擎。"""
    return [SeedVCEngine(), OpenVoiceEngine(), CloudEngine(), PitchFitVoiceEngine()]


def get_engine(name: str) -> BaseVoiceEngine:
    for e in iter_engines():
        if e.name == name:
            return e
    raise KeyError(f"未知声音引擎: {name}")


def resolve_engine(name: str) -> BaseVoiceEngine:
    """auto → 第一个可用的本地引擎；全部不可用 → 演示引擎（永远可用）。

    安全设计：auto 模式跳过 kind == "api" 的云端引擎——付费调用只允许
    用户在界面上明确选中 cloud 引擎时发生，绝不静默扣费。
    """
    if name != "auto":
        return get_engine(name)
    for e in iter_engines():
        if e.kind == "api":
            continue
        ok, _ = e.available()
        if ok:
            return e
    return PitchFitVoiceEngine()


def engine_status() -> list[dict]:
    out = []
    for e in iter_engines():
        ok, reason = e.available()
        out.append({"name": e.name, "label": e.label, "kind": e.kind,
                    "description": e.description, "available": ok, "reason": reason})
    return out
