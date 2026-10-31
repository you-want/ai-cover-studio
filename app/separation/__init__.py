"""人声/伴奏分离模块。"""
from .base import BaseSeparator, SeparationResult
from .center_extract import CenterExtractSeparator
from .demucs_sep import DemucsSeparator


def iter_separators():
    """按质量优先级排序。"""
    return [DemucsSeparator(), CenterExtractSeparator()]


def get_separator(name: str) -> BaseSeparator:
    for s in iter_separators():
        if s.name == name:
            return s
    raise KeyError(f"未知分离器: {name}")


def resolve_separator(name: str) -> tuple[BaseSeparator, str | None]:
    """auto → 按优先级选第一个可用；全部不可用 → 中心声道兜底。

    返回 (分离器, 提示说明)。
    """
    if name != "auto":
        return get_separator(name), None
    for s in iter_separators():
        ok, _ = s.available()
        if ok:
            return s, None
    return CenterExtractSeparator(), "未找到高质量分离器，已使用中心声道兜底模式"


def separator_status() -> list[dict]:
    out = []
    for s in iter_separators():
        ok, reason = s.available()
        out.append({"name": s.name, "label": s.label, "kind": s.kind,
                    "description": s.description, "available": ok, "reason": reason})
    return out
