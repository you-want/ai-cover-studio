"""分离器基类与结果结构。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# 进度回调：progress(0.0~1.0 阶段内进度, 消息)
ProgressCB = Callable[[float, str], None]


@dataclass
class SeparationResult:
    vocals: Path
    instrumental: Path
    note: str = ""


class BaseSeparator:
    name: str = "base"
    label: str = "基类"
    kind: str = "base"          # model / dsp
    description: str = ""

    def available(self) -> tuple[bool, str]:
        return True, ""

    def separate(self, song_wav: Path, out_dir: Path,
                 progress: ProgressCB = lambda p, m: None) -> SeparationResult:
        raise NotImplementedError
