from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


@dataclass(frozen=True)
class Action:
    gain: float
    lines: int

    def __post_init__(self):
        if (self.gain not in (1.04, 13.43, 75.2) or not isinstance(self.lines, int)
                or isinstance(self.lines, bool) or not 40 <= self.lines <= 512):
            raise ValueError("Action outside bounded STD development envelope")

    @property
    def key(self):
        return f"g{self.gain:g}_l{self.lines}"

    @property
    def nominal_ms(self):
        return self.lines * 0.0065


class TaskKind(str, Enum):
    LOCALIZATION = "localization"
    RELATIVE_CONTRAST = "relative_contrast"
    RELATIVE_INTENSITY = "relative_intensity"


@dataclass(frozen=True)
class Task:
    kind: TaskKind
    tolerance: float
    version: int = 1

    def __post_init__(self):
        if (not isinstance(self.kind, TaskKind) or not isinstance(self.tolerance, (int, float))
                or isinstance(self.tolerance, bool) or not math.isfinite(self.tolerance)
                or not 0 < self.tolerance < 100):
            raise ValueError("Explicit task kind and positive tolerance required")
        if self.version < 1:
            raise ValueError("Task version must be positive")


@dataclass(frozen=True)
class Prediction:
    action: Action
    error: float
    upper_error: float
    min_snr: float
    supported: bool
    saturation_risk: bool
    loss: float
    reason: str = ""


@dataclass(frozen=True)
class Decision:
    action: Action | None
    task_version: int
    observation_seq: int
    reason: str
    predictions: tuple[Prediction, ...] = ()
