"""Knee-angle math and a visibility-aware squat repetition state machine."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

Point = Tuple[float, float]


class Phase(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    ASCENDING = "ascending"


@dataclass(frozen=True)
class CounterUpdate:
    reps: int
    phase: Phase
    accepted: bool
    counted: bool = False


def knee_angle(hip: Point, knee: Point, ankle: Point) -> Optional[float]:
    """Return the interior hip-knee-ankle angle in degrees, or None if degenerate."""
    coordinates = (*hip, *knee, *ankle)
    if not all(math.isfinite(value) for value in coordinates):
        return None

    hip_vector = (hip[0] - knee[0], hip[1] - knee[1])
    ankle_vector = (ankle[0] - knee[0], ankle[1] - knee[1])
    hip_length = math.hypot(*hip_vector)
    ankle_length = math.hypot(*ankle_vector)
    if hip_length == 0.0 or ankle_length == 0.0:
        return None

    cosine = (
        hip_vector[0] * ankle_vector[0] + hip_vector[1] * ankle_vector[1]
    ) / (hip_length * ankle_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


class SquatCounter:
    """Count complete standing -> deep squat -> standing cycles."""

    def __init__(
        self,
        standing_threshold: float = 160.0,
        bottom_threshold: float = 100.0,
        min_visibility: float = 0.6,
    ) -> None:
        if not 0.0 <= bottom_threshold < standing_threshold <= 180.0:
            raise ValueError(
                "thresholds must satisfy 0 <= bottom < standing <= 180"
            )
        if not 0.0 <= min_visibility <= 1.0:
            raise ValueError("min_visibility must be between 0 and 1")
        self.standing_threshold = standing_threshold
        self.bottom_threshold = bottom_threshold
        self.min_visibility = min_visibility
        self.phase = Phase.STANDING
        self.reps = 0

    def update(
        self, angle: Optional[float], visibility: float = 1.0
    ) -> CounterUpdate:
        """Consume one frame's knee angle and landmark visibility."""
        if not math.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
            raise ValueError("visibility must be a finite value between 0 and 1")
        if angle is None or visibility < self.min_visibility:
            return CounterUpdate(self.reps, self.phase, accepted=False)
        if not math.isfinite(angle) or not 0.0 <= angle <= 180.0:
            raise ValueError("angle must be a finite value between 0 and 180")

        counted = False
        if self.phase is Phase.STANDING:
            if angle < self.standing_threshold:
                self.phase = Phase.DESCENDING
                if angle <= self.bottom_threshold:
                    self.phase = Phase.ASCENDING
        elif self.phase is Phase.DESCENDING:
            if angle <= self.bottom_threshold:
                self.phase = Phase.ASCENDING
            elif angle >= self.standing_threshold:
                self.phase = Phase.STANDING
        elif self.phase is Phase.ASCENDING and angle >= self.standing_threshold:
            self.reps += 1
            self.phase = Phase.STANDING
            counted = True

        return CounterUpdate(self.reps, self.phase, accepted=True, counted=counted)
