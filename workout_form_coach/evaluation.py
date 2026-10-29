"""Replay pose-observation CSVs and measure clip-level squat counts."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Tuple

from .counter import SquatCounter, knee_angle

POSE_COLUMNS = (
    "frame",
    "hip_x",
    "hip_y",
    "knee_x",
    "knee_y",
    "ankle_x",
    "ankle_y",
    "visibility",
)
MANIFEST_COLUMNS = ("clip_id", "observations_csv", "ground_truth_reps")


@dataclass(frozen=True)
class ClipMetrics:
    clip_id: str
    frames: int
    predicted_reps: int
    ground_truth_reps: int
    absolute_rep_error: int
    missed_detection_frames: int
    missed_detection_rate: float
    processing_fps: float


def _read_point(row: Dict[str, str], prefix: str) -> Optional[Tuple[float, float]]:
    x_text = row[prefix + "_x"].strip()
    y_text = row[prefix + "_y"].strip()
    if not x_text and not y_text:
        return None
    if not x_text or not y_text:
        return None
    try:
        point = (float(x_text), float(y_text))
    except ValueError as error:
        raise ValueError("invalid {} coordinates in pose CSV".format(prefix)) from error
    return point


def _visibility(row: Dict[str, str]) -> float:
    text = row["visibility"].strip()
    if not text:
        return 0.0
    try:
        visibility = float(text)
    except ValueError as error:
        raise ValueError("invalid visibility in pose CSV") from error
    if not math.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
        raise ValueError("visibility must be between 0 and 1")
    return visibility


def evaluate_clip(
    clip_id: str,
    observations_path: Path,
    ground_truth_reps: int,
    standing_threshold: float = 160.0,
    bottom_threshold: float = 100.0,
    min_visibility: float = 0.6,
) -> ClipMetrics:
    if ground_truth_reps < 0:
        raise ValueError("ground-truth reps must be non-negative")

    counter = SquatCounter(
        standing_threshold=standing_threshold,
        bottom_threshold=bottom_threshold,
        min_visibility=min_visibility,
    )
    frames = 0
    missed_detection_frames = 0
    last_frame = -1
    started = time.perf_counter()

    with observations_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("pose CSV is empty: {}".format(observations_path))
        missing_columns = [name for name in POSE_COLUMNS if name not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                "pose CSV is missing columns: {}".format(", ".join(missing_columns))
            )

        for row in reader:
            frames += 1
            try:
                frame = int(row["frame"])
            except (TypeError, ValueError) as error:
                raise ValueError("frame must be a non-negative integer") from error
            if frame != last_frame + 1:
                raise ValueError("frame numbers must start at zero and be contiguous")
            last_frame = frame

            hip = _read_point(row, "hip")
            knee = _read_point(row, "knee")
            ankle = _read_point(row, "ankle")
            visibility = _visibility(row)
            angle = (
                knee_angle(hip, knee, ankle)
                if hip is not None and knee is not None and ankle is not None
                else None
            )
            update = counter.update(angle, visibility)
            if not update.accepted:
                missed_detection_frames += 1

    if frames == 0:
        raise ValueError("pose CSV contains no frame observations: {}".format(observations_path))

    elapsed = time.perf_counter() - started
    processing_fps = frames / elapsed if elapsed > 0 else float("inf")
    missed_rate = missed_detection_frames / frames
    predicted_reps = counter.reps
    return ClipMetrics(
        clip_id=clip_id,
        frames=frames,
        predicted_reps=predicted_reps,
        ground_truth_reps=ground_truth_reps,
        absolute_rep_error=abs(predicted_reps - ground_truth_reps),
        missed_detection_frames=missed_detection_frames,
        missed_detection_rate=missed_rate,
        processing_fps=processing_fps,
    )


def evaluate_manifest(
    manifest_path: Path,
    standing_threshold: float = 160.0,
    bottom_threshold: float = 100.0,
    min_visibility: float = 0.6,
) -> List[ClipMetrics]:
    results: List[ClipMetrics] = []
    seen_clip_ids = set()
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("manifest is empty: {}".format(manifest_path))
        missing_columns = [
            name for name in MANIFEST_COLUMNS if name not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "manifest is missing columns: {}".format(", ".join(missing_columns))
            )

        for row in reader:
            clip_id = row["clip_id"].strip()
            observations_name = row["observations_csv"].strip()
            if not clip_id or not observations_name:
                raise ValueError("manifest clip_id and observations_csv are required")
            if clip_id in seen_clip_ids:
                raise ValueError("duplicate clip_id in manifest: {}".format(clip_id))
            seen_clip_ids.add(clip_id)
            try:
                ground_truth_reps = int(row["ground_truth_reps"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "ground_truth_reps must be a non-negative integer"
                ) from error

            observations_path = manifest_path.parent / observations_name
            results.append(
                evaluate_clip(
                    clip_id=clip_id,
                    observations_path=observations_path,
                    ground_truth_reps=ground_truth_reps,
                    standing_threshold=standing_threshold,
                    bottom_threshold=bottom_threshold,
                    min_visibility=min_visibility,
                )
            )
    if not results:
        raise ValueError("manifest contains no clips")
    return results


def write_report(metrics: Iterable[ClipMetrics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "clip_id",
                "frames",
                "predicted_reps",
                "ground_truth_reps",
                "absolute_rep_error",
                "missed_detection_frames",
                "missed_detection_rate",
                "processing_fps",
            ),
        )
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "clip_id": item.clip_id,
                    "frames": item.frames,
                    "predicted_reps": item.predicted_reps,
                    "ground_truth_reps": item.ground_truth_reps,
                    "absolute_rep_error": item.absolute_rep_error,
                    "missed_detection_frames": item.missed_detection_frames,
                    "missed_detection_rate": "{:.6f}".format(
                        item.missed_detection_rate
                    ),
                    "processing_fps": "{:.2f}".format(item.processing_fps),
                }
            )
