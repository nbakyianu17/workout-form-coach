import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from workout_form_coach.evaluation import evaluate_clip, evaluate_manifest, write_report


POSE_FIELDS = [
    "frame",
    "hip_x",
    "hip_y",
    "knee_x",
    "knee_y",
    "ankle_x",
    "ankle_y",
    "visibility",
]


def pose_row(frame, angle, visibility=1.0):
    if angle is None:
        coordinates = ["", "", "", "", "", ""]
    else:
        radians = math.radians(angle)
        coordinates = [
            "0",
            "1",
            "0",
            "0",
            str(math.sin(radians)),
            str(math.cos(radians)),
        ]
    return dict(
        zip(POSE_FIELDS, [str(frame)] + coordinates + [str(visibility)])
    )


class EvaluationTests(unittest.TestCase):
    def write_pose_csv(self, directory, name, rows):
        path = directory / name
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=POSE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_clip_metrics_count_reps_and_missed_detections(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pose_path = self.write_pose_csv(
                directory,
                "observations.csv",
                [
                    pose_row(0, 175),
                    pose_row(1, 145),
                    pose_row(2, None, visibility=0.2),
                    pose_row(3, 95),
                    pose_row(4, 170),
                ],
            )
            metrics = evaluate_clip("test-clip", pose_path, ground_truth_reps=1)

        self.assertEqual(metrics.frames, 5)
        self.assertEqual(metrics.predicted_reps, 1)
        self.assertEqual(metrics.absolute_rep_error, 0)
        self.assertEqual(metrics.missed_detection_frames, 1)
        self.assertAlmostEqual(metrics.missed_detection_rate, 0.2)
        self.assertGreater(metrics.processing_fps, 0)

    def test_manifest_writes_machine_readable_per_clip_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_pose_csv(
                directory,
                "clip.csv",
                [pose_row(0, 175), pose_row(1, 145), pose_row(2, 95),
                 pose_row(3, 170)],
            )
            manifest = directory / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=["clip_id", "observations_csv", "ground_truth_reps"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "clip_id": "clip-01",
                        "observations_csv": "clip.csv",
                        "ground_truth_reps": "1",
                    }
                )
            report = directory / "reports" / "metrics.csv"
            metrics = evaluate_manifest(manifest)
            write_report(metrics, report)

            with report.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["clip_id"], "clip-01")
        self.assertEqual(rows[0]["predicted_reps"], "1")
        self.assertEqual(rows[0]["absolute_rep_error"], "0")
        self.assertIn("processing_fps", rows[0])

    def test_cli_writes_report_and_labels_its_throughput_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_pose_csv(
                directory,
                "clip.csv",
                [pose_row(0, 175), pose_row(1, 145), pose_row(2, 95),
                 pose_row(3, 170)],
            )
            manifest = directory / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=["clip_id", "observations_csv", "ground_truth_reps"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "clip_id": "clip-01",
                        "observations_csv": "clip.csv",
                        "ground_truth_reps": "1",
                    }
                )
            report = directory / "metrics.csv"
            project_root = Path(__file__).resolve().parents[1]
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "workout_form_coach",
                    "evaluate",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(report),
                ],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                check=False,
            )
            report_exists = report.is_file()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("does not include video-model inference", completed.stdout)
        self.assertTrue(report_exists)

    def test_manifest_rejects_duplicate_ids_and_bad_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_pose_csv(directory, "clip.csv", [pose_row(0, 170)])
            manifest = directory / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=["clip_id", "observations_csv", "ground_truth_reps"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "clip_id": "same",
                        "observations_csv": "clip.csv",
                        "ground_truth_reps": "0",
                    }
                )
                writer.writerow(
                    {
                        "clip_id": "same",
                        "observations_csv": "clip.csv",
                        "ground_truth_reps": "0",
                    }
                )
            with self.assertRaisesRegex(ValueError, "duplicate clip_id"):
                evaluate_manifest(manifest)

    def test_clip_rejects_non_contiguous_frame_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            pose_path = self.write_pose_csv(
                Path(temporary),
                "clip.csv",
                [pose_row(0, 170), pose_row(2, 150)],
            )
            with self.assertRaisesRegex(ValueError, "contiguous"):
                evaluate_clip("bad-clip", pose_path, ground_truth_reps=0)


if __name__ == "__main__":
    unittest.main()
