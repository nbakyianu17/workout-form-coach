"""Command-line entry point for Workout Form Coach."""

import argparse
from pathlib import Path
import sys

from .evaluation import evaluate_manifest, write_report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="workout-form-coach")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate squat counts from pose-observation CSV files"
    )
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument(
        "--standing-threshold",
        type=float,
        default=160.0,
        help="knee angle that completes a rep (degrees; default: 160)",
    )
    evaluate.add_argument(
        "--bottom-threshold",
        type=float,
        default=100.0,
        help="maximum knee angle considered a deep squat (degrees; default: 100)",
    )
    evaluate.add_argument(
        "--min-visibility",
        type=float,
        default=0.6,
        help="minimum landmark visibility accepted (0-1; default: 0.6)",
    )
    evaluate.add_argument(
        "--transition-frames",
        type=int,
        default=1,
        help="consecutive valid frames required to change phase (default: 1)",
    )

    args = parser.parse_args(argv)
    try:
        metrics = evaluate_manifest(
            manifest_path=args.manifest,
            standing_threshold=args.standing_threshold,
            bottom_threshold=args.bottom_threshold,
            min_visibility=args.min_visibility,
            transition_frames=args.transition_frames,
        )
        write_report(metrics, args.output)
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2

    total_error = sum(item.absolute_rep_error for item in metrics)
    mean_absolute_error = total_error / len(metrics)
    exact_count_clips = sum(item.absolute_rep_error == 0 for item in metrics)
    exact_count_rate = exact_count_clips / len(metrics)
    predicted_reps = sum(item.predicted_reps for item in metrics)
    ground_truth_reps = sum(item.ground_truth_reps for item in metrics)
    total_frames = sum(item.frames for item in metrics)
    missed_frames = sum(item.missed_detection_frames for item in metrics)
    missed_rate = missed_frames / total_frames
    print(
        "Evaluated {} clips; mean absolute rep-count error: {:.2f}. "
        "CSV replay throughput does not include video-model inference.".format(
            len(metrics), mean_absolute_error
        )
    )
    print(
        "Exact-count clips: {}/{} ({:.2%})".format(
            exact_count_clips, len(metrics), exact_count_rate
        )
    )
    print(
        "Rep totals: predicted {}, labeled {}".format(
            predicted_reps, ground_truth_reps
        )
    )
    print(
        "Missed detections: {}/{} ({:.2%})".format(
            missed_frames, total_frames, missed_rate
        )
    )
    print(
        "Thresholds: standing >= {:.1f} deg, bottom <= {:.1f} deg, "
        "visibility >= {:.2f}".format(
            args.standing_threshold,
            args.bottom_threshold,
            args.min_visibility,
        )
    )
    print("Transition confirmation: {} consecutive frame(s)".format(
        args.transition_frames
    ))
    print("Wrote per-clip metrics to {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
