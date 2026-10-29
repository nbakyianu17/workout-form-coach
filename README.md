# Workout Form Coach

Workout Form Coach is an in-progress computer-vision squat-counting project.
The work is being added in tested milestones; this repository does not yet
analyze video or claim accuracy results.

## Current milestone

**Implemented:** a pure Python knee-angle helper and a three-phase squat
counter (standing, descending, ascending) that ignores missing or low-
visibility observations; plus a CSV replay evaluator that compares predicted
rep counts to labeled counts and reports missed detections and CSV-processing
throughput. Unit tests cover angle geometry, full and partial movement cycles,
confidence filtering, invalid input, and per-clip report output.

**Not implemented yet:** person detection or cropping, MediaPipe pose
estimation, video ingestion, annotated output, or evaluation on real labeled
clips. The current FPS field measures replaying pose CSV rows only; it is not
end-to-end video or model inference speed. No rep-count accuracy is claimed
without a real labeled dataset.

## Run the tests

Python 3.9 or newer is required. The initial counter and tests use only the
Python standard library:

```sh
python3 -m unittest discover -s tests -v
```

The counter accepts a knee angle in degrees and a visibility score per frame.
It counts a rep only after the observed angle moves below the configured
bottom threshold and later returns above the standing threshold. This
threshold-based prototype is not a medical or coaching assessment.

## Replay labeled pose observations

The evaluator currently consumes per-frame landmarks exported to CSV; it does
not yet run a pose-estimation or detection model. Use normalized `x` and `y`
coordinates and include these columns in each observations CSV:

```csv
frame,hip_x,hip_y,knee_x,knee_y,ankle_x,ankle_y,visibility
```

Include one row per frame, with contiguous zero-based frame numbers. Missing
landmark coordinates or visibility below the threshold count as a missed
detection frame. Create a manifest with paths relative to the manifest:

```csv
clip_id,observations_csv,ground_truth_reps
clip-01,clip-01.csv,12
```

Then write a per-clip report:

```sh
python3 -m workout_form_coach evaluate \
  --manifest dataset/manifest.csv \
  --output reports/evaluation.csv
```
