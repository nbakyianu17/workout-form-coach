# Workout Form Coach

Workout Form Coach is an in-progress computer-vision squat-counting project.
The work is being added in tested milestones; this repository does not yet
analyze video or claim accuracy results.

## Current milestone

**Implemented:** a pure Python knee-angle helper and a three-phase squat
counter (standing, descending, ascending) that ignores missing or low-
visibility observations. Unit tests cover angle geometry, full and partial
movement cycles, confidence filtering, and invalid input.

**Not implemented yet:** person detection or cropping, MediaPipe pose
estimation, video ingestion, annotated output, labeled-clip evaluation, and
rep-error/FPS measurements.

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
