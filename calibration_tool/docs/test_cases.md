# Manual Validation Checklist

1. Activate the virtual environment and install dependencies.
2. Launch the tool: `python -m calibration_tool.app.main`.
3. Load a sample camera frame image.
4. Click four or more markers on the camera image, recording their coordinates.
5. Click the matching locations on the ground plane image in the same order.
6. Compute the homography and verify the 3x3 matrix populates in the UI.
7. Press "Test Transform" and click a point on the camera frame to confirm the projected location appears on the ground plane.
8. Export the calibration and confirm a JSON file plus optional overlays appear under `calibration_tool/exports/`.
