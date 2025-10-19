# Calibration Tool Build Plan

## Goal
- Build a single-user desktop utility that lets you collect homography calibration points per camera with point-and-click annotations and export the matrices/results.
- Keep dependencies minimal and stick closely to the calibration steps laid out in `README.md`.

## Step-by-Step Roadmap

1. **Prepare the Environment**
   - Use Python 3.10+ and create/activate a virtual environment just for this tool.
   - Install only the essentials: `opencv-python`, `numpy`, and `Pillow` for image handling. The UI layer will rely on the standard `tkinter` module (already bundled with Python).

2. **Organize the Project Skeleton**
   - Create folders: `app/` for code, `data/` for saved frames, `exports/` for results, `assets/` for the ground plane image.
   - Start a simple `main.py` entry point and a `requirements.txt` listing the three dependencies.

3. **Generate the Blank Canvas Ground Plane**
   - Add a small script/function that creates the 2000x1500 blank map per the README and stores it in `assets/ground_plane_map.png`.
   - Include an input for scale metadata (e.g., pixels-per-meter) and store it alongside the PNG in a `ground_plane.json`.

4. **Build the Minimal UI Frame**
   - Construct the window directly with `tkinter`, using `Frame` sections and `Canvas` widgets for the camera view and ground plane, plus a side panel for controls.
   - Add buttons for: `Load Camera Frame`, `Add Camera Point`, `Add Map Point`, `Undo Last`, `Compute Homography`, and `Export`.

5. **Implement Point Collection Logic**
   - When the user clicks on the camera image, append the (x, y) pixel to the current camera’s source point list (show the point index overlay).
   - Mirror the same interaction for the ground plane image to capture destination points (ensure pairs align by index).
   - Enforce at least four point pairs and display warnings if counts mismatch.

6. **Compute and Display the Homography**
   - Once four or more pairs exist, enable `Compute Homography`.
   - Call `cv2.findHomography` with the collected points to produce the matrix; show it in the side panel.
   - Provide a quick sanity check: allow the user to click a test point on the camera image and display its projected location on the map.

7. **Repeat Flow Per Camera**
   - Maintain a simple dropdown to switch between cameras (Camera 1, Camera 2, etc.).
   - Persist each camera’s source/destination points and homography matrix in memory while the app is open.

8. **Export Results**
   - On `Export`, write a JSON file in `exports/` containing:
     - Ground plane metadata (image path, scale).
     - A record per camera with point lists and the 3x3 homography matrix.
   - Optionally include a PNG of the map with plotted points for documentation.

9. **Define Entry/Exit Zones (Optional Extra)**
   - If needed later, add a simple polygon drawing tool on the ground plane to label entry/exit zones as described in the README and include them in the export.

10. **Lightweight Validation**
    - Test the flow end-to-end with sample images: load a frame, mark points, compute homography, verify transformations, then export.
    - Keep a short checklist in the repo (e.g., `test_cases.md`) noting the manual steps taken for validation.
