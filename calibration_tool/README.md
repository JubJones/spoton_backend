### Phase 1: Detailed Setup and Calibration Blueprint

This is the one-time setup to create your unified tracking environment. Accuracy here is critical for the entire system's performance.

#### Step 1: Create the Blank Canvas Ground Plane

Your "Blank Canvas" is the single source of truth for all locations. It is a 2D coordinate system that will represent a top-down view of the entire area.

*   **Action:** Create a blank white or black image of a fixed size. For example, a 2000x1500 pixel image using OpenCV or PIL.
    ```python
    import numpy as np
    import cv2

    # Define the dimensions of your top-down map
    map_width = 2000
    map_height = 1500
    ground_plane_map = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    ```
*   **Concept of Scale:** You need to decide on a rough scale. For example, you might decide that 100 pixels on this map represents 1 meter in the real world. This isn't strictly necessary for the math to work, but it helps in setting realistic distance thresholds later.

#### Step 2: The Homography Calibration Workflow (Per Camera)

You must repeat this entire process for **each camera individually**.

**2a. Physical Setup:**
*   Place at least four, but preferably 6-8, distinct markers on the ground. These markers must be visible in the camera you are currently calibrating.
*   **Best Practice:** Use flat, non-symmetrical markers like colored L-shapes made of tape. This helps you identify the exact corner pixel. Cones are okay, but be consistent about using the center of the base.
*   Distribute these markers across the camera's field of view—some near, some far, some left, some right. This leads to a more accurate homography matrix.

**2b. Data Collection (The Two Sets of Points):**
This is the most important manual step. For the camera you are calibrating (let's say Camera 1):

1.  **Define Destination Points (on the Canvas):**
    *   Look at your physical setup and decide where those markers should logically be on your blank canvas map.
    *   Open your blank `ground_plane_map` image. For each physical marker, choose and record its `(x, y)` coordinate on the map. You are essentially "drawing" the layout of your markers onto the canvas.
    *   *Example:*
        *   Marker 1 is at `(350, 400)` on your map.
        *   Marker 2 is at `(350, 1100)` on your map.
        *   Marker 3 is at `(950, 450)` on your map.
        *   Marker 4 is at `(950, 1150)` on your map.
    *   This becomes your list of `destination_points`.

2.  **Collect Source Points (from the Camera Image):**
    *   Capture a clear, static frame from Camera 1.
    *   Using an image editor or a simple OpenCV script, find the exact pixel coordinates `(x, y)` of each of those markers *in that camera frame*.
    *   *Example (for Camera 1):*
        *   Marker 1 appears at pixel `(58, 420)`.
        *   Marker 2 appears at pixel `(75, 950)`.
        *   Marker 3 appears at pixel `(850, 410)`.
        *   Marker 4 appears at pixel `(890, 930)`.
    *   This becomes your list of `source_points`.

**2c. Computation:**
*   Use your two lists of points to compute the homography matrix for Camera 1.
    ```python
    # For Camera 1
    src_pts_cam1 = np.array([[58, 420], [75, 950], [850, 410], [890, 930]], dtype=np.float32)
    dst_pts_map = np.array([[350, 400], [350, 1100], [950, 450], [950, 1150]], dtype=np.float32)

    homography_matrix_cam1, _ = cv2.findHomography(src_pts_cam1, dst_pts_map)
    # Save this matrix for Camera 1
    ```
*   **Crucially, repeat steps 2a, 2b, and 2c for all other cameras.** Camera 2 will have its own set of source points and its own homography matrix, even if it maps to some of the same destination points on the canvas.

**2d. Verification (Sanity Check):**
*   Before moving on, test your matrix. Pick a few random points in the camera image, manually guess where they should land on the map, and then use `cv2.perspectiveTransform` to see how close you were. This ensures your matrix is behaving as expected.

#### Step 3: Define Entry and Exit Zones on the Canvas

Now that your coordinate system is defined, you can add semantic meaning to it.

*   **Action:** On your single `ground_plane_map` image, draw polygons that define logical transition areas.
*   **How to Define them:**
    *   An **"Exit Zone"** for Camera 1 is an area on the map where people tend to disappear from Camera 1's view.
    *   An **"Entry Zone"** for Camera 2 is an area on the map where people tend to first appear in Camera 2's view.
*   **Data Structure:** Store these as a list of polygons (arrays of points). You will use them later to check if a person's last/first appearance occurred in one of these zones, which is a strong hint that a re-identification is needed.
    *   *Example:* `entry_zones['cam2'] = [np.array([[1200, 300], [1400, 310], ...]])]`

At the end of Phase 1, you have:
1.  A single `ground_plane_map` image.
2.  A unique `homography_matrix` for each camera.
3.  A set of `entry/exit_zones` defined on the map.
