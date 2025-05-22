# SpotOn Frontend Integration Guide

This guide provides frontend developers with the necessary steps and information to integrate with the SpotOn backend for displaying video streams, person detection bounding boxes, tracking IDs, and map coordinates.

## 0. Prerequisites

*   The SpotOn backend server is running and accessible (e.g., at `http://localhost:8000`).
*   You have a basic understanding of REST APIs and WebSockets.
*   Your frontend application is capable of:
    *   Making HTTP GET and POST requests.
    *   Establishing and handling WebSocket connections.
    *   Playing video streams (e.g., using HTML5 `<video>` elements).
    *   Drawing overlays (like bounding boxes and text) on top of video players.
    *   Displaying a 2D map and plotting points on it.

## 1. Initial Backend Health Check (Critical for Startup)

Before initiating any processing task or attempting to connect to WebSockets, it's **essential** to ensure the backend is fully initialized and ready. The backend loads AI models and performs other setup tasks on startup, which can take some time.

*   **Action:** Continuously poll the backend's `/health` REST endpoint.
    *   **Endpoint:** `GET {BACKEND_BASE_URL}/health` (e.g., `http://localhost:8000/health`)
    *   **Expected Success Response (200 OK):**
        ```json
        {
          "status": "healthy", // or "degraded" if some non-critical parts are down
          "detector_model_loaded": true,
          "prototype_tracker_loaded (reid_model)": true,
          "homography_matrices_precomputed": true
        }
        ```
    *   **Frontend Logic:**
        1.  On frontend application load, or before starting a new analysis, start polling `/health`.
        2.  Use a reasonable polling interval (e.g., every 3-5 seconds) with a maximum number of retries.
        3.  **Proceed to the next step (Initiate Processing Task) ONLY when:**
            *   The HTTP request to `/health` returns a `200 OK` status.
            *   The `status` field in the response is `"healthy"`.
            *   All critical boolean flags (`detector_model_loaded`, `prototype_tracker_loaded (reid_model)`, `homography_matrices_precomputed`) are `true`.
        4.  If the backend doesn't become healthy after a timeout or max retries, display an appropriate error message to the user indicating the backend is unavailable or still starting.

## 2. Initiate Processing Task

Once the backend is healthy, the user can select an environment (e.g., "campus," "factory") to analyze.

*   **Action:** Send a POST request to start a processing task.
    *   **Endpoint:** `POST {BACKEND_BASE_URL}/api/v1/processing-tasks/start`
    *   **Request Body (JSON):**
        ```json
        {
          "environment_id": "campus" // Or "factory", etc.
        }
        ```
    *   **Expected Success Response (202 Accepted):**
        ```json
        {
          "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", // A unique UUID for this task
          "message": "Processing task for environment 'campus' initiated.",
          "status_url": "/api/v1/processing-tasks/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/status", // Relative URL
          "websocket_url": "/ws/tracking/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" // Relative URL
        }
        ```
    *   **Frontend Logic:**
        1.  Store the `task_id`. This ID is crucial for all subsequent communication related to this specific analysis session.
        2.  Store the full WebSocket URL by prepending your WebSocket base URL (e.g., `ws://localhost:8000`) to the relative `websocket_url` path.

## 3. Establish WebSocket Connection

With the WebSocket URL obtained, connect to the backend to receive real-time updates.

*   **Action:** Open a WebSocket connection to the URL provided in the previous step.
    *   Example URL: `ws://localhost:8000/ws/tracking/{task_id}`
*   **Frontend Logic:**
    1.  Attempt to establish the WebSocket connection.
    2.  **Error Handling:** If the connection fails (e.g., backend was quick to respond to REST but WebSockets aren't fully up yet, or network issues), you might briefly re-check `/health` and implement a retry mechanism for the WebSocket connection with exponential backoff.
    3.  Once connected, keep the connection open to listen for messages from the backend.

## 4. Handling WebSocket Messages from the Backend

The backend will push various message types over the WebSocket. Your frontend needs to parse these JSON messages and act accordingly. Each message will have a `type` field and a `payload` field.

### 4.1. `status_update` Message

Provides updates on the overall progress and current stage of the processing task.

*   **Example Payload:**
    ```json
    {
      "type": "status_update",
      "payload": {
        "task_id": "...",
        "status": "PROCESSING", // e.g., QUEUED, INITIALIZING, DOWNLOADING_DATA, PROCESSING, COMPLETED, FAILED
        "progress": 0.2550,    // Overall task progress (0.0 to 1.0)
        "current_step": "Processing sub-video 1, frame batch 80 (Global Index: 79)",
        "details": "Optional additional info or error message if status is FAILED"
      }
    }
    ```
*   **Frontend Logic:**
    1.  Update UI elements to reflect the `status`, `progress`, and `current_step` (e.g., progress bar, status text).
    2.  If `status` is "FAILED", display the `details` message to the user.
    3.  If `status` is "COMPLETED", indicate that the analysis is finished.

### 4.2. `media_available` Message (Crucial for Video Playback)

Notifies the frontend that a new batch of sub-videos is ready on the backend for streaming. This message provides the necessary URLs and synchronization information.

*   **Example Payload:**
    ```json
    {
      "type": "media_available",
      "payload": {
        "sub_video_batch_index": 0, // 0-indexed identifier for this batch of sub-videos
        "media_urls": [
          {
            "camera_id": "c01",
            "sub_video_filename": "sub_video_01.mp4",
            "url": "/api/v1/media/tasks/.../c01/sub_videos/sub_video_01.mp4", // Backend-relative URL
            "start_global_frame_index": 0,
            "num_frames_in_sub_video": 750 // Example: 30s video @ 25fps
          },
          {
            "camera_id": "c02",
            "sub_video_filename": "sub_video_01.mp4",
            "url": "/api/v1/media/tasks/.../c02/sub_videos/sub_video_01.mp4",
            "start_global_frame_index": 0,
            "num_frames_in_sub_video": 750
          }
          // ... more entries if multiple cameras are processed in parallel for this batch
        ]
      }
    }
    ```
*   **Frontend Logic:**
    1.  Parse `sub_video_batch_index`.
    2.  For each entry in `media_urls`:
        *   Construct the full HTTP URL: `{BACKEND_BASE_URL}{entry.url}` (e.g., `http://localhost:8000/api/v1/media/...`).
        *   Store `start_global_frame_index` and `num_frames_in_sub_video` associated with this camera and `sub_video_batch_index`. This mapping is VITAL for synchronization.
        *   Update the `src` attribute of the HTML `<video>` player corresponding to `camera_id` with this new full URL.
        *   Listen for video player events like `canplaythrough` or `loadeddata` to know when the video is ready.
        *   **Do not auto-play immediately.** Playback should be managed in conjunction with incoming `tracking_update` messages or user interaction.

### 4.3. `tracking_update` Message (Core Data for Overlays)

Contains the detected/tracked person data for a batch of frames.

*   **Example Payload:**
    ```json
    {
      "type": "tracking_update",
      "payload": {
        "global_frame_index": 123, // Absolute frame index for the task
        "scene_id": "campus",      // Corresponds to environment_id
        "timestamp_processed_utc": "2023-10-27T10:30:05.456Z",
        "cameras": {
          "c01": { // Camera ID (string)
            "image_source": "000123.jpg", // Identifier for the frame (can be based on global_frame_index)
            "tracks": [
              {
                "track_id": 5, // Intra-camera track ID (integer)
                "global_id": "person-uuid-abc-123", // System-wide unique person ID (string, null if not identified)
                "bbox_xyxy": [110.2, 220.5, 160.0, 330.8], // Bounding box [x1, y1, x2, y2]
                "confidence": 0.92, // Detection confidence (float, optional)
                "class_id": 1,      // Object class ID (integer, e.g., 1 for person)
                "map_coords": [12.3, 45.6] // Projected [X, Y] on the map (null if no homography)
              }
              // ... more tracked persons in camera "c01"
            ]
          },
          "c02": { /* ... data for camera "c02" ... */ }
          // ... other cameras active in this frame batch
        }
      }
    }
    ```
*   **Frontend Logic (Synchronization is Key!):**
    1.  When a `tracking_update` message arrives, get its `global_frame_index`.
    2.  **Determine the Active Sub-Video:** Identify which `sub_video_batch_index` is currently being displayed or is cued up for display based on the `media_available` messages received.
    3.  **Map Global to Local Frame:** Using the `start_global_frame_index` and `num_frames_in_sub_video` (stored from the `media_available` message for the active sub-video batch), check if the incoming `global_frame_index` falls within the range of the currently loaded sub-video for each camera.
        *   `local_frame_offset = global_frame_index - start_global_frame_index_of_current_sub_video`
    4.  **Synchronize with Video Player:**
        *   Get the current playback time of the video player for each camera. Convert this time to a frame number based on the video's FPS (or the processing FPS if backend guarantees fixed rate processing per sub-video).
        *   Compare this player's current frame number with the `local_frame_offset` calculated above.
        *   **Apply Overlays:** If the `tracking_update` corresponds to the frame currently (or very nearly) visible in the player:
            *   For each camera in the `cameras` dictionary of the payload:
                *   Iterate through its `tracks`.
                *   Use `bbox_xyxy` to draw bounding boxes on that camera's video player.
                *   Display `global_id` (or `track_id` if `global_id` is null) near the box.
                *   If `map_coords` are available, plot a point/marker for this `global_id` on your 2D map display.
    5.  **Buffering Metadata:** If `tracking_update` messages arrive for frames that are "ahead" of what the video player is currently showing (e.g., video is still buffering or playing slower), the frontend **must queue/buffer these metadata messages.** As the video player catches up to a particular frame, the frontend can then retrieve and apply the corresponding buffered metadata. This prevents losing information and ensures overlays appear at the correct time.
    6.  **Starting Playback:** Once a video player for a new sub-video batch is ready (e.g., `canplaythrough` event) and you start receiving `tracking_update` messages that fall within its `global_frame_index` range, you can confidently start playing that video.

### 4.4. `batch_processing_complete` Message

Indicates that the backend has finished processing all frames for a specific sub-video batch.

*   **Example Payload:**
    ```json
    {
      "type": "batch_processing_complete",
      "payload": {
        "sub_video_batch_index": 0 // The 0-indexed batch that just finished
      }
    }
    ```
*   **Frontend Logic:**
    1.  Use this as a signal that no more `tracking_update` messages will arrive for this `sub_video_batch_index`.
    2.  This can help in managing the transition to the next sub-video batch. For instance, if the current video player for this batch has also finished playing, you can now confidently switch to the players for the next batch (if their `media_available` has been received).

## 5. Managing Video Transitions

*   When a sub-video for `batch_X` finishes playing (player "ended" event) AND/OR you receive `batch_processing_complete` for `batch_X`:
    *   Check if `media_available` for `batch_X+1` has been received.
    *   If yes, and the video players for `batch_X+1` are loaded/ready, switch the active display to these new videos.
    *   The synchronization logic described in 4.3 (Handling `tracking_update`) will then apply to this new batch.

## 6. Task End

*   The task is considered fully complete when the `status_update` message indicates `status: "COMPLETED"`.
*   At this point, the WebSocket connection might be closed by the server, or you can close it from the frontend.

## Summary of Frontend Responsibilities

1.  **Health Check:** Poll `/health` before anything else.
2.  **Initiate Task:** `POST` to `/api/v1/processing-tasks/start`.
3.  **Connect WebSocket:** Establish and maintain the WebSocket connection.
4.  **Handle `media_available`:**
    *   Load new sub-video URLs into players.
    *   Store `start_global_frame_index` and `num_frames_in_sub_video` for each.
5.  **Handle `tracking_update`:**
    *   Synchronize `global_frame_index` with current video playback.
    *   Draw bounding boxes and other overlays.
    *   Update map display with `map_coords`.
    *   Buffer metadata if it arrives ahead of video playback.
6.  **Handle `status_update`:** Update UI with progress and status.
7.  **Handle `batch_processing_complete`:** Use as a signal for managing sub-video transitions.
8.  **Manage Video Player State:** Control play/pause, handle "ended" events, and switch video sources for new batches.
9.  **Error Handling:** Gracefully handle API errors, WebSocket disconnections, and missing data.

By following these steps and understanding the data flow, frontend developers can build a responsive and accurately synchronized interface for the SpotOn system.