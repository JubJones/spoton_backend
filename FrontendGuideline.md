# SpotOn Frontend Integration Guide

This guide provides frontend developers with the necessary steps and information to integrate with the SpotOn backend for displaying person detection bounding boxes, tracking IDs, map coordinates, and the video frames themselves directly via WebSockets.

## 0. Prerequisites

*   The SpotOn backend server is running and accessible (e.g., at `http://localhost:3847`).
*   You have a basic understanding of REST APIs and WebSockets.
*   Your frontend application is capable of:
    *   Making HTTP GET and POST requests.
    *   Establishing and handling WebSocket connections.
    *   Decoding base64 encoded images and displaying them (e.g., in an `<img>` tag or on a `<canvas>`).
    *   Drawing overlays (like bounding boxes and text) on top of these displayed images.
    *   Displaying a 2D map and plotting points on it.

## 1. Initial Backend Health Check (Critical for Startup)

Before initiating any processing task or attempting to connect to WebSockets, it's **essential** to ensure the backend is fully initialized and ready.

*   **Action:** Continuously poll the backend's `/health` REST endpoint.
    *   **Endpoint:** `GET {BACKEND_BASE_URL}/health` (e.g., `http://localhost:3847/health`)
    *   **Expected Success Response (200 OK):**
        ```json
        {
          "status": "healthy",
          "detector_model_loaded": true,
          "prototype_tracker_loaded": true,
          "homography_matrices_precomputed": true
        }
        ```
    *   **Frontend Logic:**
        1.  On frontend application load, or before starting a new analysis, start polling `/health`.
        2.  Use a reasonable polling interval (e.g., every 3-5 seconds) with a maximum number of retries.
        3.  **Proceed to the next step (Initiate Processing Task) ONLY when:**
            *   The HTTP request to `/health` returns a `200 OK` status.
            *   The `status` field in the response is `"healthy"`.
            *   All critical boolean flags are `true`.
        4.  If the backend doesn't become healthy, display an error message.

## 2. Initiate Processing Task

Once the backend is healthy, the user can select an environment to analyze.

*   **Action:** Send a POST request to start a processing task.
    *   **Endpoint:** `POST {BACKEND_BASE_URL}/api/v1/detection-processing-tasks/start`
    *   **Request Body (JSON):** `{"environment_id": "campus"}`
    *   **Expected Success Response (202 Accepted):**
        ```json
        {
          "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
          "message": "Processing task initiated.",
          "status_url": "/api/v1/detection-processing-tasks/{task_id}/status",
          "websocket_url": "/ws/tracking/{task_id}"
        }
        ```
    *   **Frontend Logic:** Store `task_id` and the full WebSocket URL.

## 3. Establish WebSocket Connection

Connect to the backend to receive real-time updates.

*   **Action:** Open a WebSocket connection to the URL from the previous step (e.g., `ws://localhost:3847/ws/tracking/{task_id}`).
*   **Frontend Logic:**
    1.  Attempt WebSocket connection.
    2.  **Error Handling:** If connection fails (e.g., backend still initializing WebSockets), re-check `/health` and implement retry logic with backoff.
    3.  Listen for messages once connected.

## 4. Handling WebSocket Messages from the Backend

The backend pushes JSON messages with a `type` and `payload`.

### 4.1. `status_update` Message

Provides updates on overall task progress.

*   **Example Payload:**
    ```json
    {
      "type": "status_update",
      "payload": {
        "task_id": "...",
        "status": "PROCESSING", // e.g., QUEUED, INITIALIZING, PROCESSING, COMPLETED, FAILED
        "progress": 0.2550,
        "current_step": "Processing frame batch 80 (Global Index: 79)",
        "details": "Optional info or error message"
      }
    }
    ```
*   **Frontend Logic:** Update UI with `status`, `progress`, `current_step`. Display `details` if `status` is "FAILED".

### 4.2. `tracking_update` Message (Core Data: Image + Overlays)

Contains the **frame image** and detected/tracked person data. This is the primary message for displaying the video feed and overlays.

*   **Example Payload:**
    ```json
    {
      "type": "tracking_update",
      "payload": {
        "global_frame_index": 123,
        "scene_id": "campus",
        "timestamp_processed_utc": "2023-10-27T10:30:05.456Z",
        "cameras": {
          "c01": { // Camera ID (string)
            "image_source": "000123.jpg", // Identifier for the frame
            "frame_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...", // Base64 encoded JPEG/PNG string << NEW
            "tracks": [
              {
                "track_id": 5,
                "global_id": "person-uuid-abc-123",
                "bbox_xyxy": [110.2, 220.5, 160.0, 330.8],
                "confidence": 0.92,
                "class_id": 1,
                "map_coords": [12.3, 45.6]
              }
              // ... more tracked persons in camera "c01"
            ]
          },
          "c02": { /* ... data for camera "c02", including its own frame_image_base64 ... */ }
        }
      }
    }
    ```
*   **Frontend Logic (Synchronization is Inherent):**
    1.  When a `tracking_update` message arrives, parse its `payload`.
    2.  For each camera ID in the `cameras` dictionary:
        *   Get the `frame_image_base64` string.
        *   **Display the Image:** Decode the base64 string and render it. For example, if using an `<img>` tag:
            ```javascript
            // Assuming 'imgElement' is your <img> DOM element for this camera
            // and 'base64ImageData' is the frame_image_base64 string.
            // The format (jpeg, png) should ideally be known or part of the message,
            // but 'image/jpeg' is common.
            imgElement.src = 'data:image/jpeg;base64,' + base64ImageData;
            ```
            Alternatively, draw it onto a `<canvas>`.
        *   **Apply Overlays:** Using the `tracks` data (also from this `payload` for this camera):
            *   Iterate through `tracks`.
            *   Use `bbox_xyxy` to draw bounding boxes directly onto the just-displayed image (or its canvas overlay).
            *   Display `global_id` (or `track_id`) near the box.
            *   If `map_coords` are available, plot a point on your 2D map display.
    3.  Since the image and its metadata arrive in the same message, direct synchronization is achieved. There's no need to buffer metadata against a separate video stream.

### 4.3. REMOVED `media_available` Message

This message type is no longer sent by the backend. Frame images are now part of `tracking_update`.

### 4.4. REMOVED `batch_processing_complete` Message

This message type is no longer sent by the backend for frontend video synchronization purposes.

## 5. Managing Video Display (Simplified)

*   The concept of "video transitions" between sub-video files is removed from the frontend's concern.
*   The frontend continuously receives `tracking_update` messages, each containing a new frame image and its associated metadata.
*   The frontend simply decodes and displays each incoming frame image in the appropriate camera view, updating the overlays. This creates the video effect.

## 6. Task End

*   The task is complete when the `status_update` message indicates `status: "COMPLETED"`.
*   The WebSocket connection might be closed by the server, or you can close it.

## Summary of Frontend Responsibilities

1.  **Health Check:** Poll `/health` before anything else.
2.  **Initiate Task:** `POST` to `/api/v1/detection-processing-tasks/start`.
3.  **Connect WebSocket:** Establish and maintain the WebSocket connection.
4.  **Handle `tracking_update`:**
    *   For each camera:
        *   Decode `frame_image_base64` and display the image.
        *   Use `tracks` data from the same message to draw bounding boxes and other overlays on the displayed image.
        *   Update map display with `map_coords`.
5.  **Handle `status_update`:** Update UI with progress and status.
6.  **Error Handling:** Gracefully handle API errors, WebSocket disconnections.
