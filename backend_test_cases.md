# Comprehensive Backend Test Cases

This document outlines the test cases for the SpotOn Backend system, covering REST APIs, WebSocket connections, MJPEG video streaming, and system health checks.

## 1. Environment Management APIs (`/api/v1`)
These endpoints manage physical environments, cameras, zones, and user preferences.

| Test Case ID | Endpoint | Method | Description | Steps to Test | Expected Result |
|---|---|---|---|---|---|
| `ENV-001` | `/` | GET | List environments | Send GET request to `/api/v1/` | Returns 200 OK with a list of active environments. |
| `ENV-002` | `/{environment_id}` | GET | Get environment metadata | Send GET request with valid `environment_id` | Returns 200 OK with environment details. |
| `ENV-003` | `/{environment_id}` | GET | Get environment metadata (Invalid ID) | Send GET request with non-existent ID | Returns 404 Not Found. |
| `ENV-004` | `/{environment_id}/cameras` | GET | List cameras in environment | Send GET to `/{id}/cameras` | Returns 200 OK with camera configurations. |
| `ENV-005` | `/{environment_id}/zones` | GET | List zones in environment | Send GET to `/{id}/zones` | Returns 200 OK with defined detection zones. |
| `ENV-006` | `/{environment_id}/sessions` | POST | Create environment session | Send POST with valid session payload | Returns 201 Created with session details. |
| `ENV-007` | `/preferences/{user_id}` | GET/PUT | User preferences | Update preferences via PUT, retrieve via GET | Returns 200 OK and matching updated data. |

## 2. Detection Processing Tasks (`/api/v1/detection-processing-tasks`)
These endpoints manage the asynchronous video processing and AI detection jobs.

| Test Case ID | Endpoint | Method | Description | Steps to Test | Expected Result |
|---|---|---|---|---|---|
| `TSK-001` | `/` | POST | Create detection task | Send POST with valid video source & config | Returns 201 Created with `task_id` and initial status. |
| `TSK-002` | `/` | POST | Create detection task (Invalid payload) | Send POST with missing required fields | Returns 422 Unprocessable Entity. |
| `TSK-003` | `/{task_id}` | GET | Get task details | Send GET with valid `task_id` | Returns 200 OK with task configuration and status. |
| `TSK-004` | `/{task_id}/status` | GET | Poll task status | Send GET with valid `task_id` | Returns 200 OK with current progress, state, and metrics. |
| `TSK-005` | `/{task_id}/results` | GET | Fetch task results | Send GET after task completion | Returns 200 OK with final tracking and detection data. |
| `TSK-006` | `/{task_id}` | DELETE | Cancel/Delete task | Send DELETE with valid active `task_id` | Returns 200 OK, task stops processing and resources are freed. |

## 3. Analytics APIs (`/api/v1/analytics`)
These endpoints provide aggregated insights and real-time metrics.

| Test Case ID | Endpoint | Method | Description | Steps to Test | Expected Result |
|---|---|---|---|---|---|
| `ANL-001` | `/dashboard` | GET | Get dashboard summary | Send GET to `/dashboard` | Returns 200 OK with high-level system metrics. |
| `ANL-002` | `/real-time/metrics` | GET | Fetch real-time metrics | Send GET to `/real-time/metrics` | Returns 200 OK with current operational stats. |
| `ANL-003` | `/behavior/analyze` | POST | Analyze behavior | Send POST with target identity/timeframe | Returns 200 OK with behavioral insights. |
| `ANL-004` | `/historical/summary` | POST | Historical data query | Send POST with date range and filters | Returns 200 OK with aggregated historical data. |
| `ANL-005` | `/system/statistics` | GET | Get system analytics stats | Send GET to `/system/statistics` | Returns 200 OK with engine performance stats. |

## 4. Playback Controls (`/api/v1/controls`)
Controls for managing the playback state of active processing streams.

| Test Case ID | Endpoint | Method | Description | Steps to Test | Expected Result |
|---|---|---|---|---|---|
| `PLY-001` | `/{task_id}/pause` | POST | Pause processing | Send POST to executing task | Returns 200 OK, stream halts processing. |
| `PLY-002` | `/{task_id}/resume` | POST | Resume processing | Send POST to paused task | Returns 200 OK, stream resumes processing. |
| `PLY-003` | `/{task_id}/seek` | POST | Seek in stream | Send POST with target timestamp/frame | Returns 200 OK, internal pointer moves to target. |

## 5. MJPEG Video Streaming (`/api/v1/stream`)
Real-time visual feedback of detection bounding boxes.

| Test Case ID | Endpoint | Method | Description | Steps to Test | Expected Result |
|---|---|---|---|---|---|
| `STR-001` | `/{task_id}/{camera_id}`| GET | Stream MJPEG Feed | Open connection via browser or HTTP client | Returns HTTP 200 with `Content-Type: multipart/x-mixed-replace`. Receives continuous JPEG frames. |
| `STR-002` | `/{task_id}/{camera_id}`| GET | Stream Disconnection | Client closes connection midway | Backend cleanly terminates generator without memory leaks or zombie threads. |
| `STR-003` | `/{invalid}/{camera_id}`| GET | Stream Invalid Task | Request stream for non-existent task | Returns 404 Not Found before initiating stream response. |

## 6. WebSocket Connections (`/ws`)
Real-time bi-directional telemetry and event streaming.

| Test Case ID | Endpoint | Type | Description | Steps to Test | Expected Result |
|---|---|---|---|---|---|
| `WS-001` | `/tracking/{task_id}` | WS | Tracking Telemetry | Connect WS client to endpoint | Server accepts connection and begins pushing JSON bounding boxes and tracker IDs matching video frame rate. |
| `WS-002` | `/frames/{task_id}` | WS | Frame metadata | Connect WS client | Receives metadata (timestamps, resolutions, FPS) for processed frames. |
| `WS-003` | `/system` | WS | Global System Events | Connect WS client | Receives global alerts, resource warnings, or broadcast events. |
| `WS-004` | `/analytics/{task_id}`| WS | Live Analytics Feed | Connect WS client | Receives periodic updates for zone counts, line crossings, and active person counts. |
| `WS-005` | `/*` | WS | Connection Drop | Connect tracking WS, then abruptly close | Backend detects client disconnect, removes from ConnectionManager, and avoids `AnyIO` closed socket errors. |
| `WS-006` | `/focus/{task_id}` | WS | Client message sending | Connect and send target ID JSON | Backend processes incoming message and adjusts tracking target priority/focus. |

## 7. System Health & Probes (`/health`, `/`)
Endpoints used by orchestrators (Kubernetes/Docker) to verify uptime.

| Test Case ID | Endpoint | Method | Description | Steps to Test | Expected Result |
|---|---|---|---|---|---|
| `HLT-001` | `/` | GET | Root Welcome | Send GET to `/` | Returns 200 OK with app name and version. |
| `HLT-002` | `/health` | GET | Basic Health Check | Send GET to `/health` | Returns 200 OK indicating `status: healthy` or `degraded`. |
| `HLT-003` | `/health/deep` | GET | Deep Health Check | Send GET to `/health/deep` | Returns 200 OK with individual component statuses (DB, Redis, Model loaded state). |
| `HLT-004` | `/health/live` | GET | Liveness Probe | Send GET to `/health/live` | Returns 200 OK instantly (fails only if event loop is blocked). |
| `HLT-005` | `/ws/health` | GET | WS Module Health | Send GET to `/ws/health` | Returns 200 OK, ensuring WebSocket router is operational. |

## Recommended Tools for Execution
- **REST APIs**: `pytest` + `httpx` (AsyncClient) / Postman
- **WebSockets**: `pytest-asyncio` + `websockets` library / Postman WebSocket client
- **MJPEG Streams**: Script reading multipart boundaries or a headless browser test (`Playwright`).
