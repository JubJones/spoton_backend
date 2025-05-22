"""
Test client for SpotOn backend.
"""
import asyncio
import httpx
import websockets
import json
import logging
import sys
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional, Any

# Configure basic logging for the client
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("websocket_client")

# --- Configuration ---
BACKEND_BASE_URL = "http://localhost:8000"
WEBSOCKET_BASE_URL = "ws://localhost:8000"
API_V1_PREFIX = "/api/v1"
HEALTH_CHECK_URL = f"{BACKEND_BASE_URL}/health"


async def check_backend_health(max_retries=10, delay_seconds=5): # Increased retries for slower startups
    """Polls the /health endpoint until the backend is healthy or retries are exhausted."""
    logger.info(f"Checking backend health at {HEALTH_CHECK_URL}...")
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(HEALTH_CHECK_URL, timeout=10.0) # Increased timeout
                response.raise_for_status()
                health_data = response.json()
                logger.info(f"Health check response (attempt {attempt + 1}): {health_data}")
                if health_data.get("status") == "healthy" and \
                   health_data.get("detector_model_loaded") is True and \
                   health_data.get("prototype_tracker_loaded (reid_model)") is True and \
                   health_data.get("homography_matrices_precomputed") is True:
                    logger.info("Backend is healthy and ready.")
                    return True
                else:
                    logger.info(f"Backend not fully ready (status: {health_data.get('status')}). Retrying in {delay_seconds}s...")
            except (httpx.HTTPStatusError, httpx.RequestError, json.JSONDecodeError) as e:
                logger.warning(f"Health check failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay_seconds}s...")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(delay_seconds)
        
        logger.error("Backend did not become healthy after multiple retries.")
        return False


async def start_processing_task(environment_id: str = "campus"):
    """
    Sends a request to start a processing task.
    """
    start_url = f"{BACKEND_BASE_URL}{API_V1_PREFIX}/processing-tasks/start"
    payload = {"environment_id": environment_id}
    logger.info(f"Requesting to start processing for environment: '{environment_id}' at {start_url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(start_url, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            task_id = data.get("task_id")
            ws_path = data.get("websocket_url")

            if not task_id or not ws_path:
                logger.error(f"Could not get task_id or websocket_url from response: {data}")
                return None, None

            if not ws_path.startswith("/"):
                ws_path = "/" + ws_path
            websocket_full_url = f"{WEBSOCKET_BASE_URL}{ws_path}"

            logger.info(f"Task '{task_id}' initiated. WebSocket URL: {websocket_full_url}")
            return task_id, websocket_full_url
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error starting task: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error starting task: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from start task: {e}")
        return None, None


async def listen_to_websocket(websocket_url: str, task_id: str):
    """
    Connects to the WebSocket and listens for messages.
    Logs detailed tracking information including map coordinates.
    """
    logger.info(f"Attempting to connect to WebSocket: {websocket_url}")

    parsed_ws_url = urlparse(websocket_url)
    origin_scheme = "http" if parsed_ws_url.scheme == "ws" else "https"
    origin_value = f"{origin_scheme}://{parsed_ws_url.netloc}"
    
    logger.info(f"Client attempting WebSocket connection to: {websocket_url}")
    logger.info(f"Client will use origin parameter for connect: '{origin_value}'")

    connect_kwargs: Dict[str, Any] = {
        "origin": origin_value,
        "ping_interval": 20,
        "ping_timeout": 20,
    }

    try:
        async with websockets.connect(websocket_url, **connect_kwargs) as websocket: # type: ignore
            logger.info(f"Successfully connected to WebSocket for task '{task_id}' at {websocket_url}")

            try:
                while True:
                    message_str = await websocket.recv()
                    try:
                        message = json.loads(message_str)
                        msg_type = message.get("type")
                        payload_data = message.get("payload", {})

                        if msg_type == "tracking_update":
                            global_frame_idx = payload_data.get("global_frame_index", "N/A") # Changed key
                            scene_id = payload_data.get("scene_id", "N/A")
                            ts_processed = payload_data.get("timestamp_processed_utc", "N/A")
                            cameras_data = payload_data.get("cameras", {})

                            logger.info(
                                f"[TASK {task_id}][TRACKING_UPDATE] GlobalFrame: {global_frame_idx}, Scene: {scene_id}, TS: {ts_processed}"
                            )
                            for cam_id, cam_content in cameras_data.items():
                                image_src = cam_content.get("image_source", "N/A")
                                tracking_data_list = cam_content.get('tracks', [])
                                logger.info(
                                    f"  Camera: {cam_id}, ImgSrc: {image_src}, Tracks: {len(tracking_data_list)}"
                                )
                                for i, person_data in enumerate(tracking_data_list):
                                    bbox_xyxy = person_data.get('bbox_xyxy', 'N/A')
                                    map_coords = person_data.get('map_coords', 'N/A')
                                    bbox_str = str(bbox_xyxy)
                                    if isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
                                        bbox_str = f"[{bbox_xyxy[0]:.1f}, {bbox_xyxy[1]:.1f}, {bbox_xyxy[2]:.1f}, {bbox_xyxy[3]:.1f}]"
                                    map_coords_str = str(map_coords)
                                    if isinstance(map_coords, list) and len(map_coords) == 2:
                                        map_coords_str = f"[{map_coords[0]:.1f}, {map_coords[1]:.1f}]"
                                    logger.info(
                                        f"    -> Person {i+1}: TrackID: {person_data.get('track_id', 'N/A')}, "
                                        f"GlobalID: {person_data.get('global_id', 'None')}, "
                                        f"BBox: {bbox_str}, Conf: {person_data.get('confidence', 'N/A')}, "
                                        f"ClassID: {person_data.get('class_id', 'N/A')}, "
                                        f"MapCoords: {map_coords_str}"
                                    )
                        elif msg_type == "status_update":
                            logger.info(f"[TASK {task_id}][STATUS_UPDATE]")
                            logger.info(f"  Status: {payload_data.get('status')}")
                            progress = payload_data.get('progress')
                            if progress is not None:
                                logger.info(f"  Progress: {progress:.2%}")
                            logger.info(f"  Current Step: {payload_data.get('current_step')}")
                            if payload_data.get('details'):
                                logger.info(f"  Details: {payload_data.get('details')}")
                        
                        elif msg_type == "media_available":
                            batch_idx = payload_data.get("sub_video_batch_index", "N/A")
                            media_urls_list = payload_data.get("media_urls", [])
                            logger.info(f"[TASK {task_id}][MEDIA_AVAILABLE] Sub-video batch index: {batch_idx}")
                            for entry in media_urls_list:
                                logger.info(
                                    f"  Cam: {entry.get('camera_id')}, File: {entry.get('sub_video_filename')}, "
                                    f"URL: {BACKEND_BASE_URL}{entry.get('url')}, " # Construct full URL
                                    f"StartGlobalFrame: {entry.get('start_global_frame_index')}, "
                                    f"NumFramesInSubVideo: {entry.get('num_frames_in_sub_video')}"
                                )
                        elif msg_type == "batch_processing_complete":
                            batch_idx = payload_data.get("sub_video_batch_index", "N/A")
                            logger.info(f"[TASK {task_id}][BATCH_PROCESSING_COMPLETE] Sub-video batch index: {batch_idx} finished.")
                        
                        else:
                            logger.warning(f"Received unknown message type ('{msg_type}') or malformed JSON: {message_str[:300]}")

                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON message: {message_str[:200]}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

            except websockets.exceptions.ConnectionClosedOK:
                logger.info(f"WebSocket connection (task {task_id}) closed normally.")
            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(f"WebSocket connection (task {task_id}) closed with error: {e}")
    
    except websockets.exceptions.InvalidStatusCode as e_status:
        logger.error(f"Failed to connect: WebSocket handshake failed with status {e_status.status_code}.")
        if e_status.status_code == 403:
             logger.error(
                "A 403 (Forbidden) error was received during WebSocket handshake. "
                f"Client used origin: '{origin_value}'"
            )
        if hasattr(e_status, 'headers') and e_status.headers:
            logger.error(f"Server response headers: {e_status.headers}")
            
    except TypeError as te:
        logger.error(f"TypeError during websockets.connect(): {te}", exc_info=True)
    except ConnectionRefusedError:
        logger.error(f"Connection refused for WebSocket: {websocket_url}. Is the server running and accessible?")
    except Exception as e_outer:
        logger.error(f"Outer error managing WebSocket connection for {websocket_url}: {e_outer}", exc_info=True)


async def main():
    """
    Main function to run the client.
    """
    environment_id_to_process = "campus"
    if len(sys.argv) > 1:
        environment_id_to_process = sys.argv[1]
        logger.info(f"Using environment_id from command line argument: '{environment_id_to_process}'")
    else:
        logger.info(f"Using default environment_id: '{environment_id_to_process}'")

    if not await check_backend_health():
        logger.error("Backend not healthy. Exiting client.")
        return

    task_id, ws_full_url = await start_processing_task(environment_id_to_process)

    if task_id and ws_full_url:
        await listen_to_websocket(ws_full_url, task_id)
    else:
        logger.error("Failed to start task or get WebSocket URL. Exiting.")

if __name__ == "__main__":
    try:
        _ws_version = getattr(websockets, "__version__", "unknown")
        logger.debug(f"Using websockets library version: {_ws_version}")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped by user (Ctrl+C).")