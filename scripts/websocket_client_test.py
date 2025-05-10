# FILE: scripts/websocket_client_test.py
"""
Test client for SpotOn backend.
"""
import asyncio
import httpx
import websockets # type: ignore
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
    Logs detailed tracking information.
    """
    logger.info(f"Attempting to connect to WebSocket: {websocket_url}")

    parsed_ws_url = urlparse(websocket_url)
    scheme = "http" if parsed_ws_url.scheme == "ws" else "https"
    origin_value = f"{scheme}://{parsed_ws_url.netloc}" # e.g. "http://localhost:8000"
    
    logger.info(f"Client attempting WebSocket connection to: {websocket_url}")
    logger.info(f"Client will use origin parameter: '{origin_value}'")

    connect_kwargs: Dict[str, Any] = {
        "origin": origin_value,
        "ping_interval": 20,
        "ping_timeout": 20,
    }

    try:
        async with websockets.connect(websocket_url, **connect_kwargs) as websocket: # type: ignore
            logger.info(f"Successfully connected to WebSocket for task '{task_id}' at {websocket_url}")
            if hasattr(websocket, 'request_headers'):
                 logger.debug(f"  WebSocket Connection Request Headers (client-side perspective): {websocket.request_headers}") # type: ignore
            if hasattr(websocket, 'response_headers'):
                 logger.debug(f"  WebSocket Connection Response Headers (client-side perspective): {websocket.response_headers}") # type: ignore
            try:
                while True:
                    message_str = await websocket.recv()
                    try:
                        message = json.loads(message_str)
                        msg_type = message.get("type")
                        payload = message.get("payload", {})

                        if msg_type == "tracking_update":
                            cam_id = payload.get('camera_id')
                            frame_ts = payload.get('frame_timestamp')
                            tracking_data_list = payload.get('tracking_data', [])
                            logger.info(
                                f"[TASK {task_id}][TRACKING_UPDATE] Cam: {cam_id}, TS: {frame_ts}, Tracks: {len(tracking_data_list)}"
                            )
                            # Log details for each tracked person
                            for i, person_data in enumerate(tracking_data_list):
                                bbox_img = person_data.get('bbox_img', 'N/A')
                                # Format bbox for readability
                                if isinstance(bbox_img, list) and len(bbox_img) == 4:
                                    bbox_str = f"[{bbox_img[0]:.1f}, {bbox_img[1]:.1f}, {bbox_img[2]:.1f}, {bbox_img[3]:.1f}]"
                                else:
                                    bbox_str = str(bbox_img)

                                logger.info(
                                    f"  -> Person {i+1}: TrackID: {person_data.get('track_id', 'N/A')}, "
                                    f"GlobalID: {person_data.get('global_person_id', 'None')}, "
                                    f"BBox: {bbox_str}, "
                                    f"Conf: {person_data.get('confidence', 'N/A')}"
                                )

                        elif msg_type == "status_update":
                            logger.info(f"[TASK {task_id}][STATUS_UPDATE]")
                            logger.info(f"  Status: {payload.get('status')}")
                            progress = payload.get('progress')
                            if progress is not None:
                                logger.info(f"  Progress: {progress:.2%}")
                            logger.info(f"  Current Step: {payload.get('current_step')}")
                            if payload.get('details'):
                                logger.info(f"  Details: {payload.get('details')}")
                        else:
                            logger.warning(f"Received unknown message type or malformed JSON: {message_str[:200]}")

                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON message: {message_str[:200]}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

            except websockets.exceptions.ConnectionClosedOK: # type: ignore
                logger.info(f"WebSocket connection (task {task_id}) closed normally.")
            except websockets.exceptions.ConnectionClosedError as e: # type: ignore
                logger.error(f"WebSocket connection (task {task_id}) closed with error: {e}")
    
    except websockets.exceptions.InvalidStatusCode as e_status: # type: ignore
        logger.error(f"Failed to connect: Status {e_status.status_code}")
        if e_status.status_code == 403:
             logger.error(
                "A 403 (Forbidden) error. "
                f"Check server logs. Client used origin parameter: '{origin_value}'"
            )
    except TypeError as te: # Catch TypeErrors from connect() itself
        logger.error(f"TypeError during websockets.connect(): {te}", exc_info=True)
        logger.error(f"This might indicate an incompatibility with websockets=={getattr(websockets, '__version__', 'unknown')} "
                     f"and the arguments: {connect_kwargs}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused for WebSocket: {websocket_url}. Is the server running and accessible?")
    except Exception as e_outer: 
        logger.error(f"Outer error managing WebSocket connection: {e_outer}", exc_info=True)


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