"""
Test client for SpotOn backend.

This script performs the following actions:
1. Sends a POST request to start a processing task for a given environment.
2. Parses the response to get the task_id and WebSocket URL.
3. Connects to the WebSocket endpoint.
4. Listens for incoming messages and prints them.
   - For 'tracking_update' messages, it prints the details, including the frame_path.
   - For 'status_update' messages, it prints the status.
5. The script will run indefinitely listening to WebSocket messages until manually stopped (Ctrl+C).

Note on displaying images:
The backend sends `frame_path`, which is a path *on the server*. To display these images
on the client, the client would need:
  a) The actual image data (bytes) sent over the WebSocket (can be inefficient).
  b) A URL from which the client can fetch the image, requiring the backend to serve these
     frame images over HTTP.
This script currently only prints the `frame_path`.

To run:
python websocket_client_test.py [environment_id]
Example:
python websocket_client_test.py campus
"""
import asyncio
import httpx
import websockets
import json
import logging
import sys

# Configure basic logging for the client
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("websocket_client")

# --- Configuration ---
# These should match your backend setup.
BACKEND_BASE_URL = "http://localhost:8000"
WEBSOCKET_BASE_URL = "ws://localhost:8000"
API_V1_PREFIX = "/api/v1"  # Match your app.core.config.settings.API_V1_PREFIX


async def start_processing_task(environment_id: str = "campus"):
    """
    Sends a request to start a processing task.

    Args:
        environment_id: The environment ID to process.

    Returns:
        A tuple (task_id, websocket_full_url) or (None, None) on failure.
    """
    start_url = f"{BACKEND_BASE_URL}{API_V1_PREFIX}/processing-tasks/start"
    payload = {"environment_id": environment_id}
    logger.info(f"Requesting to start processing for environment: '{environment_id}' at {start_url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(start_url, json=payload, timeout=30.0)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            task_id = data.get("task_id")
            ws_path = data.get("websocket_url") # This is relative, e.g., /ws/tracking/{task_id}

            if not task_id or not ws_path:
                logger.error(f"Could not get task_id or websocket_url from response: {data}")
                return None, None

            # Construct the full WebSocket URL
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
    """
    logger.info(f"Attempting to connect to WebSocket: {websocket_url}")
    try:
        async with websockets.connect(websocket_url, ping_interval=20, ping_timeout=20) as websocket:
            logger.info(f"Successfully connected to WebSocket for task '{task_id}' at {websocket_url}")
            try:
                while True:
                    message_str = await websocket.recv()
                    try:
                        message = json.loads(message_str)
                        msg_type = message.get("type")
                        payload = message.get("payload", {})

                        if msg_type == "tracking_update":
                            logger.info(f"[TASK {task_id}][TRACKING_UPDATE]")
                            logger.info(f"  Camera ID: {payload.get('camera_id')}")
                            logger.info(f"  Timestamp: {payload.get('frame_timestamp')}")
                            logger.info(f"  Frame Path (on server): {payload.get('frame_path')}")
                            tracking_data = payload.get('tracking_data', [])
                            logger.info(f"  Tracked Objects: {len(tracking_data)}")
                            for i, person_data in enumerate(tracking_data[:3]): # Log first 3 tracks
                                logger.info(f"    Person {i+1}: "
                                            f"TrackID={person_data.get('track_id')}, "
                                            f"GlobalID={person_data.get('global_person_id', 'N/A')}, "
                                            f"BBox={person_data.get('bbox_img')}")
                            if len(tracking_data) > 3:
                                logger.info(f"    ... and {len(tracking_data) - 3} more tracks.")


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
                        logger.warning(f"Received non-JSON message: {message_str[:200]}") # Log first 200 chars
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

            except websockets.exceptions.ConnectionClosedOK:
                logger.info(f"WebSocket connection (task {task_id}) closed normally.")
            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(f"WebSocket connection (task {task_id}) closed with error: {e}")

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"Failed to connect to WebSocket: Status {e.status_code}")
        logger.error("This often means an issue with the WebSocket endpoint path, or an "
                     "authentication/authorization problem not handled by a typical CORS fix "
                     "(e.g., if the server explicitly rejected the connection).")
    except ConnectionRefusedError:
        logger.error(f"Connection refused for WebSocket: {websocket_url}. Is the server running and accessible?")
    except Exception as e:
        logger.error(f"Failed to connect or listen to WebSocket: {e}", exc_info=True)


async def main():
    """
    Main function to run the client.
    """
    environment_id_to_process = "campus" # Default environment
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
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped by user (Ctrl+C).")