import cv2
import os
import asyncio
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import numpy as np # Import numpy
import base64 # For image encoding

logger = logging.getLogger(__name__)

# Synchronous function for frame extraction - run in thread pool
def extract_frames_from_video_to_disk(
    video_path: str,
    output_folder: str,
    frame_filename_prefix: str = "frame_",
    target_fps: Optional[int] = 23,
    jpeg_quality: int = 95
) -> Tuple[List[str], str]:
    """
    Extracts frames from a video file and saves them to disk.
    This function is synchronous and should be run in a thread pool.

    Args:
        video_path: Path to the video file.
        output_folder: Directory to save extracted frames.
        frame_filename_prefix: Prefix for saved frame filenames.
        target_fps: Desired frames per second to extract. If None or 0, extracts all frames.
        jpeg_quality: Quality for saving JPEG frames (0-100).

    Returns:
        A tuple containing:
        - A list of full paths to the saved frame images, sorted numerically.
        - A status message string.
    """
    if not os.path.exists(video_path):
        msg = f"Video file not found: {video_path}"
        logger.error(msg)
        return [], msg

    try:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Could not create output directory {output_folder}: {e}"
        logger.error(msg)
        return [], msg

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Could not open video file: {video_path}"
        logger.error(msg)
        return [], msg

    video_fps_actual = cap.get(cv2.CAP_PROP_FPS)
    if video_fps_actual <= 0:
        logger.warning(f"Video FPS is invalid ({video_fps_actual}) for {video_path}. Assuming 25 FPS for frame skipping.")
        video_fps_actual = 25.0

    skip_interval = 1
    effective_target_fps = target_fps
    if target_fps is not None and target_fps > 0 and video_fps_actual > target_fps:
        skip_interval = max(1, round(video_fps_actual / target_fps)) # Ensure interval is at least 1
    elif target_fps is None or target_fps <= 0: # Extract all frames
        effective_target_fps = video_fps_actual # For reporting purposes
        skip_interval = 1

    extracted_frame_paths: List[str] = []
    frame_count_read = 0
    saved_frame_count = 0

    logger.info(f"Starting frame extraction from '{Path(video_path).name}' at ~{effective_target_fps} FPS (Interval: {skip_interval})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if this frame should be saved based on the skip interval
            if frame_count_read % skip_interval == 0:
                frame_filename = f"{frame_filename_prefix}{saved_frame_count:06d}.jpg"
                full_frame_path = os.path.join(output_folder, frame_filename)

                success = cv2.imwrite(full_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if success:
                    extracted_frame_paths.append(full_frame_path)
                    saved_frame_count += 1
                else:
                    logger.warning(f"Failed to write frame {frame_filename} for video {video_path}")

            frame_count_read += 1
            # Log progress occasionally
            if frame_count_read % 100 == 0:
                logger.debug(f"Read {frame_count_read} frames, saved {saved_frame_count} frames from '{Path(video_path).name}'...")

    finally:
        cap.release()

    status_msg = (
        f"Extracted {saved_frame_count} frames from '{Path(video_path).name}' "
        f"(Orig FPS: {video_fps_actual:.2f}, Target FPS: ~{effective_target_fps}, Interval: {skip_interval}). "
        f"Saved to: {output_folder}"
    )
    logger.info(status_msg)

    # Sort paths numerically before returning
    extracted_frame_paths.sort()

    return extracted_frame_paths, status_msg


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image (from OpenCV) to RGB."""
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 2: # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image # Return as-is if not 3-channel BGR or grayscale


async def encode_frame_to_base64(
    frame_np: np.ndarray,
    format_ext: str = ".jpg",
    jpeg_quality: int = 75
) -> Optional[str]:
    """
    Encodes a NumPy frame array to a base64 string.

    Args:
        frame_np: The frame image as a NumPy array (BGR format from OpenCV).
        format_ext: The desired image format extension (e.g., ".jpg", ".png").
        jpeg_quality: Quality for JPEG encoding (0-100), ignored for PNG.

    Returns:
        A base64 encoded string of the image, or None if encoding fails.
    """
    if frame_np is None or frame_np.size == 0:
        logger.warning("Attempted to encode an empty frame.")
        return None

    encode_params = []
    if format_ext.lower() == ".jpg" or format_ext.lower() == ".jpeg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif format_ext.lower() == ".png":
        # PNG_COMPRESSION is 0-9, higher is more compression but slower.
        # Default is often around 3-6. Let's use a moderate default.
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        logger.warning(f"Unsupported image format for encoding: {format_ext}. Defaulting to JPEG.")
        format_ext = ".jpg"
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    try:
        # cv2.imencode is synchronous, run in thread pool for async context
        success, encoded_image_bytes = await asyncio.to_thread(
            cv2.imencode, format_ext, frame_np, encode_params
        )
        if not success:
            logger.error(f"Failed to encode frame to {format_ext.upper()}.")
            return None

        base64_encoded_string = base64.b64encode(encoded_image_bytes).decode('utf-8')
        return base64_encoded_string
    except Exception as e:
        logger.error(f"Error during frame encoding to base64: {e}", exc_info=True)
        return None