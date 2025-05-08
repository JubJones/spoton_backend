import cv2
import os
import asyncio
from typing import List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

async def extract_frames_from_video_to_disk(
    video_path: str,
    output_folder: str,
    frame_filename_prefix: str = "frame_",
    target_fps: Optional[int] = 23,
    jpeg_quality: int = 95
) -> Tuple[List[str], str]:
    """
    Extracts frames from a video file and saves them to disk.
    This function is synchronous and should be run in a thread pool (e.g., using asyncio.to_thread).

    Args:
        video_path: Path to the video file.
        output_folder: Directory to save extracted frames.
        frame_filename_prefix: Prefix for saved frame filenames.
        target_fps: Desired frames per second to extract. If None or 0, extracts all frames.
        jpeg_quality: Quality for saving JPEG frames (0-100).

    Returns:
        A tuple containing:
        - A list of full paths to the saved frame images.
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
        skip_interval = round(video_fps_actual / target_fps)
        if skip_interval <= 0: 
             skip_interval = 1
    elif target_fps is None or target_fps <= 0: # Extract all frames
        effective_target_fps = video_fps_actual # For reporting purposes
        skip_interval = 1
    
    extracted_frame_paths: List[str] = []
    frame_count_read = 0
    saved_frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
                
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
    finally:        
        cap.release()

    status_msg = (
        f"Extracted {saved_frame_count} frames from '{Path(video_path).name}' "
        f"(Original FPS: {video_fps_actual:.2f}, Target Extraction FPS: {effective_target_fps}, Interval: {skip_interval}). "
        f"Frames saved to: {output_folder}"
    )
    logger.info(status_msg)
    return extracted_frame_paths, status_msg