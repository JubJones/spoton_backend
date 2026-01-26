
import asyncio
import cv2
import numpy as np
import logging
import sys
import os
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
from contextlib import asynccontextmanager

# Ensure app is in path
sys.path.append("/app")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoC")

from app.models.yolo_detector import YOLODetector

# Global variables
detector = None
video_path = "/app/videos/campus/c01/sub_video_01.mp4"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    # Startup
    logger.info("Starting YOLO PoC with Web UI...")
    model_name = "weights/yolo26m.pt"
    logger.info(f"Initializing YOLODetector with model: {model_name}")
    detector = YOLODetector(model_name=model_name)
    await detector.load_model()
    logger.info("Warming up model...")
    await detector.warmup()
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

async def frame_generator():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return

    fps_start_time = time.time()
    frame_count_fps = 0
    total_frame_count = 0
    fps = 0

    while True:
        success, frame = cap.read()
        if not success:
            # Restart video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        total_frame_count += 1

        # Resize for faster processing if needed (optional, keeping original size for now, or resize to generic size)
        # frame = cv2.resize(frame, (640, 480))

        # Perform detection
        try:
            detections = await detector.detect(frame)
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            logger.error(f"Detection error: {e}")

        # Calculate FPS
        frame_count_fps += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = frame_count_fps / elapsed_time
            frame_count_fps = 0
            fps_start_time = time.time()

        # Draw FPS and Frame Count
        info_text = f"Frame: {total_frame_count} | FPS: {fps:.2f}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Control loop speed slightly to prevent 100% CPU usage if processing is super fast (unlikely with CPU detection)
        await asyncio.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head>
            <title>YOLO PoC Detection</title>
        </head>
        <body>
            <h1>YOLO Detection Feed (Campus)</h1>
            <img src="/video_feed" width="800" />
        </body>
    </html>
    """

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    logger.info("🚀 PoC Server Starting!")
    logger.info("👉 Access via Browser:")
    logger.info("   - If on the same PC:  http://localhost:8000")
    logger.info("   - If remote:          http://<YOUR_PC_IP>:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
