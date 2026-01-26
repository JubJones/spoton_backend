
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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoC")
logger.setLevel(logging.INFO)

# Suppress other loggers
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("app.models.yolo_detector").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("speed_debug_yolo").setLevel(logging.WARNING)

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
    
    # Profiler buckets
    prof_read = []
    prof_detect = []
    prof_draw = []
    prof_encode = []

    while True:
        # --- SPEED PROFILING ---
        t_start = time.perf_counter()

        # 1. READ
        t0 = time.perf_counter()
        success, frame = cap.read()
        t_read = (time.perf_counter() - t0) * 1000
        
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        total_frame_count += 1

        # 2. DETECT
        t1 = time.perf_counter()
        try:
            detections = await detector.detect(frame)
        except Exception as e:
            logger.error(f"Detection error: {e}")
            detections = []
        t_detect = (time.perf_counter() - t1) * 1000

        # 3. DRAW
        t2 = time.perf_counter()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw FPS and Frame Count
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = frame_count_fps / elapsed_time
            frame_count_fps = 0
            fps_start_time = time.time()
        info_text = f"Frame: {total_frame_count} | FPS: {fps:.2f}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        t_draw = (time.perf_counter() - t2) * 1000

        # 4. ENCODE
        t3 = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        t_encode = (time.perf_counter() - t3) * 1000

        # Accumulate metrics
        prof_read.append(t_read)
        prof_detect.append(t_detect)
        prof_draw.append(t_draw)
        prof_encode.append(t_encode)

        frame_count_fps += 1
        
        # Log every 20 frames
        if total_frame_count % 20 == 0:
            avg_read = sum(prof_read) / len(prof_read)
            avg_detect = sum(prof_detect) / len(prof_detect)
            avg_draw = sum(prof_draw) / len(prof_draw)
            avg_encode = sum(prof_encode) / len(prof_encode)
            
            logger.info(
                f"[SPEED] Avg over 20 frames: "
                f"Read={avg_read:.1f}ms | "
                f"Detect={avg_detect:.1f}ms | "
                f"Draw={avg_draw:.1f}ms | "
                f"Encode={avg_encode:.1f}ms | "
                f"Total={avg_read+avg_detect+avg_draw+avg_encode:.1f}ms"
            )
            # Reset
            prof_read = []
            prof_detect = []
            prof_draw = []
            prof_encode = []

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Control loop speed slightly to prevent 100% CPU usage
        await asyncio.sleep(0.001)

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
