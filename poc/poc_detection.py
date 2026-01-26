
import asyncio
import cv2
import numpy as np
import logging
import sys
import os
import time
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
from contextlib import asynccontextmanager
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PoC")
logger.setLevel(logging.INFO)

# Global variables
model = None
video_path = "/app/videos/campus/c01/sub_video_01.mp4"
model_path = "weights/yolo26m.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Starting Standalone YOLO PoC...")
    
    # OPTIMIZATION 1: CUDNN Benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("🚀 CUDNN Benchmark Enabled")
    
    logger.info(f"Loading YOLO model: {model_path}")
    # Load model
    try:
        model = YOLO(model_path)
        
        # Warmup
        logger.info("Warming up model...")
        # Check if CUDA available for half precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Dummy inference
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        model.predict(dummy, half=(device=='cuda'), verbose=False)
        
        logger.info(f"✅ Model Loaded on {device.upper()}")
        if device == 'cuda':
             logger.info("   - FP16: ENABLED")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

async def frame_generator():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return

    # Profiler buckets
    prof_read = []
    prof_detect = []
    prof_draw = []
    prof_encode = []
    
    frame_count = 0
    t_last_log = time.time()
    
    # JPEG Compression Params (Quality 70 = Faster)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    
    while True:
        # --- SPEED PROFILING ---
        # 1. READ
        t0 = time.perf_counter()
        success, frame = cap.read()
        t_read = (time.perf_counter() - t0) * 1000
        
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # 2. DETECT
        t1 = time.perf_counter()
        try:
            # Resize
            h, w = frame.shape[:2]
            target_w = 640
            scale = target_w / w
            target_h = int(h * scale)
            resized_frame = cv2.resize(frame, (target_w, target_h))
            
            # Predict
            # classes=[0] -> Person only (Optimization)
            # half=True -> FP16 (Optimization)
            # verbose=False -> No stdout noise (Optimization)
            results = model.predict(
                resized_frame, 
                conf=0.5, 
                classes=[0], 
                half=True, 
                verbose=False,
                device=0 if torch.cuda.is_available() else 'cpu' 
            )
            
            # Parse results
            detections = []
            if len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    # Move to CPU once
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    # cls = boxes.cls.cpu().numpy() # We know it's person (0)
                    
                    for i in range(len(xyxy)):
                        detections.append((xyxy[i], conf[i]))
                        
            # Draw (Coordinate scaling)
            # We draw on the ORIGINAL frame, so we scale the boxes back up
            # Alternatively: Draw on resized frame and send that (faster encode), 
            # but user might want high res view? Let's stick to scaling coordinates back for now.
            for (box, score) in detections:
                x1, y1, x2, y2 = box
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {score:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            logger.error(f"Detection error: {e}")
        t_detect = (time.perf_counter() - t1) * 1000

        # 3. DRAW EXTRAS (Info)
        t2 = time.perf_counter()
        # Simple FPS Calc
        if frame_count % 10 == 0:
             curr_time = time.time()
             fps = 10 / (curr_time - t_last_log)
             t_last_log = curr_time
             cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        t_draw = (time.perf_counter() - t2) * 1000 # This includes the loop drawing above actually, splitting hairs for PoC

        # 4. ENCODE
        # OPTIMIZATION: Lower JPEG quality
        t3 = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()
        t_encode = (time.perf_counter() - t3) * 1000

        # Accumulate metrics
        prof_read.append(t_read)
        prof_detect.append(t_detect)
        prof_draw.append(t_draw)
        prof_encode.append(t_encode)
        
        # Log Speed
        if frame_count % 20 == 0:
            avg_read = sum(prof_read) / len(prof_read)
            avg_detect = sum(prof_detect) / len(prof_detect)
            avg_draw = sum(prof_draw) / len(prof_draw)
            avg_encode = sum(prof_encode) / len(prof_encode)
            
            logger.info(
                f"[SPEED] Avg 20 frames: Read={avg_read:.1f}ms | Det={avg_detect:.1f}ms | Enc={avg_encode:.1f}ms | Total={avg_read+avg_detect+avg_encode:.1f}ms"
            )
            prof_read = []
            prof_detect = []
            prof_draw = []
            prof_encode = []

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Tiny sleep to yield control
        await asyncio.sleep(0.001)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head><title>Standalone YOLO PoC</title></head>
        <body>
            <h1>Standalone YOLO Detection (Optimized)</h1>
            <img src="/video_feed" width="800" />
        </body>
    </html>
    """

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    logger.info("🚀 Standalone PoC Starting")
    logger.info("👉 Access via Browser:")
    logger.info("   - If on the same PC:  http://localhost:8000")
    logger.info("   - If remote:          http://<YOUR_PC_IP>:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
