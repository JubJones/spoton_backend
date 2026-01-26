
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
video_paths = [
    "/app/videos/c09.mp4",
    "/app/videos/c12.mp4",
    "/app/videos/c13.mp4",
    "/app/videos/c16.mp4"
]
model_path = "/app/weights/yolo26m.engine"


# Workaround for 'AttributeError: Can't get attribute 'PatchedC3k2'
def fix_yolo_serialization():
    try:
        from ultralytics.nn.modules import block
        
        # Helper to safely set main attribute
        def safe_set_main(name, cls):
            if not hasattr(sys.modules['__main__'], name):
                setattr(sys.modules['__main__'], name, cls)

        # Fix C3k2
        if hasattr(block, 'C3k2'):
            safe_set_main('PatchedC3k2', block.C3k2)

        # Fix SPPF
        SPPF = getattr(block, 'SPPF', None)
        if SPPF:
            safe_set_main('PatchedSPPF', SPPF)
    except Exception as e:
        logger.warning(f"Could not apply serialization fix: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Starting Standalone YOLO PoC...")
    
    # Apply serialization fix needed for some custom YOLO weights
    fix_yolo_serialization()
    
    # OPTIMIZATION 1: CUDNN Benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("🚀 CUDNN Benchmark Enabled")
    
    logger.info(f"Loading YOLO model: {model_path}")
    # Load model
    try:
        model = YOLO(model_path, task='detect')
        
        # Warmup
        logger.info("Warming up model...")
        # Check if CUDA available for half precision
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # TensorRT/ONNX models don't support .to(), they are device specific or auto-handled
        if not model_path.endswith('.engine') and not model_path.endswith('.onnx'):
            model.to(device)
        
        # Dummy inference
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        # Pass device argument explicitly for exported models to avoid ambiguity
        dev_arg = 0 if device == 'cuda' else 'cpu'
        model.predict(dummy, half=(device=='cuda'), verbose=False, device=dev_arg)
        
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
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_paths[i]}")
            # Continue anyway, might just have black frames for that one
    
    # Profiler buckets
    prof_read = []
    prof_detect = []
    prof_draw = []
    prof_encode = []
    
    frame_count = 0
    t_last_log = time.time()
    
    # JPEG Compression Params (Quality 70 = Faster)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    
    # Target size for each quadrant
    target_w = 640
    target_h = 360  # 16:9 aspect ratio approximation
    
    while True:
        # --- SPEED PROFILING ---
        # 1. READ (All 4 streams)
        t0 = time.perf_counter()
        frames = []
        for cap in caps:
            if not cap.isOpened():
                # Push blank frame
                frames.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
                continue
                
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
                if not success: # Still failed?
                     frames.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
                     continue

            frames.append(frame)
            
        t_read = (time.perf_counter() - t0) * 1000
        
        frame_count += 1
        
        # 2. DETECT & DRAW Loop
        t1 = time.perf_counter()
        
        processed_frames = []
        
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0: # Should be handled above but double check
                processed_frames.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
                continue
                
            try:
                # Resize first to target quadrant size
                # Note: We resize BEFORE detection to keep it fast, 
                # but depending on original resolution we might want to detect on original then resize. 
                # For POC speed, resize first is better.
                resized_frame = cv2.resize(frame, (target_w, target_h))
                
                # Predict
                # classes=[0] -> Person only
                results = model.predict(
                    resized_frame, 
                    conf=0.5, 
                    classes=[0], 
                    half=True, 
                    verbose=False,
                    device=0 if torch.cuda.is_available() else 'cpu' 
                )
                
                # Draw
                # Plot results directly on the resized frame
                # .plot() returns BGR numpy array
                if results:
                    annotated_frame = results[0].plot(img=resized_frame)
                else:
                    annotated_frame = resized_frame
                
                # Add Camera Label
                cv2.putText(annotated_frame, f"CAM {i+1}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                           
                processed_frames.append(annotated_frame)

            except Exception as e:
                logger.error(f"Detection error on cam {i}: {e}")
                # Fallback to just resized frame
                resized = cv2.resize(frame, (target_w, target_h)) if frame.size > 0 else np.zeros((target_h, target_w, 3), dtype=np.uint8)
                processed_frames.append(resized)

        t_detect = (time.perf_counter() - t1) * 1000

        # 3. COMPOSE GRID & FPS
        t2 = time.perf_counter()
        
        # Grid:
        # [0] [1]
        # [2] [3]
        top_row = cv2.hconcat([processed_frames[0], processed_frames[1]])
        bot_row = cv2.hconcat([processed_frames[2], processed_frames[3]])
        composite = cv2.vconcat([top_row, bot_row])
        
        # Simple FPS Calc
        curr_time = time.time()
        fps = 1 / (curr_time - t_last_log) if (curr_time - t_last_log) > 0 else 0
        t_last_log = curr_time
        
        # Smoothed FPS for display
        cv2.putText(composite, f"TOTAL FPS: {fps:.1f}", (composite.shape[1] - 250, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                   
        t_draw = (time.perf_counter() - t2) * 1000

        # 4. ENCODE
        t3 = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', composite, encode_param)
        frame_bytes = buffer.tobytes()
        t_encode = (time.perf_counter() - t3) * 1000

        # Accumulate metrics
        prof_read.append(t_read)
        prof_detect.append(t_detect)
        prof_draw.append(t_draw)
        prof_encode.append(t_encode)
        
        # Log Speed every 20 frames
        if frame_count % 20 == 0:
            avg_read = sum(prof_read) / len(prof_read)
            avg_detect = sum(prof_detect) / len(prof_detect)
            avg_draw = sum(prof_draw) / len(prof_draw)
            avg_encode = sum(prof_encode) / len(prof_encode)
            
            logger.info(
                f"[SPEED] Avg 20 frames: Read={avg_read:.1f}ms | Det (4x)={avg_detect:.1f}ms | Grid={avg_draw:.1f}ms | Enc={avg_encode:.1f}ms"
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
