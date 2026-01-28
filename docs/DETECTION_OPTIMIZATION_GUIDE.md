# Detection Pipeline FPS Optimization Guide

This document summarizes all optimizations implemented to maximize FPS for the multi-camera YOLO detection pipeline.

---

## Performance Summary

| Stage | Before | After | Improvement |
|-------|--------|-------|-------------|
| Overall FPS | 2.6 | 8-10 (Eff: 24-30) | **3-4x faster** |
| BatchDet (16 imgs) | 254ms | 140-170ms | **~40% faster** |
| Per-image inference | 63ms | 9-10ms | **6x faster** |

---

## 1. TensorRT Engine Export

### What We Did
Exported YOLO model to TensorRT with FP16 and batch=16 for multi-camera inference.

### Export Configuration
```python
# scripts/export_yolo_tensorrt.py
export_to_tensorrt(
    model_path="/app/weights/yolo26m.pt",
    imgsz=640,        # Input size (480 for faster, 640 for accuracy)
    batch_size=16,    # Max batch size (4 cams × 4 frames)
    half=True,        # FP16 precision for RTX GPUs
    workspace_gb=1,   # TensorRT optimization workspace
    dynamic=True      # Supports batch 1-16
)
```

### Export Command
```bash
docker exec -it spoton_backend_service python /app/scripts/export_yolo_tensorrt.py
```

### Key Points
- **batch_size=16**: Process up to 16 images in single GPU call
- **half=True**: FP16 halves memory and improves throughput
- **dynamic=True**: Accepts any batch size from 1 to 16
- **workspace_gb**: Scratch memory for TensorRT optimizer (only during export)

---

## 2. Mega-Batch Inference Pipeline

### What We Did
Modified detection loop to accumulate multiple frames from each camera before running batch inference.

### Configuration (.env)
```env
BATCH_ACCUMULATION_SIZE=4  # Frames per camera (4 cams × 4 frames = 16)
```

### How It Works
```
Before: Read 4 frames (1/cam) → Inference (batch=4) → Process → Repeat
After:  Read 16 frames (4/cam) → Single Inference (batch=16) → Process All
```

### Log Format
```
[SPEED_DEBUG] MEGA-BATCH Frames=60-63 | BatchSize=16 | BatchDet=140ms | FPS=9.5 (Eff: 28.5)
```

---

## 3. Frame Skip

### What We Did
Skip processing of intermediate source frames to reduce GPU load.

### Configuration (.env)
```env
FRAME_SKIP=3  # Process every 3rd frame (skip 2)
FRAME_SKIP=5  # Process every 5th frame (skip 4)
```

### Impact
- `FRAME_SKIP=3`: 3x effective FPS multiplier
- `FRAME_SKIP=5`: 5x effective FPS multiplier

---

## 4. Parallel Encoding

### What We Did
MJPEG frame encoding runs concurrently for all cameras using asyncio.gather.

### Code Location
`app/services/detection_video_service.py` - Encoding phase uses `asyncio.to_thread` with gather.

---

## 5. Docker Compose Environment Fix

### Issue
The `poc/docker-compose.gpu.yml` was not loading `.env` file settings.

### Fix Applied
Added `env_file` directive to load settings from parent `.env`:

```yaml
# poc/docker-compose.gpu.yml
services:
  yolo_poc_gpu:
    env_file:
      - ../.env
    environment:
      - PYTHONPATH=/app
      # ... other vars
```

---

## 6. Configuration Reference

### Performance Settings (.env)
```env
# TensorRT
USE_TENSORRT=true

# Batch Processing  
BATCH_ACCUMULATION_SIZE=4    # Frames per camera per batch

# Frame Skip
FRAME_SKIP=3                 # Process every Nth source frame

# Detection
DETECTION_IMGSZ=640          # 480=faster, 640=balanced, 832=accuracy
DETECTION_HALF=true          # FP16 for GPU

# Encoding
FRAME_JPEG_QUALITY=85        # 1-100 (minimal speed impact)

# Features (disable for speed)
ENABLE_DEBUG_REPROJECTION=false
ENABLE_ENHANCED_VISUALIZATION=false
```

### Recommended Settings by GPU VRAM

| GPU VRAM | batch_size | workspace_gb | BATCH_ACCUMULATION_SIZE |
|----------|------------|--------------|-------------------------|
| 8GB | 16 | 1 | 4 |
| 12GB | 16 | 2 | 4 |
| 24GB | 32 | 4 | 8 |

---

## 7. Monitoring

### Speed Debug Logs
```bash
docker logs spoton_backend_service 2>&1 | grep "SPEED_DEBUG"
```

### Key Metrics
- **BatchDet**: GPU inference time (target: <200ms for batch=16)
- **Track**: Tracker update time (target: <50ms)
- **WsSend**: WebSocket transmission (network-dependent)
- **Enc**: JPEG encoding time (target: <30ms)
- **FPS (Eff)**: Raw FPS × FRAME_SKIP = effective FPS

### Example Good Performance
```
MEGA-BATCH Frames=60-63 | BatchDet=140ms | Track=70ms | FPS=9.5 (Eff: 28.5)
```

---

## 8. Potential Further Optimizations

| Option | Impact | Trade-off |
|--------|--------|-----------|
| Use yolo26n (nano) | ~50% faster inference | Lower accuracy |
| Use imgsz=480 | ~30% faster inference | Smaller objects missed |
| Increase FRAME_SKIP | Linear reduction | Less smooth tracking |
| Disable REID | ~5-10% faster | No cross-camera ID |
