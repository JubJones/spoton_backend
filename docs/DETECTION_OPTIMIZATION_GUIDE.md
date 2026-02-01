# Detection Pipeline Optimization Summary

## Performance Result
**Before:** 2.6 FPS → **After:** 8-10 FPS (Effective: 24-30 FPS) — **3-4x faster**

---

## Key Optimizations

| # | Optimization | What It Does | Impact |
|---|--------------|--------------|--------|
| 1 | **TensorRT Export** | Convert PyTorch to optimized GPU engine (FP16, batch=16) | 6x faster inference |
| 2 | **Mega-Batch Pipeline** | Accumulate 4 frames/camera before single GPU call | 16 images per inference |
| 3 | **Frame Skip** | Process every Nth source frame | Linear FPS multiplier |
| 4 | **Parallel Tracking** | Run all camera trackers concurrently | Track 170ms → 50ms |
| 5 | **Stream Resize** | Resize frames to 640px max before JPEG encode | Smaller payload |
| 6 | **JPEG Quality** | Adjustable compression (1-100) | Smaller WS payload |
| 7 | **Input Size** | Detection at 480/640 instead of original | Faster inference |

---

## Quick Config (.env)

```env
# Batch Processing
BATCH_ACCUMULATION_SIZE=4   # 4 cams × 4 frames = batch=16
FRAME_SKIP=3                # Process every 3rd frame

# TensorRT
USE_TENSORRT=true           # Use .engine file

# Detection
DETECTION_IMGSZ=640         # 480=faster, 640=balanced

# Streaming Quality
FRAME_JPEG_QUALITY=85       # 1-100 (lower=smaller)

# Disable for Speed
ENABLE_DEBUG_REPROJECTION=false
REID_ENABLED=false
```

---

## Export TensorRT Engine

```bash
docker exec -it spoton_backend_service python /app/scripts/export_yolo_tensorrt.py
```

---

## Monitor Performance

```bash
docker logs spoton_backend_service 2>&1 | grep "SPEED_DEBUG"
```

**Good result:** `MEGA-BATCH ... BatchDet=140ms ... Track=50ms ... FPS=9.5 (Eff: 28.5)`
