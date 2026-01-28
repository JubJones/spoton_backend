# Detection Pipeline Optimization Guide

This document summarizes all optimizations implemented to maximize FPS for the multi-camera YOLO detection pipeline.

---

## Performance Summary

| Stage | Before | After | Improvement |
|-------|--------|-------|-------------|
| Overall FPS | 2.6 | 8-10 (Eff: 24-29) | **3-4x faster** |
| BatchDet (4 imgs) | 254ms | 35-40ms | **6x faster** |
| Per-image inference | 63ms | 9-10ms | **6x faster** |

---

## 1. TensorRT Engine Optimization

### What We Did
Exported YOLO model to TensorRT with optimized batch size for multi-camera inference.

### Configuration
```python
# scripts/export_yolo_tensorrt.py
export_to_tensorrt(
    model_path="/app/weights/yolo26m.pt",
    imgsz=640,
    batch_size=16,    # Supports up to 16 images per batch
    half=True,        # FP16 precision for RTX GPUs
    workspace_gb=1,   # TensorRT optimization workspace
    dynamic=True      # Dynamic batching (1-16 images)
)
```

### Key Points
- **batch_size=16**: Allows processing 4 cameras × 4 frames = 16 images in one GPU call
- **half=True**: FP16 reduces memory and increases throughput on RTX GPUs
- **dynamic=True**: Engine accepts any batch size from 1 to 16
- **workspace_gb**: Scratch memory for TensorRT optimizer (only used during export)

### Export Command
```bash
docker exec -it spoton_backend_service python /app/scripts/export_yolo_tensorrt.py
```

---

## 2. Batch Inference Pipeline

### What We Did
Modified the detection loop to accumulate multiple frames before running batch inference.

### Configuration
```env
# .env
BATCH_ACCUMULATION_SIZE=4  # Frames per camera before batch inference
```

### How It Works
```
Before: Read 4 frames (1/cam) → Inference → Process → Repeat
After:  Read 16 frames (4/cam) → Single Batch Inference → Process All
```

### Code Location
- `app/services/detection_video_service.py` - `_process_frames_simple_detection()` method

### Log Format
```
[SPEED_DEBUG] MEGA-BATCH Frames=60-63 | BatchSize=16 | BatchDet=140ms | FPS=9.5 (Eff: 28.5)
```

---

## 3. GPU-Level Optimizations

### 3.1 Multi-Stream Pipeline
Three CUDA streams for overlapped execution:
- **Preprocess stream** (priority -1): Data transfer to GPU
- **Inference stream** (priority -1): Model execution
- **Postprocess stream** (priority 0): Results transfer to CPU

### 3.2 CUDA Graph Capture
Captures and replays inference graph for reduced kernel launch overhead:
```python
# Warmup captures the graph
self._cuda_graph_manager.capture(self.model, warmup_input, ...)

# Inference replays the captured graph
with torch.cuda.graph(self._cuda_graph_manager.graph):
    results = self.model(input_tensor)
```

### 3.3 Pinned Memory Pool
Pre-allocated page-locked memory for faster CPU↔GPU transfers:
```python
self._pinned_memory_pool = PinnedMemoryPool(
    buffer_size=imgsz * imgsz * 3,  # RGB image
    num_buffers=8
)
```

### 3.4 Pre-allocated Device Tensors
Reuse GPU tensors to avoid allocation overhead:
```python
self._preallocated_input = torch.empty(
    (max_batch, 3, imgsz, imgsz),
    dtype=torch.float16,
    device='cuda'
)
```

### Code Location
- `app/models/yolo_detector.py` - `YOLODetector` class
- `app/models/rtdetr_detector.py` - `RTDETRDetector` class

---

## 4. Frame Skip

### What We Did
Skip processing of intermediate frames to reduce load.

### Configuration
```env
# .env
FRAME_SKIP=2  # Process every 2nd frame (skip 1)
FRAME_SKIP=3  # Process every 3rd frame (skip 2)
```

### Impact
- `FRAME_SKIP=2`: 2x effective FPS with same compute
- `FRAME_SKIP=3`: 3x effective FPS with same compute

---

## 5. Parallel Encoding

### What We Did
MJPEG frame encoding runs concurrently for all cameras using `asyncio.gather`.

### Code
```python
async def _encode_frame(cid, frm):
    return cid, await asyncio.to_thread(
        self.annotator.frame_to_jpeg_bytes, frm
    )

encoding_tasks = [_encode_frame(cid, data[0]) for cid, data in frame_camera_data.items()]
results = await asyncio.gather(*encoding_tasks)
```

---

## 6. Configuration Reference

### Performance Settings (.env)
```env
# TensorRT
USE_TENSORRT=true
YOLO_MODEL_PATH_TENSORRT=/app/weights/yolo26m.engine

# Batch Processing
BATCH_ACCUMULATION_SIZE=4    # Frames per camera per batch
FRAME_SKIP=2                 # Skip every Nth frame

# YOLO Settings
YOLO_CONFIDENCE_THRESHOLD=0.3
YOLO_INPUT_SIZE=640

# Encoding Quality
FRAME_JPEG_QUALITY=90        # Lower = faster encode, smaller size

# Tracking
TRACK_BUFFER_SIZE=30         # Smaller = faster
```

### Recommended Settings by GPU

| GPU | batch_size | workspace_gb | BATCH_ACCUMULATION_SIZE |
|-----|------------|--------------|-------------------------|
| RTX 4060 (8GB) | 16 | 1-2 | 4 |
| RTX 3060 (12GB) | 16 | 2-4 | 4 |
| RTX 4090 (24GB) | 32 | 4-8 | 8 |

---

## 7. Monitoring & Debugging

### Speed Debug Logs
```bash
docker logs spoton_backend_service 2>&1 | grep "SPEED_DEBUG"
```

### Key Metrics to Watch
- `BatchDet`: GPU inference time (should be <200ms for batch=16)
- `Track`: Tracker update time (should be <50ms typically)
- `WsSend`: WebSocket transmission (should be <100ms)
- `Enc`: JPEG encoding time (should be <30ms)

### Log Examples
```
# Good performance
[SPEED_DEBUG] MEGA-BATCH ... BatchDet=140ms Track=70ms FPS=9.5

# Problem: inference too slow
[SPEED_DEBUG] MEGA-BATCH ... BatchDet=500ms  # Check if TensorRT engine loaded

# Problem: tracking spike
[SPEED_DEBUG] _process_tracking | TrackerUpdate=103.6ms  # GC or contention
```

---

## 8. Troubleshooting

### Engine Not Loading
```
# Check if .engine file exists
docker exec spoton_backend_service ls -la /app/weights/*.engine

# Re-export if needed
docker exec spoton_backend_service python /app/scripts/export_yolo_tensorrt.py
```

### BatchSize=4 Instead of 16
```
# Verify BATCH_ACCUMULATION_SIZE is set
docker exec spoton_backend_service env | grep BATCH

# Restart container after config change
docker-compose -f docker-compose.gpu.yml restart
```

### OOM During Export
```python
# Reduce batch size and workspace in export script
batch_size=8,    # Instead of 16
workspace_gb=1,  # Instead of 2
```

---

## 9. Phase 2 Optimizations (WsSend + Track)

### Implemented
- [x] **Parallel tracker updates**: All 16 frame-camera pairs tracked concurrently with `asyncio.gather`
- [x] **Lower JPEG quality**: `FRAME_JPEG_QUALITY=75` (was 90) - faster encode, smaller payload
- [x] **Reduced track buffer**: `TRACK_BUFFER_SIZE=20` (was 30) - less memory/CPU
- [x] **Stream resize**: Already resizing to max 640px width before encoding

### Configuration
```env
# .env
FRAME_JPEG_QUALITY=75        # 1-100, lower = faster
TRACK_BUFFER_SIZE=20         # Frames of history per track
STREAM_RESIZE_FACTOR=1.0     # Optional: 0.5 = half size
```

### Expected Impact
| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Track | 70-170ms (sequential) | 20-40ms (parallel) | 3-4x faster |
| Enc | ~30ms | ~20ms | 30% faster |
| Total MEGA-BATCH | 400-500ms | 250-350ms | ~40% faster |

---

## 10. Future Optimizations

### Potential
- [ ] INT8 quantization (requires calibration dataset)
- [ ] Multi-GPU inference for >8 cameras
- [ ] Direct GPU→network transfer (GPU Direct RDMA)
- [ ] WebSocket binary mode instead of JSON
