# RT-DETR Detection Pipeline Requirements

## 1. Project Overview

Create a new detection-enabled video streaming endpoint that builds upon the existing raw video endpoint (`/api/v1/raw-processing-tasks`) by integrating RT-DETR person detection. This system will replace the current complex multi-stage processing pipeline with a streamlined RT-DETR-based approach.

## 2. Core Requirements

### 2.1 Endpoint Structure
- **Base Path**: `/api/v1/detection-processing-tasks`
- **Pattern**: Clone the existing raw endpoint structure but add RT-DETR detection processing
- **Replacement Target**: This will replace the current `/api/v1/processing-tasks` endpoint flow

### 2.2 RT-DETR Integration
```python
from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("path/to/bus.jpg")
```

### 2.3 Processing Pipeline
1. **Download video frames** from S3 storage (same as raw endpoint)
2. **Run RT-DETR detection** on each frame
3. **Annotate images** with detection bounding boxes and confidence scores
4. **Extract detection data** (bboxes, confidence, class, etc.)
5. **Stream both annotated images and detection data** to frontend via WebSocket

## 3. Technical Specifications

### 3.1 Model Configuration
- **Model**: RT-DETR-l (Large variant for better accuracy)
- **Weights**: `rtdetr-l.pt` (COCO-pretrained)
- **Input**: Video frames (JPEG images from S3)
- **Processing**: Real-time inference per frame
- **Classes**: Person detection (class_id: 0 in COCO)

### 3.2 Detection Processing
- **Confidence Threshold**: 0.5 (configurable)
- **NMS Threshold**: 0.45 (configurable)
- **Input Resolution**: 640x640 (RT-DETR standard)
- **Annotation**: Draw bounding boxes with confidence scores on images
- **Output Format**: Base64 encoded annotated images + structured detection data

## 4. API Endpoints

### 4.1 Required Endpoints
```
GET    /api/v1/detection-processing-tasks/environments
POST   /api/v1/detection-processing-tasks/start
GET    /api/v1/detection-processing-tasks/{task_id}/status
GET    /api/v1/detection-processing-tasks/{task_id}
GET    /api/v1/detection-processing-tasks
DELETE /api/v1/detection-processing-tasks/{task_id}/stop
DELETE /api/v1/detection-processing-tasks/environment/{environment_id}/cleanup
```

### 4.2 WebSocket Endpoint
```
/ws/detection-tracking/{task_id}
```

## 5. Response Schema Design

### 5.1 WebSocket Message Format
```json
{
  "message_type": "detection_update",
  "task_id": "uuid-string",
  "timestamp": "2025-01-06T10:30:00.123Z",
  "camera_id": "c09",
  "frame_number": 145,
  "frame_data": {
    "original_image_b64": "base64-encoded-original-image",
    "annotated_image_b64": "base64-encoded-annotated-image",
    "image_dimensions": {
      "width": 1920,
      "height": 1080
    }
  },
  "detection_data": {
    "detections": [
      {
        "detection_id": "det_001",
        "class_name": "person",
        "class_id": 0,
        "confidence": 0.87,
        "bbox": {
          "x1": 100.5,
          "y1": 150.2,
          "x2": 200.8,
          "y2": 400.9,
          "width": 100.3,
          "height": 250.7,
          "center_x": 150.65,
          "center_y": 275.55
        }
      }
    ],
    "detection_count": 1,
    "processing_time_ms": 45.2
  },
  "future_pipeline_data": {
    "tracking_data": null,
    "reid_data": null,
    "homography_data": null,
    "mapping_coordinates": null
  }
}
```

### 5.2 Status Update Messages
```json
{
  "message_type": "status_update",
  "task_id": "uuid-string",
  "timestamp": "2025-01-06T10:30:00.123Z",
  "status": "PROCESSING",
  "progress": 45.2,
  "current_step": "Running RT-DETR detection on frame 145/320",
  "camera_processing": {
    "c09": {"status": "PROCESSING", "frame": 145},
    "c12": {"status": "PROCESSING", "frame": 143},
    "c13": {"status": "PROCESSING", "frame": 147},
    "c16": {"status": "PROCESSING", "frame": 144}
  }
}
```

## 6. Future Pipeline Integration

### 6.1 Extensible Schema Design
The response schema includes placeholder fields for future pipeline components:

#### 6.2 Tracking Data (Future)
```json
"tracking_data": {
  "track_id": "track_001",
  "track_age": 15,
  "track_status": "confirmed",
  "velocity": {"x": 2.3, "y": -1.1},
  "trajectory": [
    {"x": 148.1, "y": 275.0, "timestamp": "2025-01-06T10:29:58.000Z"},
    {"x": 150.65, "y": 275.55, "timestamp": "2025-01-06T10:30:00.123Z"}
  ]
}
```

#### 6.3 Re-ID Data (Future)
```json
"reid_data": {
  "global_person_id": "person_global_123",
  "reid_confidence": 0.92,
  "feature_embedding": "base64-encoded-clip-features",
  "appearance_features": {
    "dominant_colors": ["blue", "black"],
    "clothing_type": ["shirt", "pants"],
    "approximate_height": 175.2
  }
}
```

#### 6.4 Homography & Mapping (Future)
```json
"homography_data": {
  "camera_matrix": "3x3-homography-matrix",
  "calibration_confidence": 0.95
},
"mapping_coordinates": {
  "world_x": 15.2,
  "world_y": 35.8,
  "floor_level": 1,
  "coordinate_system": "factory_map_v1"
}
```

## 7. Implementation Requirements

### 7.1 Service Architecture
- **DetectionVideoService**: New service class similar to `RawVideoService`
- **RTDETRDetector**: RT-DETR model wrapper with inference methods
- **DetectionAnnotator**: Image annotation utility for drawing bboxes
- **DetectionWebSocketManager**: WebSocket communication handler

### 7.2 Configuration
```python
# Environment variables
RTDETR_MODEL_PATH = "weights/rtdetr-l.pt"
RTDETR_CONFIDENCE_THRESHOLD = 0.5
RTDETR_NMS_THRESHOLD = 0.45
RTDETR_INPUT_SIZE = 640
DETECTION_ANNOTATION_ENABLED = True
DETECTION_SAVE_ORIGINAL_FRAMES = True
```

### 7.3 Error Handling
- Model loading failures
- GPU/CPU fallback scenarios
- Frame processing timeouts
- WebSocket connection management
- S3 download failures

## 8. Performance Requirements

### 8.1 Throughput Targets
- **FPS**: 15-20 frames per second per camera
- **Latency**: <100ms detection processing per frame
- **Memory**: <2GB GPU memory usage
- **CPU**: Efficient CPU fallback for development

### 8.2 Scalability
- Support 4 cameras simultaneously
- Handle 1-hour video sessions
- Graceful degradation under load
- Automatic model optimization

## 9. Testing Requirements

### 9.1 Unit Tests
- RT-DETR model inference accuracy
- Bounding box annotation correctness
- WebSocket message formatting
- Error handling scenarios

### 9.2 Integration Tests
- End-to-end pipeline testing
- Multi-camera processing
- WebSocket communication
- Performance benchmarking

## 10. Deployment Considerations

### 10.1 Model Assets
- Download `rtdetr-l.pt` to `weights/` directory
- Model version management
- GPU driver compatibility
- CUDA toolkit requirements

### 10.2 Environment Support
- Docker container updates
- GPU acceleration setup
- Development environment configuration
- Production deployment guidelines