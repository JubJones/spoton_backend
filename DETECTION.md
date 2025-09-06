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
2. **Run RT-DETR detection** on each frame (batched processing)
3. **Multi-object tracking** using BoxMOT trackers per camera
4. **Re-ID feature extraction** for new/refreshed tracks (batched)
5. **Cross-camera association** using cosine similarity and EMA gallery updates
6. **Homography projection** to map coordinates (if calibrated)
7. **Handoff trigger detection** based on exit rules and quadrant overlap
8. **Annotate images** with detection bboxes, track IDs, and global IDs
9. **Stream enhanced tracking data** with re-ID and mapping info via WebSocket

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
- **Annotation**: Draw bounding boxes with track IDs, global IDs, and confidence scores
- **Output Format**: Base64 encoded annotated images + structured tracking data

### 3.3 Re-ID Processing
- **Feature Model**: CLIP Market1501-trained model for person re-identification
- **Feature Dimensions**: 512-dimensional embeddings (CLIP standard)
- **Similarity Metric**: Cosine similarity on L2-normalized embeddings
- **Similarity Threshold**: 0.75 (configurable)
- **Gallery Management**: Main gallery + lost track gallery with EMA updates
- **Refresh Interval**: 30 frames (configurable)
- **Conflict Resolution**: Best match wins, second-pass matching for reverted tracks

### 3.4 Homography Mapping
- **Calibration Method**: RANSAC-based homography computation
- **Input Points**: Minimum 4 corresponding image-map point pairs
- **Projection Target**: Foot point (center-bottom of bounding box)
- **Coordinate System**: Bird's eye view map in meters
- **Error Handling**: Graceful fallback when projection fails

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
        },
        "track_id": 15,
        "global_id": 123,
        "map_coords": {
          "map_x": 15.23,
          "map_y": 35.87
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
  "global_id": 123,
  "reid_confidence": 0.92,
  "feature_vector": "base64-encoded-clip-embedding-512d",
  "feature_model": "clip_market1501",
  "gallery_type": "main",
  "similarity_score": 0.87,
  "reid_state": {
    "last_reid_frame": 145,
    "reid_refresh_due": false,
    "handoff_triggered": true,
    "association_type": "matched"
  },
  "gallery_metadata": {
    "ema_alpha": 0.1,
    "last_seen_camera": "c09",
    "first_seen_frame": 120,
    "track_age": 25,
    "embedding_dimension": 512
  }
}
```

#### 6.4 Homography & Mapping (Future)
```json
"homography_data": {
  "matrix": [
    [1.2, 0.1, -45.6],
    [0.05, 1.1, -12.3],
    [0.0001, 0.0002, 1.0]
  ],
  "matrix_available": true,
  "calibration_points": {
    "image_points": [[100, 200], [300, 150], [250, 400], [450, 380]],
    "map_points": [[10.5, 20.2], [30.1, 18.7], [25.3, 40.8], [45.2, 38.5]]
  }
},
"mapping_coordinates": {
  "map_x": 15.23,
  "map_y": 35.87,
  "projection_successful": true,
  "foot_point": {
    "image_x": 150.65,
    "image_y": 275.55
  },
  "coordinate_system": "bev_map_meters"
}
```

## 7. Implementation Requirements

### 7.1 Service Architecture
- **DetectionVideoService**: New service class similar to `RawVideoService`
- **RTDETRDetector**: RT-DETR model wrapper with inference methods
- **DetectionAnnotator**: Image annotation utility for drawing bboxes
- **DetectionWebSocketManager**: WebSocket communication handler
- **ReIDStateManager**: Manages global ID galleries and track-to-global mappings
- **ReIDFeatureExtractor**: Handles feature extraction decisions and batched processing
- **ReIDAssociationService**: Performs similarity matching and conflict resolution
- **HomographyService**: Manages coordinate transformations using calibrated matrices

### 7.2 Configuration
```python
# Environment variables
RTDETR_MODEL_PATH = "weights/rtdetr-l.pt"
RTDETR_CONFIDENCE_THRESHOLD = 0.5
RTDETR_NMS_THRESHOLD = 0.45
RTDETR_INPUT_SIZE = 640
DETECTION_ANNOTATION_ENABLED = True
DETECTION_SAVE_ORIGINAL_FRAMES = True

# Re-ID Configuration
REID_MODEL_PATH = "weights/clip_market1501.pt"
REID_MODEL_TYPE = "clip"
REID_SIMILARITY_THRESHOLD = 0.75
REID_REFRESH_INTERVAL_FRAMES = 30
REID_GALLERY_EMA_ALPHA = 0.1
REID_LOST_TRACK_BUFFER_FRAMES = 100

# Homography Configuration
HOMOGRAPHY_POINTS_DIR = "homography_points/"
ENABLE_BEV_MAP_PROJECTION = True
HOMOGRAPHY_RANSAC_THRESHOLD = 5.0

# Handoff Configuration
MIN_BBOX_OVERLAP_RATIO_IN_QUADRANT = 0.5
CAMERA_HANDOFF_RULES_ENABLED = True
POSSIBLE_CAMERA_OVERLAPS = [("c09", "c12"), ("c12", "c13"), ("c13", "c16")]
```

### 7.3 Error Handling
- Model loading failures (Detection and Re-ID models)
- GPU/CPU fallback scenarios
- Frame processing timeouts
- WebSocket connection management
- S3 download failures
- Re-ID feature extraction failures
- Homography matrix computation errors
- Gallery state corruption recovery
- Track association conflicts
- Invalid embedding normalization
- Handoff trigger validation failures

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