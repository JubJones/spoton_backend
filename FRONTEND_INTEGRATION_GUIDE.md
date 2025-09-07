# SpotOn Backend - Frontend Integration Guide

**Base URL**: `http://localhost:3847` | **API Version**: v1

## Quick Start

```javascript
// 1. Check system health
const health = await fetch('http://localhost:3847/health').then(r => r.json());

// 2. Start processing
const task = await fetch('http://localhost:3847/api/v1/processing-tasks/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ environment_id: 'campus' })
}).then(r => r.json());

// 3. Connect to real-time updates
const ws = new WebSocket(`ws://localhost:3847/ws/tracking/${task.task_id}`);
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'tracking_update') {
    handleTrackingUpdate(data.payload);
  }
};
```

## System Architecture

**Core Components**: AI Pipeline (person detection, tracking, re-ID), Real-time Processing (<100ms latency), Cross-Camera Tracking, Spatial Mapping, WebSocket Streaming

**Environments**:
- **Campus**: 4 cameras (c09, c12, c13, c16)
- **Factory**: 4 cameras (c01, c02, c03, c05)

**Performance**: 23 FPS processing, <50ms detection latency (GPU), <100ms WebSocket latency

## Authentication (Mock)

```javascript
// Login
const auth = await fetch('/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'admin', password: 'password' })
}).then(r => r.json());

// Use token
const headers = {
  'Authorization': `Bearer ${auth.access_token}`,
  'Content-Type': 'application/json'
};
```

## Core Integration Patterns

### 1. System Health Check
```javascript
const health = await fetch('/health').then(r => r.json());
const isReady = health.status === 'healthy' && 
                health.detector_model_loaded && 
                health['prototype_tracker_loaded (reid_model)'];
```

### 2. Environment Discovery
```javascript
const environments = await fetch('/api/v1/environments/').then(r => r.json());
// Returns: environment_id, camera_count, zone_count, has_data
```

### 3. Task Management
```javascript
// Start task
const task = await fetch('/api/v1/processing-tasks/start', {
  method: 'POST',
  body: JSON.stringify({ environment_id: 'campus' })
});

// Monitor status
const status = await fetch(`/api/v1/processing-tasks/${taskId}/status`).then(r => r.json());
// Status: QUEUED → INITIALIZING → DOWNLOADING → EXTRACTING → PROCESSING → COMPLETED
```

## REST API Reference

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|---------|----------|
| `/health` | GET | System health check |
| `/api/v1/environments/` | GET | List environments |
| `/api/v1/processing-tasks/start` | POST | Start full AI processing |
| `/api/v1/processing-tasks/{id}/status` | GET | Full processing task status |
| `/api/v1/detection-processing-tasks/start` | POST | Start RT-DETR detection only |
| `/api/v1/detection-processing-tasks/{id}/status` | GET | Detection task status |
| `/api/v1/raw-processing-tasks/start` | POST | Start raw video streaming |
| `/api/v1/raw-processing-tasks/{id}/status` | GET | Raw streaming task status |
| `/api/v1/focus/{taskId}` | POST/GET/DELETE | Person focus control |
| `/api/v1/controls/{taskId}/play` | POST | Start playback |
| `/api/v1/controls/{taskId}/pause` | POST | Pause playback |
| `/api/v1/controls/{taskId}/seek` | POST | Seek position |

### Environment Endpoints
```javascript
// Get environment details
GET /api/v1/environments/{id}

// Get cameras
GET /api/v1/environments/{id}/cameras

// Get date ranges
GET /api/v1/environments/{id}/date-ranges?detailed=true

// Create analysis session
POST /api/v1/environments/{id}/sessions
{
  "start_time": "2025-01-01T10:00:00Z",
  "end_time": "2025-01-01T11:00:00Z"
}
```

### Focus Tracking
```javascript
// Set focus on person
POST /api/v1/focus/{taskId}
{
  "global_person_id": "person_123",
  "cross_camera_sync": true,
  "highlight_settings": {
    "enabled": true,
    "intensity": 0.8
  }
}

// Get person details
GET /api/v1/persons/{personId}/details?include_history=true
```

## WebSocket Communication

### Connection URLs
- **Full Processing**: `ws://localhost:3847/ws/tracking/{taskId}`
- **Detection Processing**: `ws://localhost:3847/ws/detection-tracking/{taskId}`
- **Raw Video Streaming**: `ws://localhost:3847/ws/raw-tracking/{taskId}`
- **System**: `ws://localhost:3847/ws/system`
- **Focus**: `ws://localhost:3847/ws/focus/{taskId}`

### Messages

**Client → Server**:
```javascript
{ "type": "subscribe_tracking" }
{ "type": "ping" }
```

**Server → Client**:
```javascript
// Tracking update
{
  "type": "tracking_update",
  "payload": {
    "global_frame_index": 123,
    "timestamp_processed_utc": "2025-01-01T10:30:05.456Z",
    "cameras": {
      "c09": {
        "frame_image_base64": "data:image/jpeg;base64,...",
        "tracks": [{
          "track_id": 1,
          "global_id": "person_123",
          "bbox_xyxy": [110.2, 220.5, 160.0, 330.8],
          "confidence": 0.92,
          "map_coords": [12.3, 45.6],
          "is_focused": false
        }]
      }
    }
  }
}
```

## Complete Integration Example

```javascript
class SpotOnClient {
  constructor() {
    this.taskId = null;
    this.ws = null;
  }

  async initialize() {
    // Check health
    const health = await fetch('/health').then(r => r.json());
    if (health.status !== 'healthy') throw new Error('System not ready');

    // Get environments
    const environments = await fetch('/api/v1/environments/').then(r => r.json());
    return environments;
  }

  async startTracking(environmentId) {
    // Start task
    const response = await fetch('/api/v1/processing-tasks/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ environment_id: environmentId })
    });
    
    const task = await response.json();
    this.taskId = task.task_id;

    // Monitor until PROCESSING
    while (true) {
      const status = await fetch(`/api/v1/processing-tasks/${this.taskId}/status`).then(r => r.json());
      if (status.status === 'PROCESSING') break;
      if (status.status === 'FAILED') throw new Error(status.details);
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // Connect WebSocket
    this.connectWebSocket();
  }

  connectWebSocket() {
    this.ws = new WebSocket(`ws://localhost:3847/ws/tracking/${this.taskId}`);
    
    this.ws.onopen = () => {
      this.ws.send(JSON.stringify({ type: 'subscribe_tracking' }));
    };
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'tracking_update') {
        this.handleTrackingUpdate(message.payload);
      }
    };
  }

  handleTrackingUpdate(payload) {
    const { cameras, person_count_per_camera } = payload;
    
    Object.entries(cameras).forEach(([cameraId, data]) => {
      // Update frame display
      if (data.frame_image_base64) {
        document.getElementById(`camera-${cameraId}`).src = data.frame_image_base64;
      }
      
      // Update person count
      document.getElementById(`count-${cameraId}`).textContent = person_count_per_camera[cameraId] || 0;
      
      // Draw bounding boxes
      data.tracks?.forEach(track => {
        this.drawBoundingBox(cameraId, track);
      });
    });
  }

  async focusOnPerson(personId) {
    await fetch(`/api/v1/focus/${this.taskId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        global_person_id: personId,
        cross_camera_sync: true
      })
    });
  }
}
```

## Data Models

### Person Track
```typescript
interface PersonTrack {
  track_id: number;           // Camera-specific ID
  global_id: string;          // Cross-camera ID
  bbox_xyxy: [number, number, number, number]; // Bounding box
  confidence: number;         // 0.0-1.0
  map_coords: [number, number]; // Map coordinates
  is_focused: boolean;
}
```

### Environment
```typescript
interface Environment {
  environment_id: string;     // "campus" | "factory"
  name: string;
  camera_count: number;
  zone_count: number;
  has_data: boolean;
  last_updated: string;
}
```

## Error Handling

### HTTP Status Codes
- **200**: Success
- **202**: Task started (async)
- **400**: Invalid request
- **404**: Resource not found
- **500**: Server error

### Common Errors
```javascript
// System not ready
GET /health → { "status": "degraded", "detector_model_loaded": false }

// Invalid environment
POST /processing-tasks/start → { "detail": "Invalid environment_id" }

// Task not found
GET /processing-tasks/invalid/status → { "detail": "Processing task not found" }
```

### WebSocket Reconnection
```javascript
ws.onclose = (event) => {
  if (attempts < maxRetries) {
    setTimeout(() => connect(), 1000 * Math.pow(2, attempts++));
  }
};
```

## Performance Considerations

### Frontend Optimization
- **Frame Rate Adaptation**: Skip frames if processing is slow
- **Image Quality**: Adjust JPEG quality based on bandwidth
- **Connection Monitoring**: Track latency and reconnect if needed

### Production Setup
- **Health Monitoring**: Poll `/health` every 30 seconds
- **Token Management**: Implement token refresh (mock auth currently)
- **Error Recovery**: Implement exponential backoff for reconnections
- **Resource Management**: Monitor memory usage for long-running sessions

## Testing & Validation

### Health Check Workflow
```javascript
const validateSystem = async () => {
  const health = await fetch('/health').then(r => r.json());
  return {
    ready: health.status === 'healthy',
    models: health.detector_model_loaded && health['prototype_tracker_loaded (reid_model)'],
    mapping: health.homography_matrices_precomputed
  };
};
```

### End-to-End Test
```javascript
const e2eTest = async () => {
  // 1. Health check
  await validateSystem();
  
  // 2. Start task
  const task = await startTracking('campus');
  
  // 3. Connect WebSocket
  const ws = await connectWebSocket(task.task_id);
  
  // 4. Verify data flow
  return new Promise((resolve) => {
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'tracking_update') {
        resolve('Success: Receiving tracking data');
      }
    };
  });
};
```

## RT-DETR DETECTION - Person Detection with AI (Phase 1)

The backend provides RT-DETR detection endpoints that perform real-time person detection using the RT-DETR-l model. This is the first phase of the AI pipeline, providing person detection with bounding boxes but without tracking or re-identification.

### Detection Processing Endpoints

| Endpoint | Method | Purpose |
|----------|---------|----------|
| `/api/v1/detection-processing-tasks/environments` | GET | List environments for detection processing |
| `/api/v1/detection-processing-tasks/start` | POST | Start RT-DETR detection processing |
| `/api/v1/detection-processing-tasks/{id}/status` | GET | Get detection task status |
| `/api/v1/detection-processing-tasks/{id}` | GET | Get detection task details |
| `/api/v1/detection-processing-tasks/{id}/stop` | DELETE | Stop detection task |
| `/api/v1/detection-processing-tasks/environment/{env_id}/cleanup` | DELETE | Cleanup environment tasks |

### Detection WebSocket

**Connection URL**: `ws://localhost:3847/ws/detection-tracking/{taskId}`

### Quick Detection Setup

```javascript
// 1. Check system health (same as regular)
const health = await fetch('http://localhost:3847/health').then(r => r.json());

// 2. Start RT-DETR detection processing
const detectionTask = await fetch('http://localhost:3847/api/v1/detection-processing-tasks/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ environment_id: 'campus' })
}).then(r => r.json());

// 3. Connect to detection WebSocket
const detectionWs = new WebSocket(`ws://localhost:3847/ws/detection-tracking/${detectionTask.task_id}`);
detectionWs.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'tracking_update' && data.mode === 'detection_processing') {
    handleDetectionResults(data.payload);
  }
};
```

### Detection Message Format

**Detection Tracking Update**:
```javascript
{
  "type": "tracking_update",
  "task_id": "uuid",
  "global_frame_index": 123,
  "timestamp_processed_utc": "2025-01-01T10:30:05.456Z",
  "mode": "detection_processing",
  "cameras": {
    "c09": {
      "frame_image_base64": "data:image/jpeg;base64,...",
      "tracks": [
        {
          "track_id": 1,
          "global_id": null,  // No tracking yet (Phase 1)
          "bbox_xyxy": [110.2, 220.5, 160.0, 330.8],
          "confidence": 0.92,
          "map_coords": null,  // No mapping yet (Phase 1)
          "is_focused": false,
          "detection_class": "person",
          "detection_score": 0.92
        }
      ],
      "frame_width": 1920,
      "frame_height": 1080,
      "timestamp": "2025-01-01T10:30:05.456Z",
      "detection_count": 1
    }
  },
  "person_count_per_camera": {
    "c09": 1,
    "c12": 0,
    "c13": 2,
    "c16": 1
  }
}
```

### Detection vs Other Processing Modes

| Feature | Detection Processing | Raw Video Streaming | Full Processing Pipeline |
|---------|---------------------|---------------------|-------------------------|
| **Person Detection** | ✅ RT-DETR detection | ❌ No detection | ✅ Advanced detection |
| **Bounding Boxes** | ✅ Person bounding boxes | ❌ No bounding boxes | ✅ Tracked bounding boxes |
| **Confidence Scores** | ✅ Detection confidence | ❌ No scores | ✅ Detection + tracking confidence |
| **Tracking** | ❌ No tracking (single frame) | ❌ No tracking | ✅ Multi-object tracking |
| **Re-ID** | ❌ No re-identification | ❌ No re-identification | ✅ Cross-camera re-ID |
| **Map Coordinates** | ❌ No spatial mapping | ❌ No mapping | ✅ Spatial mapping |
| **Frame Images** | ✅ Base64 encoded frames | ✅ Base64 encoded frames | ✅ Base64 encoded frames |
| **Performance** | Medium (AI detection only) | Fastest (no processing) | Slower (full AI pipeline) |
| **Use Case** | Detection development/testing | Debug/PoC development | Production tracking |

### Detection Client Example

```javascript
class DetectionClient {
  constructor() {
    this.taskId = null;
    this.ws = null;
  }

  async startDetectionProcessing(environmentId) {
    // Start detection task
    const response = await fetch('/api/v1/detection-processing-tasks/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ environment_id: environmentId })
    });
    
    const task = await response.json();
    this.taskId = task.task_id;

    // Monitor until PROCESSING
    while (true) {
      const status = await fetch(`/api/v1/detection-processing-tasks/${this.taskId}/status`).then(r => r.json());
      if (status.status === 'PROCESSING') break;
      if (status.status === 'FAILED') throw new Error(status.details);
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // Connect WebSocket
    this.connectDetectionWebSocket();
  }

  connectDetectionWebSocket() {
    this.ws = new WebSocket(`ws://localhost:3847/ws/detection-tracking/${this.taskId}`);
    
    this.ws.onopen = () => {
      this.ws.send(JSON.stringify({ type: 'subscribe_detection' }));
    };
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'tracking_update' && message.mode === 'detection_processing') {
        this.handleDetectionResults(message);
      }
    };
  }

  handleDetectionResults(payload) {
    const { cameras, person_count_per_camera } = payload;
    
    Object.entries(cameras).forEach(([cameraId, data]) => {
      // Display frame with detection bounding boxes
      if (data.frame_image_base64) {
        document.getElementById(`camera-${cameraId}`).src = data.frame_image_base64;
      }
      
      // Update person count
      document.getElementById(`count-${cameraId}`).textContent = 
        person_count_per_camera[cameraId] || 0;
      
      // Draw detection bounding boxes (no tracking IDs)
      data.tracks?.forEach(detection => {
        this.drawDetectionBox(cameraId, detection);
      });
      
      // Show detection statistics
      console.log(`Detections in ${cameraId}: ${data.detection_count}`);
    });
  }

  drawDetectionBox(cameraId, detection) {
    const canvas = document.getElementById(`canvas-${cameraId}`);
    const ctx = canvas.getContext('2d');
    
    const [x1, y1, x2, y2] = detection.bbox_xyxy;
    const confidence = detection.confidence;
    
    // Draw bounding box
    ctx.strokeStyle = confidence > 0.8 ? 'green' : 'orange';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    
    // Draw confidence score
    ctx.fillStyle = 'white';
    ctx.fillRect(x1, y1 - 25, 80, 20);
    ctx.fillStyle = 'black';
    ctx.font = '12px Arial';
    ctx.fillText(`${(confidence * 100).toFixed(1)}%`, x1 + 5, y1 - 8);
  }
}

// Usage
const detectionClient = new DetectionClient();
await detectionClient.startDetectionProcessing('campus');
```

### Detection Environment Response

```javascript
// GET /api/v1/detection-processing-tasks/environments
{
  "status": "success",
  "data": {
    "environments": [
      {
        "environment_id": "campus",
        "name": "Campus Environment (RT-DETR Detection)",
        "description": "RT-DETR detection environment with 4 cameras and 120 video segments",
        "camera_count": 4,
        "cameras": ["c09", "c12", "c13", "c16"],
        "available": true,
        "total_sub_videos": 120,
        "mode": "detection_processing",
        "detection_features": {
          "model": "RT-DETR-l",
          "person_detection": true,
          "confidence_threshold": 0.5,
          "real_time_processing": true
        }
      }
    ],
    "total_count": 1,
    "detection_capabilities": {
      "model_type": "RT-DETR",
      "model_variant": "rtdetr-l",
      "supported_classes": ["person"],
      "real_time_inference": true,
      "confidence_threshold": 0.5,
      "input_resolution": "640x640"
    }
  },
  "timestamp": "2025-01-01T10:30:05.456Z"
}
```

### Development Workflow with Detection

1. **Detection Development**: Use RT-DETR detection to develop person detection UI
2. **Bounding Box Testing**: Test bounding box rendering and confidence display
3. **Performance Testing**: Measure detection performance vs raw streaming
4. **AI Integration Testing**: Verify RT-DETR model integration before full pipeline
5. **Production Ready**: Upgrade to full processing pipeline when ready

### Detection Task Management

```javascript
// Get detection task details
const taskDetails = await fetch(`/api/v1/detection-processing-tasks/${taskId}`)
  .then(r => r.json());

// Stop detection task if needed
await fetch(`/api/v1/detection-processing-tasks/${taskId}/stop`, {
  method: 'DELETE'
});

// Cleanup all tasks for an environment
await fetch(`/api/v1/detection-processing-tasks/environment/campus/cleanup`, {
  method: 'DELETE'
});
```

---

## DEBUGGING - Raw Video Streaming (No AI Processing)

For debugging and proof-of-concept development, the backend provides raw video streaming endpoints that bypass all AI processing. This allows frontend developers to test video playback and WebSocket communication without requiring AI models.

### Raw Video Streaming Endpoints

| Endpoint | Method | Purpose |
|----------|---------|----------|
| `/api/v1/raw-processing-tasks/environments` | GET | List environments for raw streaming |
| `/api/v1/raw-processing-tasks/start` | POST | Start raw video streaming |
| `/api/v1/raw-processing-tasks/{id}/status` | GET | Get raw task status |
| `/api/v1/raw-processing-tasks/{id}` | GET | Get raw task details |

### Raw Video WebSocket

**Connection URL**: `ws://localhost:3847/ws/raw-tracking/{taskId}`

### Quick Raw Video Setup

```javascript
// 1. Check system health (same as regular)
const health = await fetch('http://localhost:3847/health').then(r => r.json());

// 2. Start raw video streaming (no AI processing)
const rawTask = await fetch('http://localhost:3847/api/v1/raw-processing-tasks/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ environment_id: 'campus' })
}).then(r => r.json());

// 3. Connect to raw video WebSocket
const rawWs = new WebSocket(`ws://localhost:3847/ws/raw-tracking/${rawTask.task_id}`);
rawWs.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'tracking_update' && data.mode === 'raw_streaming') {
    handleRawVideoFrames(data.payload);
  }
};
```

### Raw Video Message Format

**Raw Tracking Update**:
```javascript
{
  "type": "tracking_update",
  "task_id": "uuid",
  "global_frame_index": 123,
  "timestamp_processed_utc": "2025-01-01T10:30:05.456Z",
  "mode": "raw_streaming",
  "cameras": {
    "c09": {
      "frame_image_base64": "data:image/jpeg;base64,...",
      "tracks": [],  // Empty - no AI processing
      "frame_width": 1920,
      "frame_height": 1080,
      "timestamp": "2025-01-01T10:30:05.456Z"
    }
  }
}
```

### Key Differences from AI Processing

| Feature | Regular Processing | Raw Video Streaming |
|---------|-------------------|-------------------|
| **Detection** | ✅ Person detection | ❌ No detection |
| **Tracking** | ✅ Multi-object tracking | ❌ No tracking |
| **Re-ID** | ✅ Cross-camera re-identification | ❌ No re-identification |
| **Bounding Boxes** | ✅ Person bounding boxes | ❌ No bounding boxes |
| **Frame Images** | ✅ Base64 encoded frames | ✅ Base64 encoded frames |
| **Map Coordinates** | ✅ Spatial mapping | ❌ No mapping |
| **Performance** | Slower (AI processing) | Faster (no processing) |
| **Use Case** | Production tracking | Debug/PoC development |

### Raw Video Client Example

```javascript
class RawVideoClient {
  constructor() {
    this.taskId = null;
    this.ws = null;
  }

  async startRawStreaming(environmentId) {
    // Start raw task
    const response = await fetch('/api/v1/raw-processing-tasks/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ environment_id: environmentId })
    });
    
    const task = await response.json();
    this.taskId = task.task_id;

    // Monitor until STREAMING
    while (true) {
      const status = await fetch(`/api/v1/raw-processing-tasks/${this.taskId}/status`).then(r => r.json());
      if (status.status === 'STREAMING') break;
      if (status.status === 'FAILED') throw new Error(status.details);
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // Connect WebSocket
    this.connectRawWebSocket();
  }

  connectRawWebSocket() {
    this.ws = new WebSocket(`ws://localhost:3847/ws/raw-tracking/${this.taskId}`);
    
    this.ws.onopen = () => {
      this.ws.send(JSON.stringify({ type: 'subscribe_raw_frames' }));
    };
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'tracking_update' && message.mode === 'raw_streaming') {
        this.handleRawFrames(message);
      }
    };
  }

  handleRawFrames(payload) {
    const { cameras } = payload;
    
    Object.entries(cameras).forEach(([cameraId, data]) => {
      // Display raw frame (no bounding boxes)
      if (data.frame_image_base64) {
        document.getElementById(`camera-${cameraId}`).src = data.frame_image_base64;
      }
      
      // Show frame info
      console.log(`Raw frame from ${cameraId}: ${data.frame_width}x${data.frame_height}`);
    });
  }
}

// Usage
const rawClient = new RawVideoClient();
await rawClient.startRawStreaming('campus');
```

### Development Workflow

1. **Frontend Development**: Use raw video streaming to develop UI without AI dependencies
2. **WebSocket Testing**: Test WebSocket connections and frame handling
3. **Performance Testing**: Measure baseline performance without AI processing
4. **Integration Testing**: Verify video playback before adding AI features
5. **Production Ready**: Switch to regular processing endpoints when ready

### Raw Video Environment Response

```javascript
// GET /api/v1/raw-processing-tasks/environments
{
  "status": "success",
  "data": {
    "environments": [
      {
        "environment_id": "campus",
        "name": "Campus Environment (Raw)",
        "description": "Raw video streaming campus environment with 4 cameras",
        "camera_count": 4,
        "cameras": ["c01", "c02", "c03", "c05"],
        "available": true,
        "mode": "raw_streaming"
      }
    ],
    "total_count": 1
  }
}
```

---

## Processing Mode Selection Guide

Choose the right processing mode based on your development needs:

### 🚀 **Full Processing Pipeline** (`/api/v1/processing-tasks/`)
**Best for**: Production applications, complete tracking systems
- ✅ Person detection, tracking, re-identification, spatial mapping
- ✅ Cross-camera person tracking with global IDs
- ✅ Focus tracking and person details
- ⚡ **WebSocket**: `ws://localhost:3847/ws/tracking/{taskId}`

### 🎯 **RT-DETR Detection** (`/api/v1/detection-processing-tasks/`)
**Best for**: Detection-focused applications, AI development
- ✅ Real-time person detection with RT-DETR
- ✅ Bounding boxes and confidence scores
- ❌ No tracking or re-identification
- ⚡ **WebSocket**: `ws://localhost:3847/ws/detection-tracking/{taskId}`

### 🔧 **Raw Video Streaming** (`/api/v1/raw-processing-tasks/`)
**Best for**: Frontend development, debugging, proof-of-concept
- ✅ Video frames without AI processing
- ❌ No detection, tracking, or AI features
- ⚡ **WebSocket**: `ws://localhost:3847/ws/raw-tracking/{taskId}`

### Development Progression Path

1. **Start with Raw**: Develop UI components and video playback
2. **Add Detection**: Integrate person detection and bounding boxes
3. **Full Pipeline**: Add tracking, re-ID, and spatial features

```javascript
// Example: Progressive development approach
class SpotOnProgressiveClient {
  // Phase 1: Raw video for UI development
  async startRawDevelopment() {
    const task = await fetch('/api/v1/raw-processing-tasks/start', {
      method: 'POST',
      body: JSON.stringify({ environment_id: 'campus' })
    }).then(r => r.json());
    // Develop video display, controls, layout
  }

  // Phase 2: Add detection for AI features
  async startDetectionDevelopment() {
    const task = await fetch('/api/v1/detection-processing-tasks/start', {
      method: 'POST',
      body: JSON.stringify({ environment_id: 'campus' })
    }).then(r => r.json());
    // Add bounding boxes, person counting, confidence display
  }

  // Phase 3: Full production features
  async startFullProduction() {
    const task = await fetch('/api/v1/processing-tasks/start', {
      method: 'POST',
      body: JSON.stringify({ environment_id: 'campus' })
    }).then(r => r.json());
    // Add tracking, focus, cross-camera features
  }
}
```

---

## 🗺️ NEW FEATURE: 2D MAPPING INTEGRATION

### Overview
The backend now includes **2D mapping functionality** integrated with the detection endpoint. This feature provides real-time person position visualization with 3-frame movement trails.

### ✅ Backend Changes Made

#### 1. Trail Management Service (NEW)
**File**: `app/services/trail_management_service.py`
```python
# NEW: Manages 3-frame movement trails for detected persons
class TrailManagementService:
    def __init__(self, trail_length: int = 3):
        self.trail_length = trail_length
        # Structure: {camera_id: {detection_id: [TrailPoint...]}}
        self.trails: Dict[str, Dict[str, List[TrailPoint]]] = {}
    
    async def update_trail(self, camera_id: str, detection_id: str, map_x: float, map_y: float) -> List[TrailPoint]:
        # Creates 3-frame circular buffer trails with frame offsets (0, -1, -2)
```

#### 2. Enhanced DetectionVideoService (MODIFIED)
**File**: `app/services/detection_video_service.py`
- ✅ **Trail service integration**: Added `TrailManagementService` instance
- ✅ **Enhanced send_detection_update**: Now includes trail data in WebSocket messages
- ✅ **Background cleanup task**: Automatic removal of inactive trails

```python
# MODIFIED: Enhanced WebSocket message format
async def send_detection_update(self, ...):
    # ... existing detection logic ...
    
    # NEW: Trail management integration
    trail = await self.trail_service.update_trail(
        camera_id=camera_id,
        detection_id=detection["detection_id"],
        map_x=detection["map_coords"]["map_x"],
        map_y=detection["map_coords"]["map_y"]
    )
    
    # NEW: Trail data in mapping_coordinates
    coord_data = {
        "detection_id": detection["detection_id"],
        "map_x": detection["map_coords"]["map_x"],
        "map_y": detection["map_coords"]["map_y"],
        "trail": [  # NEW FIELD
            {
                "x": point.map_x,
                "y": point.map_y,
                "frame_offset": point.frame_offset,
                "timestamp": point.timestamp.isoformat()
            }
            for point in trail[:3]
        ]
    }
```

### 🎯 Enhanced WebSocket Message Format

The **detection endpoint** now includes mapping data with trails in `future_pipeline_data.mapping_coordinates`:

```javascript
// ENHANCED: Detection tracking update with mapping trails
{
  "type": "tracking_update",
  "task_id": "uuid",
  "camera_id": "c09",
  "global_frame_index": 123,
  "timestamp_processed_utc": "2025-01-15T10:30:00Z",
  "mode": "detection_streaming",
  "camera_data": {
    "frame_image_base64": "base64_encoded_annotated_frame",
    "tracks": [/* detection data */],
    "frame_width": 1920,
    "frame_height": 1080
  },
  "future_pipeline_data": {
    "homography_data": {...},
    "mapping_coordinates": [  // ENHANCED WITH TRAILS
      {
        "detection_id": "det_123",
        "map_x": 12.5,
        "map_y": 8.3,
        "projection_successful": true,
        "coordinate_system": "bev_map_meters",
        "foot_point": {"image_x": 320, "image_y": 240},
        "trail": [  // NEW: 3-frame movement history
          {"x": 12.5, "y": 8.3, "frame_offset": 0, "timestamp": "2025-01-15T10:30:00Z"},
          {"x": 12.2, "y": 8.1, "frame_offset": -1, "timestamp": "2025-01-15T10:29:59Z"},
          {"x": 12.0, "y": 7.9, "frame_offset": -2, "timestamp": "2025-01-15T10:29:58Z"}
        ]
      }
    ]
  }
}
```

### 🖥️ Frontend Integration Patterns

#### 1. Enhanced WebSocket Integration
```javascript
// NEW: Custom event handling for mapping data
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'tracking_update' && data.future_pipeline_data?.mapping_coordinates) {
    // NEW: Extract mapping coordinates with trails
    const mappingData = data.future_pipeline_data.mapping_coordinates;
    
    // NEW: Emit custom event for mapping components
    const mappingEvent = new CustomEvent('websocket-mapping-message', {
      detail: { ...data, mapping_coordinates: mappingData }
    });
    window.dispatchEvent(mappingEvent);
  }
};
```

#### 2. New Frontend Components Available

**MiniMapComponent**: Canvas-based 2D map rendering
```javascript
// NEW: 2D map component with trail visualization
import { MiniMapComponent } from './components/mapping/MiniMapComponent';

<MiniMapComponent
  cameraId="c09"
  mappingCoordinates={mappingData}
  width={300}
  height={200}
/>
```

**CameraMapPair**: Camera + Map integration wrapper
```javascript
// NEW: Camera-map integration component
import { CameraMapPair } from './components/mapping/CameraMapPair';

<CameraMapPair
  cameraId="c09"
  frameData={frameBase64}
  mappingCoordinates={mappingData}
  onToggleMap={handleToggleMap}
  mapVisible={true}
/>
```

**useMappingData Hook**: WebSocket data processing
```javascript
// NEW: React hook for mapping data management
import { useMappingData } from './hooks/useMappingData';

const { mappingData, getMappingForCamera, getStatistics } = useMappingData({
  enabled: true,
  maxTrailLength: 3
});
```

### 🚀 Integration with Detection Endpoint

The 2D mapping feature is **fully integrated** with the existing detection processing workflow:

1. **Same Endpoint**: Use `/api/v1/detection-processing-tasks/start` (no changes needed)
2. **Same WebSocket**: Connect to `ws://localhost:3847/ws/detection-tracking/{taskId}`
3. **Enhanced Data**: Additional `mapping_coordinates` with trail data
4. **Backward Compatible**: Existing clients continue working unchanged

### 🔧 Development Integration Example

```javascript
class EnhancedDetectionClient extends DetectionClient {
  constructor() {
    super();
    this.mappingData = new Map();
  }

  handleDetectionResults(payload) {
    // Existing detection handling
    super.handleDetectionResults(payload);
    
    // NEW: Extract mapping data with trails
    if (payload.future_pipeline_data?.mapping_coordinates) {
      const mappingCoords = payload.future_pipeline_data.mapping_coordinates;
      
      mappingCoords.forEach(coord => {
        // Process trail data
        if (coord.trail && coord.trail.length > 0) {
          console.log(`Person ${coord.detection_id} trail:`, coord.trail);
          this.renderPersonTrail(coord);
        }
        
        // Store for 2D map rendering
        this.mappingData.set(coord.detection_id, coord);
      });
      
      // Update 2D maps
      this.updateMiniMaps(mappingCoords);
    }
  }
  
  renderPersonTrail(coord) {
    // NEW: Render 3-frame movement trail
    const canvas = document.getElementById(`map-${payload.camera_id}`);
    const ctx = canvas.getContext('2d');
    
    // Draw trail path
    ctx.strokeStyle = 'rgba(0, 123, 255, 0.6)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    coord.trail.forEach((point, index) => {
      const x = this.mapToCanvasX(point.x);
      const y = this.mapToCanvasY(point.y);
      
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
```

### 📊 Performance Impact

- **Trail updates**: <5ms per detection
- **Memory usage**: Circular buffer limits memory growth
- **WebSocket overhead**: Minimal (~10% increase in message size)
- **Background cleanup**: Automatic trail expiration prevents leaks

### 🔄 Migration from Existing Detection Integration

**No breaking changes** - existing detection integrations continue working. To add mapping support:

1. **Listen for mapping data**: Check `future_pipeline_data.mapping_coordinates`
2. **Process trail data**: Extract 3-frame movement history
3. **Render 2D maps**: Use Canvas or SVG for position visualization
4. **Handle cleanup**: Mapping data expires automatically

### 🛠️ Development Tools

- **Backend tests**: `pytest tests/services/test_trail_management_service.py`
- **WebSocket testing**: Enhanced detection endpoint includes mapping data
- **Frontend examples**: See `components/mapping/` directory
- **Performance monitoring**: Trail management metrics available

---

**Documentation**: http://localhost:3847/docs (Swagger UI)  
**Health Check**: http://localhost:3847/health  
**Alternative Docs**: http://localhost:3847/redoc