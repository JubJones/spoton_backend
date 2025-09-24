# SpotOn Backend - Frontend Integration Guide

**Base URL**: `http://localhost:3847` | **API Version**: v1

## Quick Start

```javascript
// 1. Check system health
const health = await fetch('http://localhost:3847/health').then(r => r.json());

// 2. Start detection processing
const task = await fetch('http://localhost:3847/api/v1/detection-processing-tasks/start', {
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

**Core Components**: RT-DETR AI Detection, Real-time Processing, Cross-Camera Tracking, Spatial Mapping, WebSocket Streaming

**Environments**:
- **Campus**: 4 cameras (c09, c12, c13, c16)
- **Factory**: 4 cameras (c01, c02, c03, c05)

**Performance**: Real-time detection processing, <100ms WebSocket latency

## REST API Reference

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|---------|----------|
| `/health` | GET | System health check |
| `/` | GET | Root endpoint |

### Detection Processing

| Endpoint | Method | Purpose |
|----------|---------|----------|
| `/api/v1/detection-processing-tasks` | GET | List detection tasks |
| `/api/v1/detection-processing-tasks/start` | POST | Start RT-DETR detection |
| `/api/v1/detection-processing-tasks/{taskId}/status` | GET | Detection task status |
| `/api/v1/detection-processing-tasks/environment/{environment}/cleanup` | POST | Cleanup environment tasks |

### Environment Management

| Endpoint | Method | Purpose |
|----------|---------|----------|
| `/api/v1/environments/` | GET | List environments |
| `/api/v1/environments/{environmentId}` | GET | Environment details |
| `/api/v1/environments/{environmentId}/date-ranges` | GET | Available date ranges |
| `/api/v1/environments/{environmentId}/cameras` | GET | Camera information |
| `/api/v1/environments/{environmentId}/zones` | GET | Zone configuration |
| `/api/v1/environments/{environmentId}/sessions` | GET | Session data |

### Analytics (Real-time)

| Endpoint | Method | Purpose |
|----------|---------|----------|
| `/api/v1/analytics/real-time/metrics` | GET | Real-time metrics |
| `/api/v1/analytics/real-time/active-persons` | GET | Active persons count |
| `/api/v1/analytics/real-time/camera-loads` | GET | Camera processing loads |
| `/api/v1/analytics/system/statistics` | GET | System statistics |

## Integration Patterns

### 1. System Health Check
```javascript
const health = await fetch('/health').then(r => r.json());
const isReady = health.status === 'healthy' &&
                health.detector_model_loaded &&
                health.prototype_tracker_loaded &&
                health.homography_matrices_precomputed;
```

### 2. Environment Discovery
```javascript
const environments = await fetch('/api/v1/environments/').then(r => r.json());
// Returns: environment_id, camera_count, zone_count, has_data
```

### 3. Detection Task Management
```javascript
// Start detection task
const task = await fetch('/api/v1/detection-processing-tasks/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ environment_id: 'campus' })
});

// Monitor status
const status = await fetch(`/api/v1/detection-processing-tasks/${taskId}/status`).then(r => r.json());
// Status: QUEUED → INITIALIZING → DOWNLOADING → EXTRACTING → PROCESSING → COMPLETED
```

## WebSocket Communication

### Connection URLs
- **Tracking Updates**: `ws://localhost:3847/ws/tracking/{taskId}`
- **Frame Streaming**: `ws://localhost:3847/ws/frames/{taskId}`
- **System Monitoring**: `ws://localhost:3847/ws/system`
- **Focus Tracking**: `ws://localhost:3847/ws/focus/{taskId}`
- **Analytics Stream**: `ws://localhost:3847/ws/analytics/{taskId}`

### Message Format

**Client → Server**:
```javascript
{ "type": "subscribe_tracking" }
{ "type": "ping" }
```

**Server → Client**:
```javascript
// Tracking update with detection results
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

  async startDetection(environmentId) {
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
    // Connect to focus WebSocket
    const focusWs = new WebSocket(`ws://localhost:3847/ws/focus/${this.taskId}`);
    focusWs.onopen = () => {
      focusWs.send(JSON.stringify({
        type: 'set_focus',
        person_id: personId
      }));
    };
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
POST /detection-processing-tasks/start → { "detail": "Invalid environment_id" }

// Task not found
GET /detection-processing-tasks/invalid/status → { "detail": "Task not found" }
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

  // 2. Start detection task
  const task = await startDetection('campus');

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

## Environment Details

### Environment Information
```javascript
// Get environment details
GET /api/v1/environments/{id}

// Get cameras
GET /api/v1/environments/{id}/cameras

// Get date ranges
GET /api/v1/environments/{id}/date-ranges?detailed=true

// Get zones
GET /api/v1/environments/{id}/zones

// Get sessions
GET /api/v1/environments/{id}/sessions
```

## Analytics Integration

### Real-time Analytics
```javascript
// Get current metrics
const metrics = await fetch('/api/v1/analytics/real-time/metrics').then(r => r.json());

// Get active persons across all cameras
const activePersons = await fetch('/api/v1/analytics/real-time/active-persons').then(r => r.json());

// Get camera processing loads
const cameraLoads = await fetch('/api/v1/analytics/real-time/camera-loads').then(r => r.json());

// Get system statistics
const systemStats = await fetch('/api/v1/analytics/system/statistics').then(r => r.json());
```

### Analytics WebSocket
```javascript
// Connect to analytics stream
const analyticsWs = new WebSocket(`ws://localhost:3847/ws/analytics/${taskId}`);
analyticsWs.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'analytics_update') {
    updateDashboard(data.metrics);
  }
};
```

---

**Documentation**: http://localhost:3847/docs (Swagger UI)
**Health Check**: http://localhost:3847/health
**Alternative Docs**: http://localhost:3847/redoc
