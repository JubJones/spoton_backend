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
| `/api/v1/processing-tasks/start` | POST | Start processing |
| `/api/v1/processing-tasks/{id}/status` | GET | Task status |
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
- **Tracking**: `ws://localhost:3847/ws/tracking/{taskId}`
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

---

**Documentation**: http://localhost:3847/docs (Swagger UI)  
**Health Check**: http://localhost:3847/health  
**Alternative Docs**: http://localhost:3847/redoc