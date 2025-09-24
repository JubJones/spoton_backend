Currently Active HTTP/REST Endpoints

  Core System Endpoints (Live)

  1. GET /health - System health check
  2. POST /api/v1/detection-processing-tasks/start - Start detection task
  3. GET /api/v1/detection-processing-tasks/{taskId}/status - Get task status
  4. GET /api/v1/analytics/real-time/metrics - Real-time metrics
  5. GET /api/v1/analytics/real-time/active-persons - Active persons data
  6. GET /api/v1/environments/{environmentId}/validate - Environment validation

  GroupViewPage Specific Endpoints (Live)

  7. GET /api/v1/raw-processing-tasks - List raw processing tasks
  8. POST /api/v1/raw-processing-tasks/start - Start raw processing task
  9. GET /api/v1/raw-processing-tasks/{taskId}/status - Raw task status
  10. POST /api/v1/raw-processing-tasks/environment/{environment}/cleanup - Cleanup environment

  Media Endpoints (Live)

  11. GET /api/v1/media/frames/{taskId}/{cameraId}?quality=85&v={timestamp} - Camera frame images

  Testing/Service Discovery (Live)

  12. GET /system/info - System information
  13. GET /system/performance - System performance metrics
  14. GET /system/limits - System limits
  15. POST /processing/start - Processing start (alt endpoint)

  Currently Active WebSocket Endpoints

  Real-time Data Streams (Live)

  1. WS /ws/tracking/{taskId} - Real-time tracking data
  2. WS /ws/raw-tracking/{taskId} - Raw tracking data stream
  3. WS /ws/test - WebSocket connection testing
  4. WS /ws/health - WebSocket health monitoring

  Base URLs Being Used

  - HTTP API: http://localhost:3847 (from VITE_API_BASE_URL)
  - WebSocket: ws://localhost:3847 (from VITE_WS_BASE_URL)

  Key Finding

  The frontend has TWO PARALLEL API PATTERNS:
  1. Standard endpoints (/api/v1/detection-processing-tasks/*) - Used by hooks
  2. Raw endpoints (/api/v1/raw-processing-tasks/*) - Used by GroupViewPage
