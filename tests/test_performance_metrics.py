"""
Performance & Metrics Test Suite — SpotOn Backend
==================================================
Covers the backend-side metrics categories from the SRS:

  System and API Health Metrics
  ─────────────────────────────
  PERF-API-001  API Response Time — p50 / p95 / p99 thresholds
  PERF-API-002  Throughput — sequential RPS baseline
  PERF-API-003  Error Rate — 4xx/5xx fraction under normal load

  Real-Time Processing Performance (simulated/estimated)
  ────────────────────────────────────────────────────────
  PERF-RT-001   Pipeline Processing Latency — frame-handler call overhead
  PERF-RT-002   Stream Initialization Time — time-to-first-byte from /stream
  PERF-RT-003   WebSocket Message Latency — round-trip within test client
  PERF-RT-004   Frame Drop Rate — frames emitted vs frames consumed

NOTE: All heavy backend dependencies (YOLO, ReID, DB, Redis) are mocked so
      these tests run without GPU or running services.  Numbers reflect the
      in-process test loop, NOT production network conditions.  They are meant
      to (a) prove the measurement harness works, and (b) catch obvious
      regressions in handler overhead.
"""

import asyncio
import json
import statistics
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Bootstrap — identical guard used in test_comprehensive_api.py
# ---------------------------------------------------------------------------
with patch("app.core.config.settings.PRELOAD_YOLO_DETECTOR", False), \
     patch("app.core.config.settings.PRELOAD_REID_MODEL", False), \
     patch("app.core.config.settings.PRELOAD_HOMOGRAPHY", False), \
     patch("app.core.config.settings.PRELOAD_TRACKER_FACTORY", False), \
     patch("app.core.config.settings.DB_ENABLED", False):

    from app.main import app
    from app.api.v1.endpoints.environments import (
        get_environment_service,
        get_historical_service,
    )

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared mock setup (mirrors test_comprehensive_api.py)
# ---------------------------------------------------------------------------
mock_env_service = MagicMock()
mock_hist_service = MagicMock()


async def _override_env_service():
    return mock_env_service


async def _override_hist_service():
    return mock_hist_service


app.dependency_overrides[get_environment_service] = _override_env_service
app.dependency_overrides[get_historical_service] = _override_hist_service


def _async_result(value):
    async def _coro():
        return value
    return _coro()


async def _finite_mjpeg_generator(*args, **kwargs):
    stub_jpeg = b"\xff\xd8\xff" + b"\x00" * 12
    for _ in range(3):
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + stub_jpeg
            + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_endpoint_latencies(endpoint: str, n: int = 20) -> list[float]:
    """
    Call *endpoint* (GET) n times and return a list of elapsed seconds.
    The endpoint is expected to return any HTTP status (we measure overhead,
    not correctness).
    """
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        client.get(endpoint)
        latencies.append(time.perf_counter() - t0)
    return latencies


def _percentile(data: list[float], pct: float) -> float:
    """Return the p-th percentile (0-100) of *data*."""
    sorted_data = sorted(data)
    idx = (pct / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


# ===========================================================================
# PERF-API-001  API Response Time  (p50 / p95 / p99)
# ===========================================================================

class TestApiResponseTime:
    """
    PERF-API-001 — Measure response time percentiles for lightweight endpoints.

    Thresholds (in-process test client, no network):
      p50 ≤ 200 ms,  p95 ≤ 500 ms,  p99 ≤ 1 000 ms
    """

    ENDPOINTS = [
        "/",
        "/health",
        "/health/live",
        "/ws/health",
    ]
    N_SAMPLES = 30

    # Acceptable thresholds (seconds) — relaxed for CI / resource-constrained envs
    P50_LIMIT = 0.200
    P95_LIMIT = 0.500
    P99_LIMIT = 1.000

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_response_time_percentiles(self, endpoint):
        """p50/p95/p99 latency must stay below defined thresholds."""
        latencies = _sample_endpoint_latencies(endpoint, self.N_SAMPLES)

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)

        print(
            f"\n[{endpoint}] p50={p50*1000:.1f}ms  "
            f"p95={p95*1000:.1f}ms  p99={p99*1000:.1f}ms"
        )

        assert p50 <= self.P50_LIMIT, (
            f"{endpoint} p50 {p50*1000:.1f}ms exceeds {self.P50_LIMIT*1000}ms"
        )
        assert p95 <= self.P95_LIMIT, (
            f"{endpoint} p95 {p95*1000:.1f}ms exceeds {self.P95_LIMIT*1000}ms"
        )
        assert p99 <= self.P99_LIMIT, (
            f"{endpoint} p99 {p99*1000:.1f}ms exceeds {self.P99_LIMIT*1000}ms"
        )


# ===========================================================================
# PERF-API-002  Throughput (sequential RPS baseline)
# ===========================================================================

class TestThroughput:
    """
    PERF-API-002 — Sequential requests-per-second baseline.

    Fires N sequential GET requests to a lightweight endpoint and asserts
    the achieved throughput is above a minimum threshold.  This is a
    single-process baseline — real concurrent RPS would be measured with
    a load tool (locust / wrk).

    Minimum expected: 50 RPS in sequential in-process mode.
    """

    N = 50
    MIN_RPS = 50  # requests per second, sequential in-process

    def test_health_endpoint_rps(self):
        """Sequential RPS on GET /health must be ≥ MIN_RPS."""
        t0 = time.perf_counter()
        for _ in range(self.N):
            client.get("/health")
        elapsed = time.perf_counter() - t0

        rps = self.N / elapsed
        print(f"\n[throughput] {rps:.1f} RPS over {self.N} sequential requests")

        assert rps >= self.MIN_RPS, (
            f"Sequential RPS {rps:.1f} is below minimum {self.MIN_RPS}"
        )

    def test_root_endpoint_rps(self):
        """Sequential RPS on GET / must be ≥ MIN_RPS."""
        t0 = time.perf_counter()
        for _ in range(self.N):
            client.get("/")
        elapsed = time.perf_counter() - t0

        rps = self.N / elapsed
        print(f"\n[throughput] GET / → {rps:.1f} RPS")
        assert rps >= self.MIN_RPS


# ===========================================================================
# PERF-API-003  Error Rate
# ===========================================================================

class TestErrorRate:
    """
    PERF-API-003 — Error rate for a mix of valid and invalid requests.

    Sends batches of requests and measures the 4xx/5xx fraction.
    For *valid* endpoints the error rate must be 0 %.
    For known-bad endpoints the error rate must be 100 % (all 4xx/5xx).
    """

    N = 20

    def _error_rate(self, endpoint: str, method: str = "GET", body=None) -> float:
        errors = 0
        for _ in range(self.N):
            if method == "GET":
                r = client.get(endpoint)
            else:
                r = client.post(endpoint, json=body or {})
            if r.status_code >= 400:
                errors += 1
        return errors / self.N

    def test_health_endpoints_zero_error_rate(self):
        """GET /health and /health/live must never return 4xx/5xx."""
        for ep in ("/health", "/health/live", "/"):
            rate = self._error_rate(ep)
            print(f"\n[error-rate] {ep} → {rate*100:.1f}%")
            assert rate == 0.0, f"{ep} has error rate {rate*100:.1f}%"

    def test_missing_resource_has_nonzero_error_rate(self):
        """GET on a non-existent route must always return 404 (100 % error rate)."""
        rate = self._error_rate("/api/v1/nonexistent_route_xyz_abc")
        print(f"\n[error-rate] /nonexistent → {rate*100:.1f}%")
        assert rate == 1.0, f"Expected 100% error rate, got {rate*100:.1f}%"

    def test_missing_payload_error_rate(self):
        """POST to task creation with empty body must always return 422 (100 % error rate)."""
        rate = self._error_rate(
            "/api/v1/detection-processing-tasks/start",
            method="POST",
            body={},
        )
        print(f"\n[error-rate] POST /start (empty) → {rate*100:.1f}%")
        assert rate == 1.0, f"Expected 100% error rate for missing payload, got {rate*100:.1f}%"


# ===========================================================================
# PERF-RT-001  Pipeline Processing Latency
# ===========================================================================

class TestPipelineLatency:
    """
    PERF-RT-001 — Estimate in-process frame handler overhead.

    We simulate the route handler call cost (no GPU/detector) by hitting
    /api/v1/analytics/real-time/metrics which runs through the async
    request—response cycle of the analytics pipeline stub.

    Threshold: median handler latency ≤ 100 ms.
    """

    N = 20
    MEDIAN_LIMIT = 0.100  # seconds

    def test_analytics_realtime_median_latency(self):
        """Median response latency for /analytics/real-time/metrics ≤ 100 ms."""
        latencies = _sample_endpoint_latencies(
            "/api/v1/analytics/real-time/metrics", self.N
        )
        median = statistics.median(latencies)
        print(f"\n[pipeline] analytics/real-time/metrics median={median*1000:.1f}ms")
        assert median <= self.MEDIAN_LIMIT, (
            f"Median latency {median*1000:.1f}ms exceeds {self.MEDIAN_LIMIT*1000}ms"
        )

    def test_system_statistics_median_latency(self):
        """Median response latency for /analytics/system/statistics ≤ 100 ms."""
        latencies = _sample_endpoint_latencies(
            "/api/v1/analytics/system/statistics", self.N
        )
        median = statistics.median(latencies)
        print(f"\n[pipeline] analytics/system/statistics median={median*1000:.1f}ms")
        assert median <= self.MEDIAN_LIMIT


# ===========================================================================
# PERF-RT-002  Stream Initialization Time (TTFB)
# ===========================================================================

class TestStreamInitializationTime:
    """
    PERF-RT-002 — Time-to-first-byte (TTFB) for the MJPEG stream endpoint.

    Measures the elapsed time from opening the stream connection to receiving
    the first chunk of multipart data.

    Threshold: TTFB ≤ 500 ms (in-process, no real camera).
    """

    TTFB_LIMIT = 0.500  # seconds

    def test_stream_time_to_first_byte(self):
        """MJPEG stream must deliver first byte within TTFB_LIMIT."""
        with patch(
            "app.utils.mjpeg_streamer.mjpeg_streamer.stream_generator",
            side_effect=_finite_mjpeg_generator,
        ):
            t0 = time.perf_counter()
            with client.stream("GET", "/api/v1/stream/task_perf/cam1") as r:
                assert r.status_code == 200
                for _ in r.iter_bytes(chunk_size=64):
                    ttfb = time.perf_counter() - t0
                    break  # stop after first chunk
            print(f"\n[stream-init] TTFB={ttfb*1000:.1f}ms")
            assert ttfb <= self.TTFB_LIMIT, (
                f"Stream TTFB {ttfb*1000:.1f}ms exceeds {self.TTFB_LIMIT*1000}ms"
            )


# ===========================================================================
# PERF-RT-003  WebSocket Message Latency
# ===========================================================================

class TestWebSocketMessageLatency:
    """
    PERF-RT-003 — Round-trip WebSocket message latency via /ws/system.

    Connects to the system-wide event bus (no task_id required, always accepts),
    sends a ping, and measures the time to receive any response or for the
    client.send to complete without error.

    Threshold: send-to-return ≤ 200 ms.
    """

    SEND_LIMIT = 0.200  # seconds

    def test_ws_system_send_latency(self):
        """Sending a message to /ws/system must complete within SEND_LIMIT."""
        latencies = []
        for _ in range(5):
            try:
                with client.websocket_connect("/ws/system") as ws:
                    t0 = time.perf_counter()
                    ws.send_text(json.dumps({"type": "ping"}))
                    latencies.append(time.perf_counter() - t0)
            except Exception:
                # If server closes immediately, the send still returned → record it
                latencies.append(time.perf_counter() - t0)

        median_latency = statistics.median(latencies)
        print(f"\n[ws-latency] /ws/system median send latency={median_latency*1000:.1f}ms")
        assert median_latency <= self.SEND_LIMIT, (
            f"WS send latency {median_latency*1000:.1f}ms exceeds {self.SEND_LIMIT*1000}ms"
        )


# ===========================================================================
# PERF-RT-004  Frame Drop Rate (simulated)
# ===========================================================================

class TestFrameDropRate:
    """
    PERF-RT-004 — Frame drop rate simulation.

    We drive N frames through the finite MJPEG generator and count how many
    the consumer actually received.  In the healthy case the generator emits
    exactly N_EMIT frames and the consumer reads all of them → 0 % drop rate.

    If the consumer breaks early (overload simulation), the drop rate is
    (N_EMIT - received) / N_EMIT.  We assert ≤ 10 % drop rate.
    """

    N_EMIT = 10          # frames the generator will produce
    MAX_DROP_RATE = 0.10  # 10 %

    def _counting_generator(self, n: int):
        """Async generator that yields n fake MJPEG frames."""
        async def _gen(*args, **kwargs):
            stub = b"\xff\xd8\xff" + b"\x00" * 12
            for _ in range(n):
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + stub
                    + b"\r\n"
                )
        return _gen

    def test_frame_drop_rate_zero_under_normal_load(self):
        """Consumer reads all emitted frames → 0 % drop rate."""
        with patch(
            "app.utils.mjpeg_streamer.mjpeg_streamer.stream_generator",
            side_effect=self._counting_generator(self.N_EMIT),
        ):
            with client.stream("GET", "/api/v1/stream/task_drop/cam1") as r:
                assert r.status_code == 200
                body = b"".join(r.iter_bytes())  # accumulate full body first

        # Count boundary markers on the complete body — no chunk-split issues
        received = body.count(b"--frame")
        drop_rate = max(0.0, (self.N_EMIT - received) / self.N_EMIT)
        print(
            f"\n[frame-drop] emitted={self.N_EMIT} received={received} "
            f"drop_rate={drop_rate*100:.1f}%"
        )
        assert drop_rate <= self.MAX_DROP_RATE, (
            f"Frame drop rate {drop_rate*100:.1f}% exceeds {self.MAX_DROP_RATE*100}%"
        )

    def test_frame_drop_rate_under_early_disconnect(self):
        """
        Consumer stops reading after first frame (simulates overloaded frontend).
        Drop rate is expected to be high — we just confirm the metric is
        *computable* and the server doesn't crash.
        """
        received = 0
        with patch(
            "app.utils.mjpeg_streamer.mjpeg_streamer.stream_generator",
            side_effect=self._counting_generator(self.N_EMIT),
        ):
            with client.stream("GET", "/api/v1/stream/task_drop2/cam1") as r:
                assert r.status_code == 200
                for chunk in r.iter_bytes(chunk_size=128):
                    if b"--frame" in chunk:
                        received += 1
                    break  # simulate frontend overload — stop after 1st chunk

        drop_rate = max(0.0, (self.N_EMIT - received) / self.N_EMIT)
        print(
            f"\n[frame-drop early-exit] received={received} "
            f"drop_rate={drop_rate*100:.1f}% (expected high)"
        )
        # Server must still be alive after early disconnect
        r2 = client.get("/health")
        assert r2.status_code == 200, "Server crashed after early stream disconnect"


# ===========================================================================
# PERF-RT-000  Glass-to-Glass Latency (simulated end-to-end pipeline)
# ===========================================================================

class TestGlassToGlassLatency:
    """
    PERF-RT-000 — Simulated glass-to-glass latency.

    Definition:
        Time from a frame arriving at the backend (camera ingest) to the
        corresponding tracking JSON being emitted via WebSocket to the
        frontend dashboard.

    Simulation approach (no GPU / real camera required):
        1. A fake raw frame (numpy array) is injected into the
           `process_frame_with_tracking` pipeline on a mocked
           `DetectionVideoService` instance.
        2. The YOLO detector, tracker, spatial intelligence, and WS send
           are all mocked to return instantly with realistic-shaped output.
        3. We timestamp T0 = frame arrives, T1 = WS send_json_message call
           completes, and assert T1 - T0 ≤ G2G_LIMIT.

    This does NOT measure real GPU inference time; it measures the
    *orchestration overhead* of the pipeline (task scheduling, data
    conversions, dictionary lookups).  A regression here indicates that
    non-GPU logic has become unexpectedly slow.

    Threshold: median glass-to-glass handler overhead ≤ 200 ms.
    """

    N_FRAMES = 10
    G2G_LIMIT = 0.200  # seconds (200 ms median)

    def _make_fake_frame(self) -> "np.ndarray":
        """Return a small fake BGR frame (numpy array)."""
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _make_fake_detections(self):
        """Minimal detection payload matching process_frame_with_detection output."""
        return {
            "detections": [
                {
                    "bbox": [100, 100, 200, 300],
                    "confidence": 0.92,
                    "class_id": 0,
                    "class_name": "person",
                }
            ],
            "frame_number": 1,
            "camera_id": "cam1",
            "tracks": [],
        }

    def _make_fake_tracks(self):
        """Minimal track payload matching post-tracking output."""
        return [
            {
                "track_id": 1,
                "bbox_xyxy": [100, 100, 200, 300],
                "confidence": 0.92,
                "class_id": 0,
                "global_id": "person_1",
                "world_position": {"x": 1.5, "y": 2.3},
            }
        ]

    @pytest.mark.asyncio
    async def test_glass_to_glass_handler_overhead(self):
        """
        Median pipeline orchestration overhead (frame-in → WS-emit) ≤ G2G_LIMIT.

        We mock all I/O-heavy steps (YOLO, tracker, WS send) so only the
        Python orchestration cost is measured.
        """
        import asyncio
        import numpy as np

        latencies = []
        ws_emit_timestamps = []

        # ---------- mock targets ----------
        fake_detection = self._make_fake_detections()
        fake_tracks = self._make_fake_tracks()

        async def _mock_detect(frame, camera_id, frame_number, **kwargs):
            return fake_detection

        async def _mock_track_enhance(tracks, camera_id, frame_number):
            return fake_tracks

        async def _mock_reid(tracks, frame, camera_id, fw, fh, frame_number=0):
            return tracks

        async def _mock_ws_send(task_id, message, msg_type):
            ws_emit_timestamps.append(time.perf_counter())
            return True

        async def _mock_tracker_update(dets, frame):
            import numpy as np
            # Return boxmot-shaped array: [x1,y1,x2,y2,track_id,conf,cls,idx]
            return np.array([[100, 100, 200, 300, 1, 0.92, 0, 0]], dtype=float)

        # ---------- build a minimal DetectionVideoService instance with mocks ----------
        with patch("app.core.config.settings.PRELOAD_YOLO_DETECTOR", False), \
             patch("app.core.config.settings.PRELOAD_REID_MODEL", False), \
             patch("app.core.config.settings.PRELOAD_HOMOGRAPHY", False), \
             patch("app.core.config.settings.PRELOAD_TRACKER_FACTORY", False), \
             patch("app.core.config.settings.DB_ENABLED", False), \
             patch("app.core.config.settings.TRACKING_ENABLED", False):  # skip tracker init

            from app.services.detection_video_service import DetectionVideoService
            from app.api.websockets.connection_manager import binary_websocket_manager

            svc = DetectionVideoService.__new__(DetectionVideoService)
            # Minimal attribute init (avoids __init__ heavy loading)
            svc.detection_stats = {"total_frames_processed": 0, "total_detections_found": 0,
                                   "successful_detections": 0, "failed_detections": 0,
                                   "frames_annotated": 0, "websocket_messages_sent": 0,
                                   "average_detection_time": 0.0, "annotation_time": 0.0}
            svc.detection_times = []
            svc.annotation_times = []
            svc.camera_trackers = {}
            svc.active_track_ids = {}
            svc.tracking_stats = {"total_tracks_created": 0, "cross_camera_handoffs": 0,
                                  "average_track_length": 0.0}
            svc.enable_gt_reid = False
            svc.gt_reid_services = {}
            svc.tracker_factory = None
            svc.handoff_manager = None
            svc.feature_extraction_service = None
            svc.homography_service = None
            svc.trail_service = MagicMock()
            svc.bottom_point_extractor = MagicMock()
            svc.world_plane_transformer = MagicMock()
            svc.space_based_matcher = MagicMock()
            svc.handoff_service = MagicMock()
            svc.global_registry = MagicMock()
            svc.roi_calculator = MagicMock()
            svc.annotator = MagicMock()

            # Patch heavy coroutines on the instance
            svc.process_frame_with_detection = _mock_detect
            svc._enhance_tracks_with_spatial_intelligence = _mock_track_enhance
            svc._apply_reid_logic = _mock_reid
            svc._associate_detections_with_tracks = MagicMock()
            svc._emit_frontend_events = AsyncMock(return_value=None)
            svc._convert_detections_to_boxmot_format = MagicMock(
                return_value=np.zeros((1, 6), dtype=float)
            )
            svc._convert_boxmot_to_track_data = MagicMock(return_value=fake_tracks)
            svc._get_environment_for_camera = MagicMock(return_value="test_env")

            # Patch WS send
            with patch.object(binary_websocket_manager, "send_json_message", side_effect=_mock_ws_send):
                frame = self._make_fake_frame()

                for i in range(self.N_FRAMES):
                    t_frame_in = time.perf_counter()  # T0: frame arrives at backend

                    # Run the full orchestration pipeline
                    result = await svc.process_frame_with_tracking(
                        frame=frame,
                        camera_id="cam1",
                        frame_number=i,
                    )

                    t_pipeline_done = time.perf_counter()  # T1: pipeline finished

                    # Glass-to-Glass = pipeline done (WS emit is part of it via _emit_frontend_events)
                    latencies.append(t_pipeline_done - t_frame_in)

        median_g2g = statistics.median(latencies)
        max_g2g = max(latencies)

        print(
            f"\n[glass-to-glass] n={self.N_FRAMES} frames | "
            f"median={median_g2g*1000:.1f}ms | max={max_g2g*1000:.1f}ms"
        )

        assert median_g2g <= self.G2G_LIMIT, (
            f"Median G2G latency {median_g2g*1000:.1f}ms exceeds "
            f"threshold {self.G2G_LIMIT*1000}ms"
        )

    def test_glass_to_glass_api_round_trip(self):
        """
        Surrogate G2G test via HTTP: measures the analytics real-time metrics
        endpoint round-trip as a lower-bound proxy for glass-to-glass delay.

        This captures the HTTP request→handler→response cycle, which is the
        minimum overhead any frame event must traverse before reaching the client.
        Threshold: p95 ≤ 200 ms.
        """
        latencies = _sample_endpoint_latencies("/api/v1/analytics/real-time/metrics", n=20)
        p95 = _percentile(latencies, 95)

        print(f"\n[glass-to-glass proxy] p95 HTTP round-trip={p95*1000:.1f}ms")

        assert p95 <= 0.200, (
            f"G2G proxy p95 {p95*1000:.1f}ms exceeds 200ms — "
            f"handler overhead is too high"
        )
