-- Migration: add geometric metrics events table for ROI/extractor/transform stats

CREATE TABLE IF NOT EXISTS geometric_metrics_events (
    id BIGSERIAL PRIMARY KEY,
    event_timestamp TIMESTAMPTZ NOT NULL,
    environment_id VARCHAR(50) NOT NULL,
    camera_id VARCHAR(50),
    extraction_total_attempts BIGINT NOT NULL DEFAULT 0,
    extraction_validation_failures BIGINT NOT NULL DEFAULT 0,
    extraction_success_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    transformation_total_attempts BIGINT,
    transformation_validation_failures BIGINT,
    transformation_success_rate DOUBLE PRECISION,
    roi_total_created BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_geometric_metrics_event_timestamp
    ON geometric_metrics_events (event_timestamp);

CREATE INDEX IF NOT EXISTS idx_geometric_metrics_env_camera
    ON geometric_metrics_events (environment_id, camera_id);
