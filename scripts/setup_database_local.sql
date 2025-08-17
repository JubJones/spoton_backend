-- SpotOn Backend - Local Database Setup Script
-- Run this script to set up the local PostgreSQL database for development

-- Connect to PostgreSQL as superuser (postgres) and run these commands:
-- psql -U postgres -f setup_database_local.sql

-- Create user for SpotOn backend
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_user WHERE usename = 'spoton_user') THEN
        CREATE USER spoton_user WITH PASSWORD 'spoton_password';
        RAISE NOTICE 'Created user: spoton_user';
    ELSE
        RAISE NOTICE 'User spoton_user already exists';
    END IF;
END $$;

-- Create database
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'spotondb') THEN
        PERFORM dblink_exec('host=localhost user=' || current_user || ' dbname=' || current_database(), 
                           'CREATE DATABASE spotondb OWNER spoton_user');
        RAISE NOTICE 'Created database: spotondb';
    ELSE
        RAISE NOTICE 'Database spotondb already exists';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        -- Fallback if dblink is not available
        RAISE NOTICE 'Please run: CREATE DATABASE spotondb OWNER spoton_user;';
END $$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE spotondb TO spoton_user;

-- Connect to the spotondb database and set up extensions
\c spotondb

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create schema for tracking data
CREATE SCHEMA IF NOT EXISTS tracking;
GRANT ALL ON SCHEMA tracking TO spoton_user;

-- Create tables for tracking data
CREATE TABLE IF NOT EXISTS tracking.detection_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    task_id VARCHAR(255) NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    global_person_id VARCHAR(255),
    local_track_id INTEGER,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    confidence REAL NOT NULL,
    map_x REAL,
    map_y REAL,
    scene_id VARCHAR(100),
    frame_index INTEGER,
    metadata JSONB
);

-- Create hypertable for time-series data
SELECT create_hypertable('tracking.detection_events', 'timestamp', 
                        chunk_time_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_detection_events_task_id ON tracking.detection_events (task_id);
CREATE INDEX IF NOT EXISTS idx_detection_events_camera_id ON tracking.detection_events (camera_id);
CREATE INDEX IF NOT EXISTS idx_detection_events_person_id ON tracking.detection_events (global_person_id);
CREATE INDEX IF NOT EXISTS idx_detection_events_timestamp ON tracking.detection_events (timestamp DESC);

-- Create table for person identity tracking
CREATE TABLE IF NOT EXISTS tracking.person_identities (
    id BIGSERIAL PRIMARY KEY,
    global_person_id VARCHAR(255) UNIQUE NOT NULL,
    first_detected TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_detections INTEGER DEFAULT 0,
    cameras_seen TEXT[] DEFAULT '{}',
    features_vector BYTEA,  -- Stored ReID features
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_person_identities_global_id ON tracking.person_identities (global_person_id);
CREATE INDEX IF NOT EXISTS idx_person_identities_first_detected ON tracking.person_identities (first_detected);

-- Create table for analytics aggregations
CREATE TABLE IF NOT EXISTS tracking.analytics_summary (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    task_id VARCHAR(255) NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    person_count INTEGER DEFAULT 0,
    avg_confidence REAL DEFAULT 0.0,
    processing_fps REAL DEFAULT 0.0,
    metadata JSONB
);

-- Create hypertable for analytics data
SELECT create_hypertable('tracking.analytics_summary', 'timestamp',
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- Create materialized view for real-time analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS tracking.realtime_metrics AS
SELECT 
    task_id,
    camera_id,
    COUNT(DISTINCT global_person_id) as active_persons,
    AVG(confidence) as avg_confidence,
    MAX(timestamp) as last_updated
FROM tracking.detection_events 
WHERE timestamp > NOW() - INTERVAL '5 minutes'
GROUP BY task_id, camera_id;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_realtime_metrics_task_camera 
ON tracking.realtime_metrics (task_id, camera_id);

-- Create table for export jobs
CREATE TABLE IF NOT EXISTS tracking.export_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    task_id VARCHAR(255),
    parameters JSONB,
    file_path VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    file_size BIGINT
);

CREATE INDEX IF NOT EXISTS idx_export_jobs_status ON tracking.export_jobs (status);
CREATE INDEX IF NOT EXISTS idx_export_jobs_created ON tracking.export_jobs (created_at DESC);

-- Grant permissions to spoton_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tracking TO spoton_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tracking TO spoton_user;
GRANT ALL PRIVILEGES ON SCHEMA tracking TO spoton_user;

-- Create function to refresh realtime metrics
CREATE OR REPLACE FUNCTION tracking.refresh_realtime_metrics()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY tracking.realtime_metrics;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION tracking.refresh_realtime_metrics() TO spoton_user;

-- Create sample data retention policy (optional)
-- Keep detailed data for 30 days, aggregated data for 1 year
SELECT add_retention_policy('tracking.detection_events', INTERVAL '30 days', if_not_exists => true);
SELECT add_retention_policy('tracking.analytics_summary', INTERVAL '1 year', if_not_exists => true);

-- Create compression policy for better storage efficiency
SELECT add_compression_policy('tracking.detection_events', INTERVAL '7 days', if_not_exists => true);
SELECT add_compression_policy('tracking.analytics_summary', INTERVAL '1 month', if_not_exists => true);

-- Create continuous aggregate for hourly analytics (optional)
CREATE MATERIALIZED VIEW IF NOT EXISTS tracking.hourly_analytics
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    task_id,
    camera_id,
    COUNT(*) as detection_count,
    COUNT(DISTINCT global_person_id) as unique_persons,
    AVG(confidence) as avg_confidence
FROM tracking.detection_events
GROUP BY hour, task_id, camera_id;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('tracking.hourly_analytics',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => true);

-- Display setup completion message
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== SpotOn Database Setup Complete ===';
    RAISE NOTICE 'Database: spotondb';
    RAISE NOTICE 'User: spoton_user';
    RAISE NOTICE 'Schema: tracking';
    RAISE NOTICE 'Extensions: timescaledb';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables created:';
    RAISE NOTICE '- tracking.detection_events (hypertable)';
    RAISE NOTICE '- tracking.person_identities';
    RAISE NOTICE '- tracking.analytics_summary (hypertable)';
    RAISE NOTICE '- tracking.export_jobs';
    RAISE NOTICE '';
    RAISE NOTICE 'Views created:';
    RAISE NOTICE '- tracking.realtime_metrics (materialized)';
    RAISE NOTICE '- tracking.hourly_analytics (continuous aggregate)';
    RAISE NOTICE '';
    RAISE NOTICE 'Connection string:';
    RAISE NOTICE 'postgresql://spoton_user:spoton_password@localhost:5432/spotondb';
    RAISE NOTICE '';
END $$;