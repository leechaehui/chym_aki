CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    login_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE,
    logout_at TIMESTAMP WITH TIME ZONE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    expire_reason VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS request_logs (
    request_id UUID NOT NULL,
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INT,
    latency_ms FLOAT,
    request_time TIMESTAMP WITH TIME ZONE NOT NULL,
    response_time TIMESTAMP WITH TIME ZONE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    payload JSONB,
    log_seq BIGSERIAL,
    is_sampled BOOLEAN DEFAULT FALSE,
    log_version VARCHAR(10) DEFAULT '1.0',
    PRIMARY KEY (request_time, request_id)
) PARTITION BY RANGE (request_time);

CREATE TABLE IF NOT EXISTS request_logs_default PARTITION OF request_logs DEFAULT;

CREATE TABLE IF NOT EXISTS request_logs_dlq (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    payload JSONB NOT NULL,
    reason TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
