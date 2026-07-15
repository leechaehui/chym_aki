"""텔레메트리 대시보드 관리자 화면용 SQL 쿼리.

연관 백엔드     : telemetry/admin_api.py
대상 테이블     : request_logs, request_logs_dlq
"""

def sql_telemetry_stats_24h() -> str:
    """최근 24시간의 트래픽, 에러율, 평균 Latency 집계 쿼리."""
    return """
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status_code >= 500) as errors,
            AVG(latency_ms) as avg_latency
        FROM request_logs
        WHERE request_time >= NOW() - INTERVAL '24 hours'
    """

def sql_telemetry_dlq_count_24h() -> str:
    """최근 24시간 DLQ(실패 로그) 카운트 쿼리."""
    return "SELECT COUNT(*) FROM request_logs_dlq WHERE created_at >= NOW() - INTERVAL '24 hours'"

def sql_telemetry_error_distribution_24h() -> str:
    """최근 24시간 HTTP 상태 코드별 에러 분산 쿼리."""
    return """
        SELECT status_code, COUNT(*) as count
        FROM request_logs
        WHERE request_time >= NOW() - INTERVAL '24 hours'
          AND status_code >= 400
        GROUP BY status_code
        ORDER BY count DESC
    """

def sql_telemetry_chart_24h() -> str:
    """시간별 트래픽 및 Latency, Error 분산 집계 차트 쿼리."""
    return """
        SELECT 
            date_trunc('hour', request_time) as hour_bucket,
            COUNT(*) as requests,
            COALESCE(AVG(latency_ms), 0) as avg_latency,
            COUNT(*) FILTER (WHERE status_code >= 500) as errors
        FROM request_logs
        WHERE request_time >= NOW() - INTERVAL '24 hours'
        GROUP BY hour_bucket
        ORDER BY hour_bucket ASC
    """

def sql_telemetry_stats_by_date() -> str:
    return """
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status_code >= 500) as errors,
            COALESCE(AVG(latency_ms), 0) as avg_latency
        FROM request_logs
        WHERE request_time >= $1 AND request_time < $2
    """

def sql_telemetry_dlq_count_by_date() -> str:
    return "SELECT COUNT(*) FROM request_logs_dlq WHERE created_at >= $1 AND created_at < $2"

def sql_telemetry_error_distribution_by_date() -> str:
    return """
        SELECT status_code, COUNT(*) as count
        FROM request_logs
        WHERE request_time >= $1 AND request_time < $2
          AND status_code >= 400
        GROUP BY status_code
        ORDER BY count DESC
    """

def sql_telemetry_chart_by_date() -> str:
    """특정 날짜(UTC 기준 start~end)의 시간별 트래픽 및 Latency 집계 쿼리."""
    return """
        SELECT 
            date_trunc('hour', request_time) as hour_bucket,
            COUNT(*) as requests,
            COALESCE(AVG(latency_ms), 0) as avg_latency,
            COUNT(*) FILTER (WHERE status_code >= 500) as errors
        FROM request_logs
        WHERE request_time >= $1 AND request_time < $2
        GROUP BY hour_bucket
        ORDER BY hour_bucket ASC
    """

def sql_telemetry_recent_logs(limit: int = 20) -> str:
    """최신 스트림 로그 조회 쿼리."""
    return f"""
        SELECT request_id, method, endpoint, status_code, latency_ms, request_time, is_sampled
        FROM request_logs
        ORDER BY request_time DESC
        LIMIT {limit}
    """

def sql_telemetry_recent_logs_by_date(limit: int = 100) -> str:
    return f"""
        SELECT request_id, method, endpoint, status_code, latency_ms, request_time, is_sampled
        FROM request_logs
        WHERE request_time >= $1 AND request_time < $2
        ORDER BY request_time DESC
        LIMIT {limit}
    """
