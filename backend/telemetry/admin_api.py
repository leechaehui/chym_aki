from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
from .streamer import streamer
from queries.telemetry_queries import (
    sql_telemetry_stats_24h,
    sql_telemetry_dlq_count_24h,
    sql_telemetry_chart_24h,
    sql_telemetry_recent_logs,
    sql_telemetry_error_distribution_24h,
    sql_telemetry_stats_by_date,
    sql_telemetry_dlq_count_by_date,
    sql_telemetry_error_distribution_by_date,
    sql_telemetry_recent_logs_by_date,
)

admin_router = APIRouter(prefix="/admin/telemetry", tags=["admin-telemetry"])

def verify_admin():
    # Admin Authentication placeholder
    pass

def get_utc_bounds(date_str: str = None):
    kst_tz = timezone(timedelta(hours=9))
    now_kst = datetime.now(timezone.utc).astimezone(kst_tz)
    if not date_str:
        target_date = now_kst.date()
    else:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    from datetime import time
    start_kst = datetime.combine(target_date, time.min).replace(tzinfo=kst_tz)
    end_kst = start_kst + timedelta(days=1)
    
    start_utc = start_kst.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_kst.astimezone(timezone.utc).replace(tzinfo=None)
    return start_utc, end_utc

@admin_router.get("/stats")
async def get_stats(date: str = None, _=Depends(verify_admin)):
    if not streamer.db_pool:
        # DB 셋업이 안된 로컬 환경용 방어 코드
        raise HTTPException(status_code=503, detail="Database connection pool is not initialized")
    
    start_utc, end_utc = get_utc_bounds(date)
    
    async with streamer.db_pool.acquire() as conn:
        row = await conn.fetchrow(sql_telemetry_stats_by_date(), start_utc, end_utc)
        
        total = row["total"] if row and row["total"] else 0
        errors = row["errors"] if row and row["errors"] else 0
        avg_lat = row["avg_latency"] if row and row["avg_latency"] else 0.0
        
        error_rate = round((errors / total * 100) if total > 0 else 0, 2)
        avg_latency_ms = round(avg_lat, 2)
        
        dropped = await conn.fetchval(sql_telemetry_dlq_count_by_date(), start_utc, end_utc)
        
        dist_rows = await conn.fetch(sql_telemetry_error_distribution_by_date(), start_utc, end_utc)
        error_dist = [{"status_code": r["status_code"], "count": r["count"]} for r in dist_rows]
        
    from .memory_guard import get_memory_usage
    mem = get_memory_usage()
        
    return {
        "total_requests": total or 0,
        "error_rate": error_rate,
        "avg_latency_ms": avg_latency_ms,
        "dropped_logs": dropped or 0,
        "memory_usage": mem,
        "error_distribution": error_dist
    }

@admin_router.get("/chart")
async def get_chart_data(date: str = None, _=Depends(verify_admin)):
    if not streamer.db_pool:
        raise HTTPException(status_code=503, detail="Database connection pool is not initialized")
        
    start_utc_naive, end_utc_naive = get_utc_bounds(date)
    kst_tz = timezone(timedelta(hours=9))
    
    from queries.telemetry_queries import sql_telemetry_chart_by_date
    async with streamer.db_pool.acquire() as conn:
        rows = await conn.fetch(sql_telemetry_chart_by_date(), start_utc_naive, end_utc_naive)
        
    data_dict = {}
    for row in rows:
        dt = row["hour_bucket"]
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(kst_tz)
        
        hour_str = dt.strftime("%H")
        data_dict[hour_str] = row
        
    data = []
    
    for i in range(24):
        hour_str = f"{i:02d}"
        if hour_str in data_dict:
            row = data_dict[hour_str]
            data.append({
                "time": f"{hour_str}:00",
                "requests": row["requests"],
                "latency": round(row["avg_latency"], 1),
                "errors": row["errors"]
            })
        else:
            data.append({
                "time": f"{hour_str}:00",
                "requests": 0,
                "latency": 0.0,
                "errors": 0
            })
            
    return data

@admin_router.get("/recent")
async def get_recent_logs(date: str = None, _=Depends(verify_admin)):
    if not streamer.db_pool:
        raise HTTPException(status_code=503, detail="Database connection pool is not initialized")
        
    start_utc, end_utc = get_utc_bounds(date)
        
    async with streamer.db_pool.acquire() as conn:
        rows = await conn.fetch(sql_telemetry_recent_logs_by_date(), start_utc, end_utc)
        
    result = []
    for r in rows:
        dt = r["request_time"]
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone()
            
        result.append({
            "request_id": str(r["request_id"]),
            "method": r["method"],
            "endpoint": r["endpoint"],
            "status_code": r["status_code"],
            "latency_ms": round(r["latency_ms"] or 0, 1),
            "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "is_sampled": r["is_sampled"]
        })
        
    return result

@admin_router.get("/active")
async def get_active_requests_endpoint(_=Depends(verify_admin)):
    from .active_requests import get_active_requests
    active = get_active_requests()
    
    result = []
    for req_id, data in active.items():
        dt = data.get("start_time")
        if dt and not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone()
            
        result.append({
            **data,
            "start_time": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None
        })
        
    # sort by start time desc
    result.sort(key=lambda x: x["start_time"] or "", reverse=True)
    return result
