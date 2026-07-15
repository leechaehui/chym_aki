import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List
import asyncpg
from datetime import datetime

from .fallback import append_to_file_fallback
from .memory_guard import should_sample
import os
from core.config import settings

logger = logging.getLogger(__name__)

DB_DSN = os.getenv("TELEMETRY_DB_DSN", settings.database_url.replace("+psycopg2", ""))

class StreamerWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.queue = asyncio.Queue(maxsize=10000)
        self.ema_load = 0.0
        self.alpha = 0.1  # Smoothing factor for EMA
        self._task = None

    def start(self, db_pool):
        self._task = asyncio.create_task(self._process_queue(db_pool))

    async def _process_queue(self, db_pool):
        batch = []
        while True:
            try:
                # Update EMA load (0 to 1 scale based on maxsize)
                current_load = self.queue.qsize() / 10000.0
                self.ema_load = (self.alpha * current_load) + ((1 - self.alpha) * self.ema_load)

                # Wait for items
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                batch.append(item)
                
                # Batch size 500
                if len(batch) >= 500:
                    await self._flush_batch(batch, db_pool)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self._flush_batch(batch, db_pool)
                    batch = []
            except asyncio.CancelledError:
                if batch:
                    await self._flush_batch(batch, db_pool)
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")

    async def _flush_batch(self, batch: List[Dict], db_pool):
        # Chunk size 50
        chunks = [batch[i:i + 50] for i in range(0, len(batch), 50)]
        for chunk in chunks:
            await self._insert_with_retry(chunk, db_pool)
        
        # Mark tasks done
        for _ in range(len(batch)):
            self.queue.task_done()

    async def _insert_with_retry(self, chunk: List[Dict], db_pool):
        retries = 3
        for attempt in range(retries):
            try:
                # bulk insert into request_logs
                async with db_pool.acquire() as conn:
                    query = """
                        INSERT INTO request_logs (
                            request_id, session_id, user_id, endpoint, method,
                            status_code, latency_ms, request_time, response_time,
                            ip_address, user_agent, payload, is_sampled
                        ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13)
                    """
                    values = [
                        (
                            c.get("request_id"), c.get("session_id"), c.get("user_id"),
                            c.get("endpoint"), c.get("method"), c.get("status_code"),
                            c.get("latency_ms"), c.get("request_time"), c.get("response_time"),
                            c.get("ip_address"), c.get("user_agent"), 
                            json.dumps(c.get("payload", {}), ensure_ascii=False, default=str), 
                            c.get("is_sampled", False)
                        )
                        for c in chunk
                    ]
                    await conn.executemany(query, values)
                return  # Success
            except Exception as e:
                logger.warning(f"Chunk insert failed (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt) # exponential backoff
                else:
                    await self._fallback_dlq(chunk, db_pool, str(e))

    async def _fallback_dlq(self, chunk: List[Dict], db_pool, reason: str):
        # Secondary fallback: request_logs_dlq
        try:
            async with db_pool.acquire() as conn:
                query = "INSERT INTO request_logs_dlq (payload, reason) VALUES ($1::jsonb, $2)"
                values = [(json.dumps(c, ensure_ascii=False, default=str), reason) for c in chunk]
                await conn.executemany(query, values)
        except Exception as e:
            logger.error(f"DLQ insert failed: {e}. Falling back to file.")
            # Tertiary fallback
            for c in chunk:
                append_to_file_fallback(c, reason)

class LogStreamer:
    def __init__(self):
        self.workers = [StreamerWorker(i) for i in range(3)]
        self.db_pool = None
        self.load_threshold = 0.8  # 80% EMA load threshold

    async def start(self):
        if DB_DSN.startswith("sqlite"):
            logger.info("SQLite detected. Disabling asyncpg streamer.")
            return
            
        self.db_pool = await asyncpg.create_pool(
            DB_DSN, 
            server_settings={'search_path': settings.app_schema}
        )
        for worker in self.workers:
            worker.start(self.db_pool)

    async def stop(self):
        for worker in self.workers:
            if worker._task:
                worker._task.cancel()
        if self.db_pool:
            await self.db_pool.close()

    def _get_target_worker(self, user_id: str) -> StreamerWorker:
        if not user_id:
            # Route to lowest load if no user_id
            return min(self.workers, key=lambda w: w.ema_load)
            
        # hash(user_id) % 3 deterministic routing
        target_idx = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 3
        target_worker = self.workers[target_idx]
        
        # Hybrid Routing Fallback
        if target_worker.ema_load > self.load_threshold:
            fallback_worker = min(self.workers, key=lambda w: w.ema_load)
            return fallback_worker
            
        return target_worker

    def push(self, log_entry: Dict[str, Any]):
        keep, state = should_sample()
        
        # if >85% memory, CRITICAL/ERROR only
        if not keep:
            status_code = log_entry.get("status_code", 200)
            if status_code < 500 and log_entry.get("exception") is None:
                return # Drop normal logs under heavy memory pressure

        log_entry["is_sampled"] = (state == "SAMPLED")

        # Payload max 2KB truncate
        payload_str = json.dumps(log_entry.get("payload", {}), ensure_ascii=False, default=str)
        if len(payload_str.encode('utf-8')) > 2048:
            log_entry["payload"] = {"_truncated": True, "data": payload_str[:2000]}

        worker = self._get_target_worker(log_entry.get("user_id"))
        try:
            worker.queue.put_nowait(log_entry)
        except asyncio.QueueFull:
            # Auto drop on queue overflow defense
            logger.warning("Queue overflow, dropping log entry to prevent memory overload")

streamer = LogStreamer()
