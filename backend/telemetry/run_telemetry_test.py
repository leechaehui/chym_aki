import asyncio
import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from telemetry.streamer import streamer
from telemetry.memory_guard import get_memory_usage
from telemetry.fallback import log_dir

async def main():
    print("="*50)
    print("1. Memory Guard Verification")
    print("="*50)
    print(f"Current System Memory Usage: {get_memory_usage()}%")
    print("\n" + "="*50)
    print("2. Hybrid Routing & DLQ Fallback Verification")
    print("="*50)
    print("Starting Streamer (Mocking DB Connection Failure to trigger 3-Tier Fallback...)")
    
    # Mock DB Pool to force failure
    class MockPool:
        def acquire(self):
            class MockConnContext:
                async def __aenter__(self):
                    raise Exception("Mocked ConnectionRefusedError: DB is down!")
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            return MockConnContext()

    streamer.db_pool = MockPool()
    for worker in streamer.workers:
        worker.start(streamer.db_pool)

    print("Pushing 1000 logs into streamer queues...")
    for i in range(1000):
        streamer.push({
            "request_id": f"req-{i}",
            "user_id": f"user-{i%10}",
            "endpoint": "/test",
            "method": "GET",
            "status_code": 200,
            "latency_ms": 12.5,
            "request_time": "2026-06-19T00:00:00Z"
        })
    
    # Print queue status
    for worker in streamer.workers:
        print(f"Worker {worker.worker_id} Queue Size: {worker.queue.qsize()}")
        
    print("\nWorkers processing batch/chunk inserts (Expect exponential backoff, then DB DLQ, then File Fallback)...")
    print("Waiting 5 seconds for processing...")
    await asyncio.sleep(5)
    
    await streamer.stop()
    
    print("\n" + "="*50)
    print("3. Tertiary Fallback File Check")
    print("="*50)
    fallback_file = log_dir / "telemetry_fallback.jsonl"
    if fallback_file.exists():
        with open(fallback_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Fallback file created successfully at: {fallback_file}")
        print(f"Total fallback entries written: {len(lines)}")
        if lines:
            print(f"Sample Entry (first line):\n{lines[0][:200]}...")
    else:
        print("Fallback file was not created.")

if __name__ == "__main__":
    asyncio.run(main())
