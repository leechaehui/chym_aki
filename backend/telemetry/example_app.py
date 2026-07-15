import asyncio
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import math

from telemetry.middleware import TelemetryMiddleware
from telemetry.streamer import streamer
from telemetry.session import create_user_session

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Telemetry Test App")
app.add_middleware(TelemetryMiddleware)

@app.on_event("startup")
async def startup_event():
    # Start the background streamer workers
    # NOTE: ensure your TELEMETRY_DB_DSN is valid and DB is up
    # await streamer.start() # Commented out for local test without real DB
    pass

@app.on_event("shutdown")
async def shutdown_event():
    await streamer.stop()

class LoginRequest(BaseModel):
    user_id: str

@app.post("/login")
async def login(req: LoginRequest):
    # Dummy login without real DB pool for demonstration
    # session_data = await create_user_session(streamer.db_pool, req.user_id, "127.0.0.1", "curl/7.68")
    return {"message": f"Logged in user {req.user_id}"}

@app.get("/heavy-task")
async def heavy_task(count: int = 100000):
    # Simulate CPU work
    total = 0
    for i in range(count):
        total += math.sqrt(i)
    return {"result": total}

if __name__ == "__main__":
    uvicorn.run("example_app:app", host="0.0.0.0", port=8000, reload=True)
