import os
import json
import uuid
import redis
from fastapi import FastAPI, Header, HTTPException
from typing import Optional
from app.schemas import PetitionRequest, TriageRunStatus, validate_and_guard

app = FastAPI(title="NIT Rourkela Academic Petition Triage Agent", version="0.4.0")
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)

QUEUE_KEY = "triage:job_queue"
RUN_PREFIX = "triage:run:"

@app.post("/v1/petitions", response_model=TriageRunStatus, status_code=202)
def submit_petition(
    payload: PetitionRequest,
    x_correlation_id: Optional[str] = Header(default=None)
):
    run_id = f"run-{uuid.uuid4().hex[:10]}"
    corr_id = x_correlation_id or f"corr-{uuid.uuid4().hex[:8]}"

    # Role 3 & 4 Check: Quarantine invalid data or block prompt injections before touching the queue
    verdict, errors = validate_and_guard(payload)
    if verdict in ("quarantine", "blocked"):
        status_obj = TriageRunStatus(
            run_id=run_id,
            correlation_id=corr_id,
            status="quarantined" if verdict == "quarantine" else "blocked",
            route=verdict,
            errors=errors,
            rationale="Request halted at API boundary; agent worker was never invoked."
        )
        redis_client.set(f"{RUN_PREFIX}{run_id}", status_obj.model_dump_json(), ex=3600)
        return status_obj

    # Valid request -> enqueue for LangGraph worker
    status_obj = TriageRunStatus(
        run_id=run_id,
        correlation_id=corr_id,
        status="queued"
    )
    redis_client.set(f"{RUN_PREFIX}{run_id}", status_obj.model_dump_json(), ex=3600)
    job_envelope = {
        "run_id": run_id,
        "correlation_id": corr_id,
        "payload": payload.model_dump()
    }
    redis_client.lpush(QUEUE_KEY, json.dumps(job_envelope))
    return status_obj

@app.get("/v1/petitions/{run_id}", response_model=TriageRunStatus)
def get_petition_status(run_id: str):
    raw = redis_client.get(f"{RUN_PREFIX}{run_id}")
    if not raw:
        raise HTTPException(status_code=404, detail=f"Run ID {run_id} not found.")
    return TriageRunStatus.model_validate_json(raw)

@app.get("/healthz")
def healthcheck():
    return {"status": "ok", "slo_p95_target_seconds": 45}