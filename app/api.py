import os
import json
import uuid
import time
from typing import Optional
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from app.schemas import PetitionRequest, TriageRunStatus, validate_and_guard
from app.storage import get_redis_client, FallbackRedis

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="NIT Rourkela Academic Petition Triage Agent", version="0.4.0")
redis_client = get_redis_client()

QUEUE_KEY = "triage:job_queue"
RUN_PREFIX = "triage:run:"

_cached_graph = None

def get_graph():
    global _cached_graph
    if _cached_graph is None:
        from app.corpus import NITRCorpusTools
        from app.agents import PetitionTriageGraph
        tools = NITRCorpusTools(pdf_path=os.getenv("PDF_PATH", "backUp.pdf"))
        _cached_graph = PetitionTriageGraph(tools=tools)
    return _cached_graph

def process_job_local(run_id: str, corr_id: str, payload_dict: dict):
    t0 = time.perf_counter()
    try:
        graph = get_graph()
        res = graph.run_petition(payload_dict)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_status = TriageRunStatus(
            run_id=run_id,
            correlation_id=corr_id,
            status="completed" if res["route"] != "escalate" else "escalated",
            route=res["route"],
            referred_to=res["referred_to"],
            citations=res["verified_citations"],
            rationale=res["rationale"],
            elapsed_ms=elapsed
        )
    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        final_status = TriageRunStatus(
            run_id=run_id,
            correlation_id=corr_id,
            status="escalated",
            route="escalate",
            errors=[f"Worker exception: {str(exc)}"],
            rationale="Run failed mid-execution; escalated to human officer without confident answer.",
            elapsed_ms=elapsed
        )
    redis_client.set(f"{RUN_PREFIX}{run_id}", final_status.model_dump_json(), ex=3600)

@app.post("/v1/petitions", response_model=TriageRunStatus, status_code=202)
def submit_petition(
    payload: PetitionRequest,
    background_tasks: BackgroundTasks,
    x_correlation_id: Optional[str] = Header(default=None)
):
    run_id = f"run-{uuid.uuid4().hex[:10]}"
    corr_id = x_correlation_id if isinstance(x_correlation_id, str) and x_correlation_id else f"corr-{uuid.uuid4().hex[:8]}"

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

    # If running in local fallback mode without an external worker process, execute via background task
    if isinstance(redis_client, FallbackRedis):
        background_tasks.add_task(process_job_local, run_id, corr_id, payload.model_dump())

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
