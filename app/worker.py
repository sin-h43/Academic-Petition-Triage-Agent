import os
import time
import json
from app.corpus import NITRCorpusTools
from app.agents import PetitionTriageGraph
from app.schemas import TriageRunStatus
from app.storage import get_redis_client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

QUEUE_KEY = "triage:job_queue"
RUN_PREFIX = "triage:run:"

def start_worker():
    r = get_redis_client()
    tools = NITRCorpusTools(pdf_path=os.getenv("PDF_PATH", "backUp.pdf"))
    agent_graph = PetitionTriageGraph(tools=tools)
    print("[Worker] Initialized FAISS index and LangGraph workflow. Listening for jobs...")

    while True:
        item = r.brpop(QUEUE_KEY, timeout=5)
        if not item:
            continue
        _, raw_job = item
        job = json.loads(raw_job)
        run_id = job["run_id"]
        corr_id = job["correlation_id"]
        t0 = time.perf_counter()

        try:
            res = agent_graph.run_petition(job["payload"])
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
            # Role 2 Resilience rule: If an agent fails mid-run, mark incomplete & escalate
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

        r.set(f"{RUN_PREFIX}{run_id}", final_status.model_dump_json(), ex=3600)
        print(f"[Worker] Completed {run_id} (corr={corr_id}) -> route={final_status.route} in {elapsed}ms")

if __name__ == "__main__":
    start_worker()
