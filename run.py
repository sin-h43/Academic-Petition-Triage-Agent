"""
Academic Petition Triage Agent - Main Launcher
Usage:
  python run.py demo    # Runs the local end-to-end benchmark test cases
  python run.py corpus  # Queries and verifies the NITR regulation corpus index
  python run.py api     # Starts the FastAPI server (with in-memory or Redis fallback)
  python run.py worker  # Starts the background queue worker
"""
import sys

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "demo"

    if mode in ("demo", "test"):
        from test_e2e_local import run_demo
        run_demo()
    elif mode == "corpus":
        from app.corpus import NITRCorpusTools
        print("[Launcher] Initializing NITR Corpus Tools...")
        tools = NITRCorpusTools()
        print(f"[Launcher] Loaded {len(tools.chunks)} regulation chunks.")
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "attendance condonation medical leave"
        print(f"[Launcher] Running search for: '{query}'")
        for hit in tools.search_corpus(query, top_k=3):
            print(f"  - [{hit['clause_id']}] (p.{hit['page_number']}, score={hit['score']}): {hit['content'][:120]}...")
    elif mode == "api":
        import uvicorn
        print("[Launcher] Starting FastAPI server on http://127.0.0.1:8000...")
        uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
    elif mode == "worker":
        from app.worker import start_worker
        start_worker()
    else:
        print("Unknown mode. Use: 'python run.py [demo|corpus|api|worker]'")

if __name__ == "__main__":
    main()

