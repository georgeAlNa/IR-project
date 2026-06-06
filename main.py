import argparse
import subprocess
import sys
import time

import uvicorn
from fastapi import FastAPI

from api.indexing import router as indexing_router
from api.evaluation import router as evaluation_router
from api.matching_ranking import router as matching_ranking_router
from api.query_refinement import router as query_refinement_router
from api.preprocessing import router as preprocessing_router

app = FastAPI(
    title="IR Microservices Gateway",
    version="1.0.0",
    description="Single entrypoint that serves preprocessing, indexing/representation, matching/ranking, evaluation, and query refinement services.",
)

app.include_router(indexing_router)
app.include_router(evaluation_router)
app.include_router(matching_ranking_router)
app.include_router(query_refinement_router)
app.include_router(preprocessing_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "IR Microservices Gateway",
        "docs": "/docs",
        "web_ui": "http://127.0.0.1:8501",
        "preprocess_endpoint": "/preprocess",
        "index_endpoint": "/index",
        "represent_endpoint": "/represent",
        "search_endpoint": "/search",
        "evaluate_endpoint": "/evaluate",
        "refine_endpoint": "/refine",
    }


def _start_api_process(host: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
    )


def _start_web_process(host: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "web_ui/app.py",
            "--server.address",
            host,
            "--server.port",
            str(port),
        ]
    )


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def run_combined(api_host: str, api_port: int, ui_host: str, ui_port: int) -> None:
    api_process = _start_api_process(api_host, api_port)
    web_process = _start_web_process(ui_host, ui_port)

    print(f"API running on http://{api_host}:{api_port}")
    print(f"Web UI running on http://{ui_host}:{ui_port}")

    try:
        while True:
            if api_process.poll() is not None:
                raise RuntimeError("API process exited unexpectedly.")
            if web_process.poll() is not None:
                raise RuntimeError("Web UI process exited unexpectedly.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_process(api_process)
        _stop_process(web_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IR backend and frontend from one entrypoint.")
    parser.add_argument("--mode", choices=["all", "api", "web"], default="all")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args()

    if args.mode == "api":
        uvicorn.run("main:app", host=args.api_host, port=args.api_port, reload=False)
    elif args.mode == "web":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "web_ui/app.py",
                "--server.address",
                args.ui_host,
                "--server.port",
                str(args.ui_port),
            ],
            check=False,
        )
    else:
        run_combined(args.api_host, args.api_port, args.ui_host, args.ui_port)
