"""Start the web backend.

Run from the repository `server/` directory on the GPU server:

    python run.py --host 0.0.0.0 --port 8000

The scientific package must be importable (pip install -e . from the repo root).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    parser = argparse.ArgumentParser(description="Enzora web backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
