"""Start the web backend.

Run from the repository root with the configured model environments:

    python src/Enzora_Software_Webs/utils/server/run.py --host 127.0.0.1 --port 8000

The launcher adds src/ to the Python search path.
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
