"""Entry point for the demo web app.

Reads the port from the PORT environment variable so the dev-server harness can assign
one (uvicorn's CLI only takes --port, which would hardcode it and collide whenever
something else already holds the port). Falls back to 8317 for a plain manual run.
"""
import os
import sys
from pathlib import Path

# Running this file directly puts app/ on sys.path rather than the repo root, so
# "app.main" would not be importable. Add the repo root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8317")),
    )
