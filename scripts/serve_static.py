"""Serve the exported static_site/ folder for local preview.

The static build is plain files, so any web server works — this exists only so the dev
harness can launch it on an assigned port (see .claude/launch.json). Reading the JSON
files needs http://, not file://, because browsers block fetch on file URLs.
"""
import functools
import http.server
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "static_site"))
    port = int(os.environ.get("PORT", "8318"))
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
