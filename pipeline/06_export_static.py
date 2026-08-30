"""Export a self-hosting static build of the demo — no backend, no database.

Calls the same functions the FastAPI app serves (rather than re-implementing their
queries, which would drift) and writes their output to plain JSON files, alongside a copy
of the frontend pointed at those files instead of /api.

Output:
    static_site/index.html
    static_site/data/summary.json
    static_site/data/<category>.json

Serve the folder with any static file server; see the README for the one-liner.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from app.main import summary, category, CATEGORY_LABELS  # noqa: E402
from common import CATEGORIES  # noqa: E402

OUT = ROOT / "static_site"
DATA = OUT / "data"
SRC_HTML = ROOT / "app" / "static" / "index.html"

# Injected ahead of the page's own script, which reads window.USERGROK_SOURCES if set.
SOURCES_SHIM = """<script>
// Static build: read the exported JSON files instead of the FastAPI backend.
window.USERGROK_SOURCES = {
  summary: 'data/summary.json',
  category: cat => 'data/' + cat + '.json',
};
</script>
"""


def write_json(path: Path, payload) -> int:
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return path.stat().st_size


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    DATA.mkdir(parents=True)

    total = write_json(DATA / "summary.json", summary())
    print(f"  data/summary.json{'':14} {total/1024:7.1f} KB")

    for cat in CATEGORIES:
        payload = category(cat)
        if payload.get("run") is None:
            print(f"  ! {cat}: no clustering run — exported empty")
        # created_at is a datetime; make the payload JSON-safe
        if payload.get("run") and payload["run"].get("created_at") is not None:
            payload["run"]["created_at"] = str(payload["run"]["created_at"])
        size = write_json(DATA / f"{cat}.json", payload)
        total += size
        print(f"  data/{cat}.json{'':{max(0, 20 - len(cat))}} {size/1024:7.1f} KB  "
              f"({len(payload.get('points', []))} points, {len(payload.get('clusters', []))} clusters)")

    html = SRC_HTML.read_text()
    marker = "<script>"
    assert marker in html, "could not find the page's script tag"
    html = html.replace(marker, SOURCES_SHIM + marker, 1)
    (OUT / "index.html").write_text(html)
    total += (OUT / "index.html").stat().st_size

    print(f"\n  static_site/ ready — {total/1024:.0f} KB total, "
          f"{len(list(DATA.glob('*.json')))} JSON files + index.html")
    print(f"  serve it:  python3 -m http.server -d {OUT.relative_to(ROOT)} 8000")


if __name__ == "__main__":
    main()
