"""Shared helpers for the UserGrok pipeline: DB connection + LLM (claude CLI) wrapper."""
from __future__ import annotations
import json, os, re, subprocess
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("pain_point", "praise")


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54422/postgres")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")


def connect() -> psycopg.Connection:
    """Open a psycopg connection with pgvector types registered."""
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    register_vector(conn)
    return conn


# --- LLM backend: the local, authenticated `claude` CLI (no API key needed) -------------

def claude(prompt: str, model: str | None = None, timeout: int = 600) -> str:
    """Run `claude -p` in JSON mode and return the model's text result.

    The CLI emits an envelope like {"type":"result","result":"<text>", ...}; we
    return the inner `result` string. Raises on non-zero exit or missing result.
    """
    model = model or CLAUDE_MODEL
    proc = subprocess.run(
        ["claude", "-p", prompt, "--model", model, "--output-format", "json"],
        capture_output=True, text=True, timeout=timeout,
        # the prompt is passed as an argument, so close stdin — otherwise the CLI waits
        # on it and aborts with "no stdin data received in 3s"
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (exit {proc.returncode}): {proc.stderr[:500]}")
    env = json.loads(proc.stdout)
    if env.get("is_error") or "result" not in env:
        raise RuntimeError(f"claude CLI returned error envelope: {str(env)[:400]}")
    return env["result"]


def extract_json(text: str):
    """Best-effort parse of a JSON array/object from an LLM response.

    Handles ```json fences and leading/trailing prose by locating the outermost
    bracket pair. Returns the parsed object, or raises json.JSONDecodeError.
    """
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Take the first complete JSON value and ignore anything after it. Models sometimes
    # append a second object or a trailing note, which plain json.loads rejects outright
    # with "Extra data" even though the payload we want is intact.
    for open_ch in ("{", "["):
        start = text.find(open_ch)
        if start != -1:
            try:
                return json.JSONDecoder().raw_decode(text, start)[0]
            except json.JSONDecodeError:
                pass
    # last resort: outermost bracket pair (handles trailing prose after the payload)
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON found", text, 0)


def claude_json(prompt: str, model: str | None = None, repair: bool = True):
    """Call the LLM and parse a JSON payload from its response, with one repair retry."""
    out = claude(prompt, model=model)
    try:
        return extract_json(out)
    except json.JSONDecodeError:
        if not repair:
            raise
        fixed = claude(
            "Convert the following to a single valid JSON value. Output ONLY the JSON, "
            "no prose, no code fences:\n\n" + out,
            model=model,
        )
        return extract_json(fixed)
