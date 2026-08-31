"""Give each HDBSCAN cluster a human-readable label + summary via the claude CLI.

For each category's latest run, gather representative quotes per cluster and ask the
LLM for a short label and one-sentence summary. Noise (cluster -1) keeps a fixed label.
"""
from __future__ import annotations
import json

from common import connect, claude_json, CATEGORIES

MAX_QUOTES_PER_CLUSTER = 8
CLUSTERS_PER_CALL = 20


def latest_run(conn, category):
    row = conn.execute(
        "select id from cluster_runs where category = %s order by created_at desc limit 1",
        (category,),
    ).fetchone()
    return row[0] if row else None


def cluster_quotes(conn, run_id):
    """Return {cluster_id: [quote text, ...]} for non-noise clusters, highest-probability first."""
    rows = conn.execute(
        """
        select qc.cluster_id, q.text, qc.probability
        from quote_clusters qc join quotes q on q.id = qc.quote_id
        where qc.run_id = %s and qc.cluster_id >= 0
        order by qc.cluster_id, qc.probability desc
        """,
        (run_id,),
    ).fetchall()
    out: dict[int, list[str]] = {}
    for cid, text, _ in rows:
        out.setdefault(cid, [])
        if len(out[cid]) < MAX_QUOTES_PER_CLUSTER:
            out[cid].append(text)
    return out


def build_prompt(category, clusters):
    human = "pain point" if category == "pain_point" else category.replace("_", " ")
    parts = [
        f'These are clusters of "{human}" quotes extracted from Notion user reviews. '
        "A product team will use these to decide what to fix and what to invest in.\n\n"
        "For each cluster give:\n"
        '- "label": 2-5 words naming the SPECIFIC shared theme. Name the actual subject '
        '("PDF export loses formatting"), never a generic bucket ("various issues", '
        '"miscellaneous", "general problems").\n'
        '- "summary": one sentence on what users are saying.\n'
        '- "mixed": true if the quotes actually cover two or more unrelated concerns '
        "rather than one, false otherwise. Be honest — this flags clusters for review.\n\n"
        "Return ONLY a JSON object mapping each cluster id (string) to "
        '{"label": "...", "summary": "...", "mixed": false}.\n\n'
    ]
    for cid, quotes in sorted(clusters.items()):
        parts.append(f"### cluster {cid}\n")
        parts.extend(f"- {q}\n" for q in quotes)
        parts.append("\n")
    return "".join(parts)


def main():
    with connect() as conn:
        for cat in CATEGORIES:
            run_id = latest_run(conn, cat)
            if run_id is None:
                print(f"[{cat}] no run — skipping")
                continue
            clusters = cluster_quotes(conn, run_id)
            if not clusters:
                print(f"[{cat}] run {run_id}: no non-noise clusters to label")
                continue
            # Batch the clusters: a fine-grained run can produce 70+ clusters, and asking
            # for all of them in one prompt makes a request large enough to time out or
            # come back truncated. Chunks also fail independently.
            items = sorted(clusters.items())
            labels = {}
            for i in range(0, len(items), CLUSTERS_PER_CALL):
                chunk = dict(items[i:i + CLUSTERS_PER_CALL])
                try:
                    got = claude_json(build_prompt(cat, chunk))
                except Exception as e:
                    print(f"[{cat}] chunk {i // CLUSTERS_PER_CALL + 1} failed: {e}")
                    continue
                if isinstance(got, dict):
                    labels.update(got)

            n = 0
            mixed = []
            for cid_str, meta in (labels or {}).items():
                if not isinstance(meta, dict):
                    continue
                try:
                    cid = int(cid_str)
                except (TypeError, ValueError):
                    continue
                is_mixed = bool(meta.get("mixed"))
                conn.execute(
                    """update clusters set label = %s, summary = %s, mixed = %s
                       where run_id = %s and cluster_id = %s""",
                    (meta.get("label"), meta.get("summary"), is_mixed, run_id, cid),
                )
                if is_mixed:
                    mixed.append(f"{cid}:{meta.get('label')}")
                n += 1
            conn.commit()
            print(f"[{cat}] run {run_id}: labeled {n} clusters")
            if mixed:
                # surfaced, not silently fixed — a mixed cluster means the clustering
                # params deserve another look, and hiding it would fake the quality
                print(f"[{cat}] LLM flagged as covering >1 concern: {', '.join(mixed)}")


if __name__ == "__main__":
    main()
