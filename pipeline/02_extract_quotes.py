"""Extract categorized verbatim quotes from each review with the claude CLI.

For every review we ask the LLM to pull short, verbatim quotes and tag each as
pain_point / praise, one concern per quote. Idempotent: reviews that already have
quotes are skipped. Runs in batches to amortize the CLI's fixed prompt overhead.
"""
from __future__ import annotations
import re, sys

from common import connect, claude_json, CATEGORIES

BATCH = 12

PROMPT_HEADER = """You are analyzing user reviews of the product Notion. The goal is to find
the themes a product team should act on: what to fix first, and what users love that the
team should double down on.

For EACH review below, extract short verbatim quotes that fall into exactly one of:
- "pain_point": anything the user is unhappy about — broken, slow, confusing, missing, or
  limited. Do NOT distinguish bugs from feature requests: "offline mode is broken" and
  "I wish it had offline mode" are the same concern and both are pain_point.
- "praise": something the user explicitly likes or values.

Rules:
- A quote MUST be copied verbatim from the review text (an exact substring), trimmed to the
  relevant clause or sentence (roughly 4-25 words). Do not paraphrase.
- ONE CONCERN PER QUOTE. If a sentence covers two different topics, emit two quotes, each
  trimmed to its own topic. Never emit a quote spanning two unrelated complaints.
- Each quote must be understandable on its own, without the surrounding review. Prefer the
  span that names the specific subject ("the mobile app is slow with large databases")
  over a vague fragment ("it's slow", "this is annoying", "needs work").
- Skip generic filler with no actionable subject ("it's good", "nothing", "n/a").
- Extract 0 or more quotes per review; a review may yield quotes of both categories.
- Ignore the section labels ("What they like:", etc.) — quote the user's own words.

Return ONLY a JSON object mapping each review's id (as a string) to an array of
{"quote": "...", "category": "..."} objects. Example:
{"12": [{"quote":"crashes when I paste large tables","category":"pain_point"},
        {"quote":"the templates save me hours every week","category":"praise"}],
 "13": []}

Reviews:
"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def build_prompt(batch):
    parts = [PROMPT_HEADER]
    for rid, body in batch:
        parts.append(f"\n### review {rid}\n{body}\n")
    return "".join(parts)


def main():
    # Optional sharding so several workers can run concurrently:
    #   python 02_extract_quotes.py --shard 0 --of 3
    # Each LLM batch takes minutes, so a single process over ~180 reviews is slow.
    # Shards are disjoint by review id, and the "already has quotes" filter makes any
    # worker safe to restart.
    shard, of = 0, 1
    if "--shard" in sys.argv:
        shard = int(sys.argv[sys.argv.index("--shard") + 1])
    if "--of" in sys.argv:
        of = int(sys.argv[sys.argv.index("--of") + 1])

    with connect() as conn:
        rows = conn.execute(
            """
            select r.id, r.body
            from reviews r
            where not exists (select 1 from quotes q where q.review_id = r.id)
              and mod(r.id, %s) = %s
            order by r.id
            """,
            (of, shard),
        ).fetchall()
        tag = f"[shard {shard}/{of}] " if of > 1 else ""
        print(f"{tag}{len(rows)} reviews need extraction")
        if not rows:
            return

        total_quotes = 0
        nonverbatim = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            bodies = {str(rid): body for rid, body in batch}
            try:
                result = claude_json(build_prompt(batch))
            except Exception as e:
                print(f"  batch {i//BATCH}: LLM/parse failed: {e}", file=sys.stderr)
                continue
            if not isinstance(result, dict):
                print(f"  batch {i//BATCH}: unexpected result type {type(result)}", file=sys.stderr)
                continue

            batch_quotes = 0
            for rid_str, items in result.items():
                body = bodies.get(str(rid_str))
                if body is None or not isinstance(items, list):
                    continue
                body_n = norm(body)
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    quote = (it.get("quote") or "").strip()
                    cat = (it.get("category") or "").strip()
                    if not quote or cat not in CATEGORIES:
                        continue
                    if norm(quote) not in body_n:
                        nonverbatim += 1  # keep it anyway, but track drift
                    conn.execute(
                        """insert into quotes (review_id, category, text)
                           values (%s, %s, %s)
                           on conflict (review_id, category, text) do nothing""",
                        (int(rid_str), cat, quote),
                    )
                    batch_quotes += 1
            conn.commit()
            total_quotes += batch_quotes
            print(f"  {tag}batch {i//BATCH + 1}/{(len(rows)+BATCH-1)//BATCH}: +{batch_quotes} quotes "
                  f"(reviews {batch[0][0]}..{batch[-1][0]})", flush=True)

        # summary
        counts = conn.execute(
            "select category, count(*) from quotes group by category order by category"
        ).fetchall()
    print(f"\ninserted ~{total_quotes} quotes ({nonverbatim} not exact-substring, kept anyway)")
    print("per category:", dict(counts))


if __name__ == "__main__":
    main()
