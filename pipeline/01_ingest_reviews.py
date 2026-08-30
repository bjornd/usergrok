"""Load data/raw_reviews.json into the reviews table (idempotent upsert on external_key)."""
from __future__ import annotations
import json
from pathlib import Path

from common import connect, ROOT


def parse_date(s):
    if not s:
        return None
    s = s.strip()
    # ISO (2026-06-19) or US (7/20/2026)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


def main():
    reviews = json.loads((ROOT / "data" / "raw_reviews.json").read_text())
    print(f"loaded {len(reviews)} reviews from disk")
    inserted = updated = 0
    with connect() as conn:
        for r in reviews:
            res = conn.execute(
                """
                insert into reviews (source, product, external_key, title, body,
                    like_text, dislike_text, problems_text, rating, author,
                    published_at, page_url, snapshot_ts)
                values ('g2', 'notion', %(external_key)s, %(title)s, %(body)s,
                    %(like_text)s, %(dislike_text)s, %(problems_text)s, %(rating)s, %(author)s,
                    %(published_at)s, %(page_url)s, %(snapshot_ts)s)
                on conflict (external_key) do update set
                    title = excluded.title, body = excluded.body,
                    like_text = excluded.like_text, dislike_text = excluded.dislike_text,
                    problems_text = excluded.problems_text, rating = excluded.rating
                returning (xmax = 0) as inserted
                """,
                {**r, "published_at": parse_date(r.get("published_at"))},
            )
            if res.fetchone()[0]:
                inserted += 1
            else:
                updated += 1
        conn.commit()
        n = conn.execute("select count(*) from reviews").fetchone()[0]
    print(f"inserted={inserted} updated={updated} | reviews table now has {n} rows")


if __name__ == "__main__":
    main()
