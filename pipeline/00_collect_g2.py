"""Collect a sample of Notion reviews from G2 via the Wayback Machine.

The live G2 site rate-limits automated navigation, so for a reproducible demo we
pull archived snapshots of the public reviews page from archive.org. Each snapshot
is a full server-rendered reviews page (~10 reviews as schema.org microdata).
Snapshots from different dates surface different reviews; we dedup by G2's stable
review id.

Output:
    data/raw/chunk_<ts>.json   one file per snapshot (provenance)
    data/raw_reviews.json      merged, deduped list of reviews (pipeline input)

Network egress goes through curl (the Python SSL trust store is broken by a local
proxy on this machine, but curl's works).
"""
from __future__ import annotations
import json, re, subprocess, sys, time, html as htmllib
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "html_cache"
OUT = ROOT / "data" / "raw_reviews.json"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRODUCT = "notion"
TARGET = 200                      # stop once we have this many unique reviews
CDX = ("http://web.archive.org/cdx/search/cdx?url=g2.com/products/notion/reviews*"
       "&output=json&from=20240101&to=20260721"
       "&filter=statuscode:200&filter=mimetype:text/html&collapse=digest&limit=400")


def curl(url: str, timeout: int = 90) -> str:
    r = subprocess.run(
        ["curl", "-s", "--compressed", "--max-time", str(timeout),
         "-A", "Mozilla/5.0 (research; feedback-demo)", url],
        capture_output=True, text=True)
    return r.stdout


def list_snapshots() -> list[tuple[str, str]]:
    rows = json.loads(curl(CDX, 60))
    hdr, data = rows[0], rows[1:]
    ti, oi = hdr.index("timestamp"), hdr.index("original")
    snaps = [(r[ti], r[oi]) for r in data]
    # newest first — recent snapshots have the current review schema & fresher reviews
    snaps.sort(key=lambda x: x[0], reverse=True)
    return snaps


def clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\r", "", s or "")).strip()


def parse_snapshot(html: str, ts: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    reviews = soup.select('[itemprop="review"]')
    out = []
    for r in reviews:
        # stable G2 id — appears in a permalink (/reviews/notion-review-N) or in the
        # response link (/survey_responses/notion-review-N/...) depending on snapshot era.
        key = None
        for a in r.select('a[href*="notion-review-"]'):
            m = re.search(r"(notion-review-\d+)", a.get("href", ""))
            if m:
                key = m.group(1)
                break
        if not key:
            continue

        # the review headline is an itemprop="name" with class l2 (author names also
        # use itemprop="name", so we can't just take the first one).
        title_el = r.select_one('[itemprop="name"].l2')
        if not title_el:
            for el in r.select('[itemprop="name"]'):
                t = el.get_text(strip=True)
                if '"' in t or "”" in t:
                    title_el = el
                    break
        title = clean(title_el.get_text()) if title_el else None
        if title:
            title = title.strip().strip('"“”').strip()

        # rating: prefer explicit ratingValue, fall back to stars-N class (N/2)
        rating = None
        rv = r.select_one('[itemprop="ratingValue"]')
        if rv and rv.get("content"):
            try: rating = float(rv["content"])
            except ValueError: pass
        if rating is None:
            stars = r.select_one('[class*="stars-"]')
            if stars:
                m = re.search(r"stars-(\d+)", " ".join(stars.get("class", [])))
                if m: rating = int(m.group(1)) / 2.0

        date = None
        dp = r.select_one('[itemprop="datePublished"]')
        if dp:
            date = dp.get("content") or dp.get_text(strip=True)
        if not date:
            t = r.select_one("time[datetime]")
            date = t["datetime"] if t else None

        author_el = r.select_one('[itemprop="author"]')
        author = clean(author_el.get_text(" ")) if author_el else None
        if author:
            author = author.split("\n")[0][:80]

        # review body: the like / dislike / problems answers. The markup differs across
        # snapshot eras (old: <div class="l5"> + <p class="formatted-text">; new:
        # <section> + <div class="elv-font-bold"> + <p>), so we split on the *heading
        # text*, which is stable in both.
        body_el = r.select_one('[itemprop="reviewBody"]')
        like = dislike = problems = ""
        if body_el:
            full = re.sub(r"\n{2,}", "\n", body_el.get_text("\n")).strip()
            full = re.sub(r"Review collected by and hosted on G2\.com\.?", " ", full)
            labels = [
                ("like", re.compile(r"What do you like best about[^?\n]*\?", re.I)),
                ("dislike", re.compile(r"What do you dislike about[^?\n]*\?", re.I)),
                ("problems", re.compile(r"What problems is[^?\n]*\?[^\n]*", re.I)),
            ]
            hits = []
            for name, rx in labels:
                m = rx.search(full)
                if m:
                    hits.append((m.start(), m.end(), name))
            hits.sort()
            sections = {}
            for i, (s, e, name) in enumerate(hits):
                end = hits[i + 1][0] if i + 1 < len(hits) else len(full)
                sections[name] = clean(full[e:end])
            like, dislike, problems = (sections.get("like", ""),
                                       sections.get("dislike", ""),
                                       sections.get("problems", ""))
            if not (like or dislike or problems):  # unrecognized heading layout: keep all
                like = clean(full)

        combined = "\n\n".join(
            f"{lbl}: {txt}" for lbl, txt in
            [("What they like", like), ("What they dislike", dislike),
             ("Problems it solves", problems)] if txt)
        if not combined.strip():
            continue

        out.append({
            "external_key": key,
            "title": title,
            "author": author,
            "rating": rating,
            "published_at": date,
            "like_text": like,
            "dislike_text": dislike,
            "problems_text": problems,
            "body": combined,
            "page_url": f"https://www.g2.com/products/{PRODUCT}/reviews/{key}",
            "snapshot_ts": ts,
        })
    return out


def main():
    snaps = list_snapshots()
    print(f"[cdx] {len(snaps)} snapshots available", flush=True)
    seen: dict[str, dict] = {}
    for i, (ts, url) in enumerate(snaps):
        wb = f"https://web.archive.org/web/{ts}id_/{url}"
        cache = CACHE_DIR / f"{ts}.html"
        if cache.exists() and cache.stat().st_size > 5000:
            html = cache.read_text(encoding="utf-8", errors="replace")
        else:
            html = curl(wb)
            if len(html) > 5000:            # cache only non-throttled responses
                cache.write_text(html, encoding="utf-8")
            time.sleep(1.0)                 # be polite to archive.org
        try:
            revs = parse_snapshot(html, ts)
        except Exception as e:
            print(f"[{i:3}] {ts} parse error: {e}", flush=True)
            continue
        (RAW_DIR / f"chunk_{ts}.json").write_text(json.dumps(revs, ensure_ascii=False, indent=1))
        new = 0
        for rv in revs:
            if rv["external_key"] not in seen:
                seen[rv["external_key"]] = rv
                new += 1
        print(f"[{i:3}] {ts} {url.split('reviews')[-1][:24]:24} parsed={len(revs):2} new={new:2} total={len(seen)}", flush=True)
        if len(seen) >= TARGET:
            print(f"[done] reached target {TARGET}", flush=True)
            break

    merged = list(seen.values())
    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    # quick rating distribution
    from collections import Counter
    dist = Counter(round(r["rating"]) if r["rating"] else "?" for r in merged)
    print(f"\n[saved] {len(merged)} unique reviews -> {OUT}")
    print(f"[ratings] {dict(sorted(dist.items(), key=lambda x: str(x[0])))}")


if __name__ == "__main__":
    main()
