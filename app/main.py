"""FastAPI backend for the UserGrok feedback-analysis demo.

Reads the clustering results out of Supabase (Postgres) and serves them to the
single-page frontend. Read-only; all the heavy lifting happened in the pipeline.
"""
from __future__ import annotations
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from common import connect, CATEGORIES  # noqa: E402

app = FastAPI(title="UserGrok — Feedback Analysis Demo")
STATIC = Path(__file__).resolve().parent / "static"

CATEGORY_LABELS = {
    "pain_point": "Pain Points",
    "praise": "Praise",
}


@app.get("/api/summary")
def summary():
    with connect() as conn:
        n_reviews = conn.execute("select count(*) from reviews").fetchone()[0]
        n_quotes = conn.execute("select count(*) from quotes").fetchone()[0]
        by_cat = dict(conn.execute(
            "select category, count(*) from quotes group by category").fetchall())
        runs = conn.execute(
            """
            select distinct on (category) category, id, metrics, params, created_at
            from cluster_runs order by category, created_at desc
            """
        ).fetchall()
    run_info = []
    for cat, rid, metrics, params, created in runs:
        m = metrics or {}
        run_info.append({
            "category": cat, "label": CATEGORY_LABELS.get(cat, cat), "run_id": rid,
            "n_clusters": m.get("n_clusters"), "n_train": m.get("n_train"),
            "n_test": m.get("n_test"), "noise_fraction": m.get("noise_fraction"),
            "params": params,
        })
    return {
        "reviews": n_reviews,
        "quotes": n_quotes,
        "quotes_by_category": {c: by_cat.get(c, 0) for c in CATEGORIES},
        "categories": [{"key": c, "label": CATEGORY_LABELS[c]} for c in CATEGORIES],
        "runs": run_info,
    }


@app.get("/api/category/{category}")
def category(category: str):
    if category not in CATEGORIES:
        raise HTTPException(404, "unknown category")
    with connect() as conn:
        run = conn.execute(
            """select id, params, metrics, model_path, created_at from cluster_runs
               where category = %s order by created_at desc limit 1""",
            (category,),
        ).fetchone()
        if not run:
            return {"category": category, "label": CATEGORY_LABELS[category],
                    "run": None, "clusters": [], "points": []}
        run_id, params, metrics, model_path, created = run

        # Rank themes by how many distinct reviewers raised them, not by quote count:
        # one effusive reviewer contributing four quotes should not outrank four people
        # independently hitting the same wall. avg_rating gives the severity read.
        clusters = [
            {"cluster_id": cid, "label": label, "summary": summ, "size": size,
             "mixed": mixed,
             "n_reviews": nrev, "avg_rating": round(float(avg), 2) if avg is not None else None}
            for cid, label, summ, size, mixed, nrev, avg in conn.execute(
                """select c.cluster_id, c.label, c.summary, c.size, c.mixed,
                          count(distinct q.review_id) as n_reviews,
                          avg(r.rating) as avg_rating
                     from clusters c
                     left join quote_clusters qc
                       on qc.run_id = c.run_id and qc.cluster_id = c.cluster_id
                     left join quotes q on q.id = qc.quote_id
                     left join reviews r on r.id = q.review_id
                    where c.run_id = %s
                    group by c.cluster_id, c.label, c.summary, c.size, c.mixed
                    order by (c.cluster_id < 0), count(distinct q.review_id) desc, c.size desc""",
                (run_id,),
            ).fetchall()
        ]

        points = [
            {
                "quote_id": qid, "text": text, "cluster_id": cid, "split": split,
                "assigned_by": how,
                "probability": round(prob, 3) if prob is not None else None,
                "x": x, "y": y,
                "review_title": rtitle, "review_rating": rrating, "review_author": rauthor,
                "review_url": rurl,
            }
            for (qid, text, cid, split, how, prob, x, y,
                 rtitle, rrating, rauthor, rurl) in conn.execute(
                """
                select q.id, q.text, qc.cluster_id, qc.split, qc.assigned_by,
                       qc.probability, qc.x, qc.y,
                       r.title, r.rating, r.author, r.page_url
                from quote_clusters qc
                join quotes q on q.id = qc.quote_id
                join reviews r on r.id = q.review_id
                where qc.run_id = %s
                order by qc.cluster_id, qc.split
                """,
                (run_id,),
            ).fetchall()
        ]
    return {
        "category": category, "label": CATEGORY_LABELS[category],
        "run": {"id": run_id, "params": params, "metrics": metrics,
                "model_path": model_path, "created_at": str(created)},
        "clusters": clusters, "points": points,
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=STATIC), name="static")
