"""Embed quote text into 384-dim vectors (all-MiniLM-L6-v2) and store in quotes.embedding."""
from __future__ import annotations
import numpy as np

from common import connect

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    with connect() as conn:
        rows = conn.execute(
            "select id, text from quotes where embedding is null order by id"
        ).fetchall()
        print(f"{len(rows)} quotes need embeddings")
        if not rows:
            return

        from sentence_transformers import SentenceTransformer
        print(f"loading {MODEL_NAME} ...")
        model = SentenceTransformer(MODEL_NAME)

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        vecs = model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        ).astype(np.float32)
        print(f"encoded {vecs.shape[0]} quotes -> dim {vecs.shape[1]}")

        for qid, vec in zip(ids, vecs):
            conn.execute("update quotes set embedding = %s where id = %s", (vec, qid))
        conn.commit()

        n = conn.execute("select count(*) from quotes where embedding is not null").fetchone()[0]
    print(f"done: {n} quotes now have embeddings")


if __name__ == "__main__":
    main()
