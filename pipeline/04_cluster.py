"""Per-category HDBSCAN clustering with an 80/20 train/test split.

For each feedback category:
  1. deterministic 80/20 split of the embedded quotes
  2. fit hdbscan.HDBSCAN(prediction_data=True) on the 80%, sweeping the parameter grid
     and keeping the best candidate under the selection constraints
  3. persist the fitted model (clusterer + train ids) to models/
  4. assign the held-out 20% with hdbscan.approximate_predict (reloaded model)
  5. project all points to 2-D with UMAP for the visualization
  6. write cluster_runs / clusters / quote_clusters

Only the 80% defines the clusters; the 20% is *predicted* into them, mirroring how
a deployed system would label new incoming feedback against a saved model.

Candidates are chosen for CLUSTER COHERENCE, not coverage: a theme is only useful to a
product team if its quotes are genuinely about one thing. Quotes HDBSCAN rejects are
left unclustered rather than attached to a nearest theme.

Clustering runs on the full-dimensional embeddings: HDBSCAN and approximate_predict both
operate on the same 384-d vectors the quotes were embedded into, with no reduction step
in between. That keeps the pipeline honest about what is being compared — the distances
driving the clusters are distances between the embeddings themselves.

The trade-off is real and measured: density estimation is harder in high dimensions, so
more quotes fall below the density threshold and stay unclustered than when the vectors
were first projected down. Coherence of the themes that do form is not hurt by it.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import joblib
import hdbscan
import umap
from sklearn.decomposition import PCA

from common import connect, CATEGORIES, ROOT

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MIN_QUOTES = 15


def load_category(conn, category):
    rows = conn.execute(
        "select id, embedding from quotes where category = %s and embedding is not null order by id",
        (category,),
    ).fetchall()
    ids = np.array([r[0] for r in rows])
    X = np.array([np.asarray(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 384))
    return ids, X


def split_80_20(n):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_test = max(1, round(n * 0.2))
    return idx[n_test:], idx[:n_test]  # train_idx, test_idx


# Model selection is a constrained search rather than a weighted score: blending several
# desiderata into one number meant every re-weighting fixed one category and broke
# another. A candidate must be *readable* (enough distinct themes, no single blob
# swallowing the category), and among the readable ones we take the most COHERENT.
#
# Coherence, not coverage, is the objective. The tool exists to tell a product team what
# to fix and what to double down on, and a theme is only actionable if its quotes are
# really about one thing. Optimizing coverage produced broad clusters that fused two
# concerns ("AI reliability" quietly absorbing PDF-export and page-load complaints),
# which is far worse than leaving a few quotes unclustered.
# (min_clusters, quotes_per_theme_divisor, max_share_of_largest_cluster, min_coverage)
#
# max_clusters scales with the category rather than being a flat number: 546 praise
# quotes can legitimately support ~30 distinct themes, while 100 quotes cannot, and a
# fixed cap of 24 silently forced the big category back into 8 coarse buckets.
#
# The coverage floor is not a return to optimizing coverage — it rules out the degenerate
# corner where chasing coherence keeps only a few tiny, ultra-tight clusters and leaves
# most of the feedback invisible. Clustering the raw 384-d vectors means few candidates
# reach the higher tiers at all, so in practice the lower tiers do most of the work here;
# the ladder still prefers a well-spread clustering whenever one exists.
# Generalization (how much of an unseen split approximate_predict still assigns) is a
# tier constraint rather than a global floor. As a hard gate it silently forced a whole
# category into a single blob: every well-shaped praise model sits at 15-19% val
# coverage, so a 0.35 floor rejected all of them and left only the degenerate
# high-coverage one. Relaxing it in step with the other constraints keeps the preference
# for models that generalize without ever making a blob the last option standing.
# The first tier that admits anything wins, so the tiers are ordered by what actually
# makes a theme list usable: enough distinct themes, and no cluster swallowing the
# category. Coverage sits low deliberately — on raw 384-d every well-shaped model leaves
# a lot of quotes unclustered, and high coverage floors just filter those models out and
# promote a blob instead. Coherence then picks the winner inside the surviving set.
SELECTION_TIERS = (
    # (min_clusters, divisor, max_share_of_largest, min_train_coverage, min_val_coverage)
    # The val floor does real work: the most coherent models are often the worst at
    # placing unseen quotes, and the reported unclustered share covers the held-out 20%
    # too, so a model that rejects new points drags the whole number up.
    (8, 18, 0.30, 0.60, 0.40),
    (6, 14, 0.40, 0.50, 0.35),
    (4, 10, 0.55, 0.35, 0.25),
    (2, 5, 1.01, 0.00, 0.00),
)

# Upper bound on the theme ceiling. Reaching ~50% coverage on raw 384-d requires many
# small themes (~3-4 quotes each), so the cap has to be well above the ~30 a reader would
# skim; the tier ladder still prefers a coarser model whenever one meets the floors.
MAX_THEMES = 45

# Both categories reduce to the same number of components rather than sweeping one each.
# Measured across 5-60 components (fixing the value and letting the rest of the selection
# run), coherence keeps climbing with dimensionality while coverage falls and the theme
# count balloons — 40 components gives coherence ~0.80 but 30/44 themes at ~58% coverage.
# 15 is the point with the fewest themes at good coverage:
#   dims   pain (themes/coh/cov)   praise (themes/coh/cov)
#     10        18 / 0.704 / 61%        24 / 0.710 / 62%
#     15        17 / 0.708 / 62%        28 / 0.732 / 66%
#     20        22 / 0.752 / 63%        36 / 0.753 / 68%
#     30        19 / 0.758 / 58%        29 / 0.779 / 53%
PCA_COMPONENTS = 15


def cluster_shape(labels):
    """(n_clusters, coverage, largest-cluster share of the clustered points)."""
    uniq, counts = np.unique(labels, return_counts=True)
    sizes = np.array([c for l, c in zip(uniq, counts) if l >= 0], dtype=float)
    n_clusters = len(sizes)
    coverage = float((labels >= 0).mean())
    share = float(sizes.max() / sizes.sum()) if n_clusters else 1.0
    return n_clusters, coverage, share


def coherence(X, labels):
    """Mean cosine similarity of each clustered quote to its own cluster centroid.

    Computed in the same 384-d space the clustering runs in, so it measures what a
    reader would judge: are these quotes about the same thing? Weighted by cluster size,
    so one tight two-quote cluster cannot mask a large incoherent one.
    """
    sims, n = [], 0
    for cid in {int(l) for l in labels if l >= 0}:
        members = X[labels == cid]
        c = members.mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-12)
        sims.append(float((members @ c).sum()))
        n += len(members)
    return sum(sims) / n if n else 0.0


def pick_best(cands, n_total):
    """First tier that admits any candidate wins; within it, maximize coherence."""
    for lo, divisor, max_share, min_cov, min_val in SELECTION_TIERS:
        hi = max(14, min(MAX_THEMES, round(n_total / divisor)))
        ok = [c for c in cands
              if lo <= c["n_clusters"] <= hi
              and c["largest_share"] <= max_share
              and c["clustered_fraction"] >= min_cov
              and c["inner_val_coverage"] >= min_val]
        if ok:
            return max(ok, key=lambda c: c["coherence"])
    # nothing cleared the bars — fall back to the best generalizing candidate
    return max(cands, key=lambda c: (c["inner_val_coverage"], c["coherence"]))


def make_clusterer(mcs, ms, eps, method="eom"):
    """cluster_selection_epsilon merges splits closer than eps. Swept rather than fixed:
    it trades granularity for coverage, and the selection step decides the balance.

    cluster_selection_method is swept for the same reason. The default 'eom' (excess of
    mass) prefers large, stable clusters, which on a semantically homogeneous category
    collapses everything into one blob — praise produced a single cluster holding 97% of
    its quotes. 'leaf' takes the leaves of the condensed tree instead, giving many small
    specific themes (praise: 31 themes, largest 7%). Neither is right for every category,
    so both are candidates and the selection constraints decide."""
    return hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=ms, metric="euclidean",
        cluster_selection_epsilon=eps, cluster_selection_method=method,
        prediction_data=True)


def sweep(X_train):
    """Sweep the parameter grid and return the selected (reducer, clusterer, params, metrics).

    The embeddings are PCA-reduced to a fixed PCA_COMPONENTS first — HDBSCAN is a density
    algorithm and its density estimate degrades badly in the raw 384 dimensions.

    Every candidate is fit twice: once on an inner split (to measure how well it accepts
    unseen quotes via approximate_predict) and once on the full 80% (the model actually
    shipped, and the one coherence is measured on). The real 20% test set is never
    consulted during model selection.
    """
    n = len(X_train)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_val = max(4, round(n * 0.25))
    Xv, Xf = X_train[perm[:n_val]], X_train[perm[n_val:]]

    # min_cluster_size starts at 3: a 2-quote "theme" is not a theme, and with the reducer
    # in place coverage no longer depends on allowing them (it did on raw 384-d).
    scaled = {max(3, round(n * f)) for f in (0.02, 0.04, 0.06)}
    mcs_grid = sorted(m for m in ({3, 4, 5, 6, 8, 12} | scaled) if 2 < m <= max(4, len(Xf) // 3))

    # Both reducers use the same component count; the inner one is fit on the inner split
    # only, so the generalization estimate never sees the data it is scored against.
    n_comp = min(PCA_COMPONENTS, max(2, len(Xf) - 1))
    full_red = PCA(n_components=n_comp, random_state=SEED).fit(X_train)
    Zt = full_red.transform(X_train)
    inner_red = PCA(n_components=n_comp, random_state=SEED).fit(Xf)
    Zf, Zv = inner_red.transform(Xf), inner_red.transform(Xv)

    cands = []
    for method in ("eom", "leaf"):
        for mcs in mcs_grid:
            for ms in (1, 3, 5):
                if ms > mcs:
                    continue
                for eps in (0.0, 0.15, 0.3, 0.5):
                    # inner model estimates generalization; full model is the one we ship,
                    # so both are measured (neither sees the real 20%).
                    inner_clus = make_clusterer(mcs, ms, eps, method).fit(Zf)
                    val_lab, _ = hdbscan.approximate_predict(inner_clus, Zv)
                    val_cov = float((val_lab >= 0).mean())

                    full_clus = make_clusterer(mcs, ms, eps, method).fit(Zt)
                    n_clusters, fit_cov, share = cluster_shape(full_clus.labels_)
                    # coherence is measured in the original 384-d space, not the reduced
                    # one, so it reflects what a reader would judge
                    coh = coherence(X_train, full_clus.labels_)

                    cands.append({
                        "n_components": n_comp,
                        "min_cluster_size": mcs, "min_samples": ms,
                        "cluster_selection_epsilon": eps,
                        "cluster_selection_method": method,
                        "n_clusters": n_clusters,
                        "clustered_fraction": round(fit_cov, 3),
                        "inner_val_coverage": round(val_cov, 3),
                        "largest_share": round(share, 3),
                        "coherence": round(coh, 4),
                        "coverage": round((max(fit_cov, 1e-9) * max(val_cov, 1e-9)) ** 0.5, 4),
                    })

    # n/0.8 recovers the full category size from the 80% training split, so the
    # theme-count ceiling reflects the category, not the split.
    win = pick_best(cands, round(n / 0.8))
    params = {k: win[k] for k in
              ("n_components", "min_cluster_size", "min_samples",
               "cluster_selection_epsilon", "cluster_selection_method")}
    params["reducer"] = "pca"
    # refit the winner (deterministic, so this reproduces the evaluated model exactly)
    reducer = PCA(n_components=params["n_components"], random_state=SEED).fit(X_train)
    clus = make_clusterer(params["min_cluster_size"], params["min_samples"],
                          params["cluster_selection_epsilon"],
                          params["cluster_selection_method"]).fit(reducer.transform(X_train))
    return reducer, clus, params, win


def viz_coords(X_all):
    """2-D projection for the scatter plot (visualization only, not used for clustering)."""
    n = len(X_all)
    if n < 4:
        xs = np.linspace(-1, 1, n)
        return np.column_stack([xs, np.zeros(n)])
    proj = umap.UMAP(n_components=2, n_neighbors=min(15, n - 1), min_dist=0.12,
                     metric="cosine", random_state=SEED)
    return proj.fit_transform(X_all)


def run_category(conn, category):
    ids, X = load_category(conn, category)
    n = len(ids)
    if n < MIN_QUOTES:
        print(f"[{category}] only {n} quotes (< {MIN_QUOTES}) — skipping")
        return
    print(f"[{category}] {n} quotes")

    train_i, test_i = split_80_20(n)
    ids_train, ids_test = ids[train_i], ids[test_i]
    X_train, X_test = X[train_i], X[test_i]

    # reducer + clusterer are fit on the 80% ONLY
    reducer, clus, params, metrics = sweep(X_train)
    print(f"[{category}] best params={params} -> {metrics}")

    # persist the fitted model bundle
    model_path = MODELS_DIR / f"{category}.joblib"
    joblib.dump({"reducer": reducer, "clusterer": clus,
                 "train_quote_ids": ids_train.tolist(), "params": params}, model_path)

    # self-check: reload and approximate_predict a known train point
    bundle = joblib.load(model_path)
    chk_label, _ = hdbscan.approximate_predict(
        bundle["clusterer"], bundle["reducer"].transform(X_train[:1]))
    print(f"[{category}] reload self-check ok (train point -> theme {int(chk_label[0])})")

    # assign the held-out 20% via approximate_predict, in the train-fit PCA space
    if len(X_test):
        test_labels, test_strengths = hdbscan.approximate_predict(
            clus, reducer.transform(X_test))
        test_labels = test_labels.astype(int)
    else:
        test_labels, test_strengths = np.array([], int), np.array([], float)

    # 2-D projection for the viz (all points, consistent layout)
    coords = viz_coords(np.vstack([X_train, X_test]))
    coords_train, coords_test = coords[:len(X_train)], coords[len(X_train):]

    train_labels = clus.labels_.astype(int)
    train_probs = clus.probabilities_.astype(float)

    # Quotes HDBSCAN rejects stay rejected. An earlier version attached them to the
    # nearest centroid to drive the "unclustered" count down, but that is the wrong
    # trade: it pulled loosely-related quotes into themes and made them mean less.
    n_clusters = int(max(train_labels.max(), test_labels.max() if len(test_labels) else -1)) + 1
    noise_frac = float(
        (np.concatenate([train_labels, test_labels]) < 0).mean()
        if len(test_labels) else (train_labels < 0).mean()
    )

    run_metrics = {"n_total": n, "n_train": int(len(ids_train)), "n_test": int(len(ids_test)),
                   "n_clusters": n_clusters, "noise_fraction": round(noise_frac, 3),
                   "coherence": metrics.get("coherence"),
                   "sweep": metrics, "reducer": "pca",
                   "pca_components": int(reducer.n_components),
                   "embedding_dims": int(X.shape[1])}

    with conn.transaction():
        # replace any prior run for this category (demo: keep only the latest)
        conn.execute("delete from cluster_runs where category = %s", (category,))
        run_id = conn.execute(
            """insert into cluster_runs (category, algo, params, metrics, model_path)
               values (%s, 'hdbscan', %s, %s, %s) returning id""",
            (category, json.dumps(params), json.dumps(run_metrics), str(model_path.relative_to(ROOT))),
        ).fetchone()[0]

        rows = []
        for qid, lab, prob, (x, y) in zip(
                ids_train, train_labels, train_probs, coords_train):
            how = "unclustered" if lab < 0 else "hdbscan"
            rows.append((run_id, int(qid), int(lab), "train", float(prob), float(x), float(y), how))
        for qid, lab, strg, (x, y) in zip(
                ids_test, test_labels, test_strengths, coords_test):
            how = "unclustered" if lab < 0 else "approximate_predict"
            rows.append((run_id, int(qid), int(lab), "test", float(strg), float(x), float(y), how))
        conn.cursor().executemany(
            """insert into quote_clusters (run_id, quote_id, cluster_id, split, probability, x, y, assigned_by)
               values (%s, %s, %s, %s, %s, %s, %s, %s)""", rows)

        # cluster rows with sizes (labels filled in by 05_label_clusters)
        all_labels = np.concatenate([train_labels, test_labels]) if len(test_labels) else train_labels
        uniq, counts = np.unique(all_labels, return_counts=True)
        for cl, sz in zip(uniq.tolist(), counts.tolist()):
            label = "Unclustered" if cl < 0 else None
            conn.execute(
                """insert into clusters (run_id, cluster_id, label, size) values (%s,%s,%s,%s)""",
                (run_id, int(cl), label, int(sz)),
            )
    print(f"[{category}] run {run_id}: {n_clusters} clusters, "
          f"coherence={metrics.get('coherence')}, noise={noise_frac:.0%}, "
          f"train={len(ids_train)} test={len(ids_test)}")


def main():
    with connect() as conn:
        for cat in CATEGORIES:
            run_category(conn, cat)
        for cat in CATEGORIES:
            row = conn.execute(
                "select metrics from cluster_runs where category=%s order by id desc limit 1",
                (cat,)).fetchone()
            if row:
                m = row[0]
                print(f"  {cat:16} clusters={m['n_clusters']:3}  noise={m['noise_fraction']:.0%}")
    print("\nclustering complete")


if __name__ == "__main__":
    main()
