-- UserGrok feedback-analysis demo schema
-- pgvector for high-dimensional quote embeddings + HDBSCAN clustering results.

create extension if not exists vector;

-- Raw feedback collected from G2 (sample product: Notion).
create table if not exists reviews (
    id            bigint generated always as identity primary key,
    source        text not null default 'g2',
    product       text not null default 'notion',
    external_key  text unique not null,           -- stable G2 review id (e.g. notion-review-10888352)
    title         text,
    body          text not null,                  -- combined like/dislike/problems text fed to the LLM
    like_text     text,
    dislike_text  text,
    problems_text text,
    rating        real,                            -- 1.0 .. 5.0
    author        text,
    author_role   text,
    published_at  date,
    page_url      text,
    snapshot_ts   text,                            -- wayback snapshot timestamp for provenance
    collected_at  timestamptz not null default now()
);

-- LLM-extracted verbatim quotes, one row per (review, category, quote).
create table if not exists quotes (
    id         bigint generated always as identity primary key,
    review_id  bigint not null references reviews(id) on delete cascade,
    category   text not null check (category in ('feature_request','bug_report','praise')),
    text       text not null,
    embedding  vector(384),                        -- all-MiniLM-L6-v2
    created_at timestamptz not null default now(),
    unique (review_id, category, text)
);
create index if not exists quotes_category_idx on quotes(category);

-- One clustering run per category (latest run per category is what the app renders).
create table if not exists cluster_runs (
    id         bigint generated always as identity primary key,
    category   text not null check (category in ('feature_request','bug_report','praise')),
    algo       text not null default 'hdbscan',
    params     jsonb not null default '{}'::jsonb,
    metrics    jsonb not null default '{}'::jsonb, -- n_train, n_test, n_clusters, noise_fraction, ...
    model_path text,                               -- saved joblib bundle (pca + clusterer + train ids)
    created_at timestamptz not null default now()
);

-- Cluster metadata (LLM-generated label + summary) for a run.
create table if not exists clusters (
    run_id     bigint not null references cluster_runs(id) on delete cascade,
    cluster_id int not null,                       -- -1 = noise / unclustered
    label      text,
    summary    text,
    size       int not null default 0,
    primary key (run_id, cluster_id)
);

-- Assignment of each quote to a cluster within a run, plus 2-D projection coords for the viz.
create table if not exists quote_clusters (
    run_id      bigint not null references cluster_runs(id) on delete cascade,
    quote_id    bigint not null references quotes(id) on delete cascade,
    cluster_id  int not null,                      -- -1 = noise
    split       text not null check (split in ('train','test')),  -- 'test' = assigned via approximate_predict
    probability real,
    x           real,                              -- t-SNE coord (viz only)
    y           real,
    primary key (run_id, quote_id)
);
create index if not exists quote_clusters_run_idx on quote_clusters(run_id);
