-- How a quote ended up in its cluster, so the UI can be honest about it:
--   hdbscan             - core clustering of the 80% training split
--   approximate_predict - held-out 20% mapped onto the saved model
--   nearest_centroid    - HDBSCAN called it noise, but it sits inside a cluster's
--                         similarity range, so it is attached (and shown as such)
--   unclustered         - genuinely far from every cluster
alter table quote_clusters
    add column if not exists assigned_by text not null default 'hdbscan';
