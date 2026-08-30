-- Whether the LLM judged a cluster to cover more than one distinct concern.
--
-- Fine-grained clustering still produces the occasional grab-bag of unrelated one-off
-- bugs ("PDF import fails" next to "cursor is invisible in light mode"). Rather than
-- hide that, the label step asks the model to flag it and the UI shows a caveat: a weak
-- theme a reader can discount is far better than one that silently looks solid.
alter table clusters
    add column if not exists mixed boolean not null default false;
