-- Drop the per-cluster "mixed concerns" flag.
--
-- It asked the labelling LLM to judge whether a cluster covered more than one concern and
-- surfaced that as a caveat badge. Removed along with the merge step: with the reducer
-- restored the clusters are coherent enough that the flag was mostly noise in the UI.
alter table clusters drop column if exists mixed;
