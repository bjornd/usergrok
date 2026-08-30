-- Collapse the three-way taxonomy into two.
--
-- "feature_request" vs "bug_report" turned out to be an unstable distinction for review
-- text: "offline mode is basically nonexistent" landed in one and "the offline mode
-- still isn't great" in the other (cosine 0.85), splitting a single theme across two
-- views. What a product team acts on is the concern, not whether the user phrased it as
-- a defect or a wish -- so both collapse into 'pain_point'.
alter table quotes drop constraint if exists quotes_category_check;
alter table cluster_runs drop constraint if exists cluster_runs_category_check;

update quotes set category = 'pain_point'
 where category in ('feature_request', 'bug_report');
update cluster_runs set category = 'pain_point'
 where category in ('feature_request', 'bug_report');

alter table quotes
    add constraint quotes_category_check check (category in ('praise', 'pain_point'));
alter table cluster_runs
    add constraint cluster_runs_category_check check (category in ('praise', 'pain_point'));
