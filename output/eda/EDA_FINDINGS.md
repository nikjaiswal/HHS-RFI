# EDA findings for the HHS-ONC-2026-0001 comment corpus

Run on n=446 comments (LLM-coded full corpus).


## 1. Position-vector coalitional structure


K-means clustering on the 6-axis governance position vector identifies **2 empirical coalitions** (silhouette-optimal). PC1+PC2 of the one-hot-encoded position space explain 24.6% and 10.1% of variance respectively, suggesting the policy debate has fewer effective dimensions than 6.


See `summary_cluster_profiles.csv` for cluster-by-cluster breakdown.


## 2. Where the corpus is silent


Liability is the least-addressed axis (53% engagement, the lowest of the six). The silence map (`fig_silence_heatmap.png`, `summary_silence_by_axis.csv`) shows where each stakeholder type is silent on which axes; this is informative about what HHS rule-making *cannot* draw from the comment record.


## 3. Topic co-occurrence


Phi-correlation matrix among 15 topic flags + hierarchical clustering reveals the modular structure of the policy debate (`fig_topic_cooccurrence_heat.png`).


## 4. Asymmetric mobilization / submission depth


Distributions of proposal counts, topic counts, and CFR-citation rates by commenter type show whether the comment record is dominated by policy-sophisticated coalition briefs or scattered individual voices (`fig_length_density.png`).


## 5. Within-stakeholder agreement


`summary_within_stakeholder.csv` quantifies how unified each stakeholder type is on governance positions. Higher modal share = more internal agreement; higher entropy = more internal split.


## 6. RFI question coverage


`fig_rfi_coverage.png` shows which of the 10 specific RFI questions each stakeholder engaged with — directly relevant to which policy levers HHS got the most input on.


## 7. Outliers


`summary_outliers.csv` lists the 15 commenters most distant from the corpus centroid in position-vector space. These are useful for the Discussion as examples of unusual policy stances or as candidates for case-study quotation.
