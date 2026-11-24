"""
clustering.py
-------------
Scales RFM features, finds a good K via the elbow method + silhouette score,
fits K-Means, and maps the resulting numeric clusters to human-readable,
business-meaningful segment labels (VIP / Loyal / Potential / At-risk).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RFM_COLS = ["Recency", "Frequency", "Monetary"]


def prepare_features(rfm_df):
    """Log-transform skewed monetary/frequency features, then standardize."""
    features = rfm_df[RFM_COLS].copy()

    # Log1p reduces skew from big spenders / highly frequent buyers
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])
    # Recency: smaller is "better" (more recent); keep as-is, scaler handles it

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def evaluate_k_range(scaled_features, k_range=range(2, 9), random_state=42):
    """Return inertia (elbow) and silhouette scores for a range of k."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(scaled_features)
        sil = silhouette_score(scaled_features, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(results)


def fit_kmeans(scaled_features, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(scaled_features)
    return km, labels


def label_segments(rfm_df, cluster_col="Cluster"):
    """
    Rank clusters by a composite 'value score' (high Frequency & Monetary,
    low Recency) and map them to business-friendly segment names.
    Works for any k >= 2 by binning ranked clusters into 4 tiers.
    """
    df = rfm_df.copy()

    cluster_stats = df.groupby(cluster_col).agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Count=(cluster_col, "count"),
    )

    # Normalize each metric 0-1 across clusters, then combine into one score.
    # Recency is inverted (lower recency = better).
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    score = (
        (1 - norm(cluster_stats["Recency"])) * 0.35
        + norm(cluster_stats["Frequency"]) * 0.35
        + norm(cluster_stats["Monetary"]) * 0.30
    )
    cluster_stats["ValueScore"] = score
    cluster_stats = cluster_stats.sort_values("ValueScore", ascending=False)

    tier_names = ["VIP", "Loyal", "Potential", "At-risk"]
    n_clusters = len(cluster_stats)
    # Distribute clusters across the 4 tiers proportionally by rank
    bins = np.array_split(cluster_stats.index.tolist(), min(4, n_clusters))
    label_map = {}
    for tier_idx, cluster_ids in enumerate(bins):
        tier = tier_names[tier_idx] if n_clusters >= 4 else tier_names[
            round(tier_idx * (len(tier_names) - 1) / max(len(bins) - 1, 1))
        ]
        for cid in cluster_ids:
            label_map[cid] = tier

    df["Segment"] = df[cluster_col].map(label_map)
    cluster_stats["Segment"] = cluster_stats.index.map(label_map)
    return df, cluster_stats.reset_index()


SEGMENT_RECOMMENDATIONS = {
    "VIP": {
        "emoji": "🟢",
        "description": "Buy frequently and spend the most. Your most valuable customers.",
        "strategy": (
            "Reward loyalty with VIP perks (early access, exclusive discounts, "
            "premium support). Assign account managers for top spenders. "
            "Use them for referrals and testimonials. Avoid generic discounting — "
            "they already buy; focus on retention and upsell."
        ),
    },
    "Loyal": {
        "emoji": "🔵",
        "description": "Purchase regularly with solid spend. Reliable, consistent revenue base.",
        "strategy": (
            "Nurture with loyalty programs and personalized recommendations. "
            "Cross-sell complementary products. Encourage them toward VIP status "
            "with milestone-based incentives (e.g., 'spend $X more to unlock...')."
        ),
    },
    "Potential": {
        "emoji": "🟡",
        "description": "Moderate spend and frequency — could become valuable with the right nudge.",
        "strategy": (
            "Run targeted promotions and limited-time offers to increase purchase "
            "frequency. Use onboarding/education campaigns. A/B test incentives "
            "to find what converts them into Loyal customers."
        ),
    },
    "At-risk": {
        "emoji": "🔴",
        "description": "Rarely purchase or spend little, and/or haven't bought recently.",
        "strategy": (
            "Launch win-back campaigns (e.g., 'we miss you' discounts). Survey for "
            "churn reasons. Deprioritize costly marketing spend on customers who "
            "don't respond; focus budget on Potential and Loyal segments instead."
        ),
    },
}


def run_pipeline(rfm_df, k=4, k_range=range(2, 9)):
    """
    Full clustering pipeline: scale -> (optionally choose k) -> fit -> label.

    Default k=4 aligns with the business goal (VIP / Loyal / Potential /
    At-risk). Pass k=None to instead auto-select k by best silhouette score —
    the eval_df (elbow + silhouette scores) is always returned so you can
    judge the trade-off yourself, e.g. in the dashboard's elbow chart.
    """
    scaled, scaler = prepare_features(rfm_df)

    eval_df = evaluate_k_range(scaled, k_range=k_range)

    if k is None:
        # choose k with best silhouette score
        k = int(eval_df.loc[eval_df["silhouette"].idxmax(), "k"])

    km, labels = fit_kmeans(scaled, k)
    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = labels

    labeled_df, cluster_stats = label_segments(rfm_df)
    return {
        "rfm_labeled": labeled_df,
        "cluster_stats": cluster_stats,
        "eval_df": eval_df,
        "chosen_k": k,
        "model": km,
        "scaler": scaler,
    }


if __name__ == "__main__":
    import os
    from rfm import build_rfm_table

    rfm, _ = build_rfm_table()
    result = run_pipeline(rfm)
    print(f"Chosen k = {result['chosen_k']}")
    print(result["cluster_stats"])
    print(result["rfm_labeled"]["Segment"].value_counts())

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "customer_segments.csv")
    result["rfm_labeled"].to_csv(out_path, index=False)
    print(f"Saved labeled segments to {out_path}")
