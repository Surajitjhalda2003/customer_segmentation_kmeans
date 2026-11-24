"""
app.py
------
Interactive Streamlit dashboard for the Customer Segmentation (K-Means) project.

Run with:
    streamlit run app.py
"""

import os
import sys
import sqlite3

import numpy as np
import pandas as pd
# pyright: reportMissingImports=false
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rfm import build_rfm_table, DB_PATH          # noqa: E402
from clustering import run_pipeline, SEGMENT_RECOMMENDATIONS, RFM_COLS  # noqa: E402

st.set_page_config(
    page_title="Customer Segmentation — K-Means",
    page_icon="📊",
    layout="wide",
)

SEGMENT_COLORS = {
    "VIP": "#22c55e",        # green
    "Loyal": "#3b82f6",      # blue
    "Potential": "#eab308",  # yellow
    "At-risk": "#ef4444",    # red
}
SEGMENT_EMOJI = {k: v["emoji"] for k, v in SEGMENT_RECOMMENDATIONS.items()}


# ---------------------------------------------------------------------------
# Data loading (cached so the DB / clustering isn't recomputed on every click)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data & computing RFM features...")
def load_data():
    if not os.path.exists(DB_PATH):
        st.error(
            "Database not found. Run `python data/generate_data.py` then "
            "`python database/create_db.py` first (see README)."
        )
        st.stop()
    rfm, clean_txns = build_rfm_table()
    return rfm, clean_txns


@st.cache_data(show_spinner="Running K-Means clustering...")
def run_clustering(rfm_df, k):
    result = run_pipeline(rfm_df, k=k)
    return result["rfm_labeled"], result["cluster_stats"], result["eval_df"], result["chosen_k"]


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Adjust clustering parameters and filters.")

k_choice = st.sidebar.slider(
    "Number of clusters (k)", min_value=2, max_value=8, value=4,
    help="4 aligns clusters directly with VIP / Loyal / Potential / At-risk tiers."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline:** transactions → clean → RFM → K-Means → segments → recommendations"
)

rfm_raw, clean_txns = load_data()
labeled_df, cluster_stats, eval_df, chosen_k = run_clustering(rfm_raw, k_choice)

region_options = ["All"] + sorted(labeled_df["Region"].dropna().unique().tolist())
region_filter = st.sidebar.selectbox("Filter by Region", region_options)

segment_options = ["All"] + sorted(labeled_df["Segment"].unique().tolist())
segment_filter = st.sidebar.selectbox("Filter by Segment", segment_options)

filtered = labeled_df.copy()
if region_filter != "All":
    filtered = filtered[filtered["Region"] == region_filter]
if segment_filter != "All":
    filtered = filtered[filtered["Segment"] == segment_filter]


# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("📊 Customer Segmentation using K-Means")
st.caption("RFM feature engineering + K-Means clustering → actionable marketing strategy per segment")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{len(labeled_df):,}")
k2.metric("Total Revenue", f"${labeled_df['Monetary'].sum():,.0f}")
k3.metric("Avg Order Value", f"${labeled_df['AvgOrderValue'].mean():,.2f}")
k4.metric("Clusters (k)", chosen_k)

st.markdown("---")

# ---------------------------------------------------------------------------
# Segment distribution + revenue contribution
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Customer Distribution by Segment")
    seg_counts = labeled_df["Segment"].value_counts().reindex(
        ["VIP", "Loyal", "Potential", "At-risk"]
    ).dropna()
    fig_pie = px.pie(
        names=[f"{SEGMENT_EMOJI.get(s,'')} {s}" for s in seg_counts.index],
        values=seg_counts.values,
        color=seg_counts.index,
        color_discrete_map={f"{SEGMENT_EMOJI.get(s,'')} {s}": SEGMENT_COLORS.get(s, "#888")
                             for s in seg_counts.index},
        hole=0.45,
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Revenue Contribution by Segment")
    rev_by_seg = labeled_df.groupby("Segment")["Monetary"].sum().reindex(
        ["VIP", "Loyal", "Potential", "At-risk"]
    ).dropna()
    fig_bar = px.bar(
        x=rev_by_seg.index, y=rev_by_seg.values,
        color=rev_by_seg.index,
        color_discrete_map=SEGMENT_COLORS,
        labels={"x": "Segment", "y": "Total Revenue ($)"},
        text_auto=".2s",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# RFM 3D scatter
# ---------------------------------------------------------------------------
st.subheader("RFM Cluster Visualization")
fig_scatter = px.scatter_3d(
    labeled_df, x="Recency", y="Frequency", z="Monetary",
    color="Segment", color_discrete_map=SEGMENT_COLORS,
    hover_data=["CustomerID", "Region", "AvgOrderValue"],
    opacity=0.75,
)
fig_scatter.update_layout(height=550, margin=dict(l=0, r=0, b=0, t=30))
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Elbow / silhouette diagnostic
# ---------------------------------------------------------------------------
with st.expander("📈 How was k chosen? (Elbow method & silhouette score)"):
    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = px.line(eval_df, x="k", y="inertia", markers=True,
                             title="Elbow Method (Inertia vs k)")
        st.plotly_chart(fig_elbow, use_container_width=True)
    with c2:
        fig_sil = px.line(eval_df, x="k", y="silhouette", markers=True,
                           title="Silhouette Score vs k")
        st.plotly_chart(fig_sil, use_container_width=True)
    st.markdown(
        "The elbow method looks for the point where adding more clusters stops "
        "meaningfully reducing inertia (within-cluster variance). The silhouette "
        "score measures how well-separated clusters are (closer to 1 is better). "
        "**k=4** is used by default here to match the business framing "
        "(VIP / Loyal / Potential / At-risk), but you can change k in the sidebar."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Segment profile table + business recommendations
# ---------------------------------------------------------------------------
st.subheader("Segment Profiles & Marketing Recommendations")

for seg in ["VIP", "Loyal", "Potential", "At-risk"]:
    if seg not in labeled_df["Segment"].unique():
        continue
    info = SEGMENT_RECOMMENDATIONS[seg]
    seg_df = labeled_df[labeled_df["Segment"] == seg]
    with st.container(border=True):
        st.markdown(f"### {info['emoji']} {seg} — {len(seg_df)} customers "
                     f"({len(seg_df)/len(labeled_df)*100:.1f}%)")
        st.markdown(f"*{info['description']}*")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Recency (days)", f"{seg_df['Recency'].mean():.0f}")
        m2.metric("Avg Frequency", f"{seg_df['Frequency'].mean():.1f}")
        m3.metric("Avg Monetary ($)", f"{seg_df['Monetary'].mean():,.0f}")
        m4.metric("Total Revenue ($)", f"{seg_df['Monetary'].sum():,.0f}")

        st.markdown(f"**Recommended strategy:** {info['strategy']}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Customer explorer table + export
# ---------------------------------------------------------------------------
st.subheader("🔍 Customer Explorer")
st.caption(f"Showing {len(filtered)} customers"
           + (f" in region '{region_filter}'" if region_filter != "All" else "")
           + (f", segment '{segment_filter}'" if segment_filter != "All" else ""))

display_cols = ["CustomerID", "Region", "Segment", "Recency", "Frequency",
                 "Monetary", "AvgOrderValue"]
st.dataframe(
    filtered[display_cols].sort_values("Monetary", ascending=False),
    use_container_width=True,
    height=350,
)

csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download filtered segments as CSV",
    data=csv,
    file_name="customer_segments_export.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Built with Python, Pandas, Scikit-learn, SQLite & Streamlit · "
    "Data is synthetic — swap in real transaction data via data/generate_data.py's output format."
)
