"""Interactive Streamlit decision tool for RFM customer segmentation."""
from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.model import RFM_COLUMNS
from src.segments import name_clusters

ARTIFACT_DIR = Path("artifacts")
ACTION_PLAYBOOK = {
    "High-Value Loyal": "Protect with VIP treatment, early access, and relevant cross-sell journeys.",
    "At-Risk Valuable": "Prioritize a win-back offer, service recovery, and contact-frequency review.",
    "Recent / Low-Spend": "Guide the second purchase with onboarding, recommendations, and a low-friction offer.",
    "Core / Mid-Value": "Nurture toward higher frequency through personalization and category discovery.",
}
st.set_page_config(page_title="Retail Customer Intelligence", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e9eef7; --muted:#8d9ab0; --line:#26344b; --panel:#111b2d; --accent:#6ee7d8; --purple:#9b8cff; }
.stApp { background: radial-gradient(circle at 15% 0%, #1b2b46 0, #0b1220 34%, #08101c 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#09111f; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
h1 { font-size:3.1rem !important; margin-bottom:.25rem !important; }
.hero { padding:1.5rem 0 1rem; }
.eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:700; margin-bottom:.6rem; }
.subtitle { color:var(--muted); font-size:1.08rem; margin-bottom:1.7rem; }
.kpi { background:linear-gradient(145deg,rgba(23,37,61,.92),rgba(13,23,40,.92)); border:1px solid var(--line); border-radius:18px; padding:1rem 1.15rem; box-shadow:0 14px 35px rgba(0,0,0,.16); }
.kpi-label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.1em; }
.kpi-value { color:var(--ink); font-family:'Space Grotesk'; font-size:1.65rem; font-weight:700; margin-top:.35rem; }
.section-label { color:var(--muted); text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; font-weight:700; margin:1rem 0 .5rem; }
div[data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { color:var(--muted); font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--accent); }
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; background:rgba(15,26,44,.45); }
.stCaption, [data-testid="stMarkdownContainer"] p { color:var(--muted); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(ARTIFACT_DIR / "rfm_kmeans_pipeline.joblib")

@st.cache_data
def load_data():
    data = pd.read_csv(ARTIFACT_DIR / "customer_segments.csv")
    mapping = name_clusters(data)
    data["Segment"] = data["Cluster"].map(mapping)
    return data, mapping

def main():
    st.markdown('<div class="hero"><div class="eyebrow">Customer analytics / retention lab</div><h1>Retail Customer Intelligence</h1><div class="subtitle">Turn purchase behavior into focused retention priorities.</div></div>', unsafe_allow_html=True)
    if not (ARTIFACT_DIR / "rfm_kmeans_pipeline.joblib").exists():
        st.error("Model artifacts are missing. Run the Day 1 training command first.")
        st.stop()
    model = load_model()
    customers, mapping = load_data()
    metadata_path = ARTIFACT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    with st.sidebar:
        st.header("Model context")
        st.metric("Customers scored", f"{len(customers):,}")
        st.metric("Selected k", metadata.get("n_clusters", customers["Cluster"].nunique()))
        st.caption(f"As-of snapshot: {metadata.get('snapshot_date', 'not recorded')}")
        st.caption("Descriptive segmentation; not a churn probability.")

    explorer, scoring, evidence = st.tabs(["Segment Explorer", "Score a Customer", "Model Evidence"])
    with explorer:
        selected = st.multiselect("Show segments", sorted(customers.Segment.unique()), default=sorted(customers.Segment.unique()))
        view = customers[customers.Segment.isin(selected)].copy()
        k1, k2, k3, k4 = st.columns(4)
        cards = [("Customers in view", f"{len(view):,}"), ("Gross value", f"£{view.Monetary.sum():,.0f}"), ("Median recency", f"{view.Recency.median():.0f} days"), ("Median frequency", f"{view.Frequency.median():.0f} invoices")]
        for slot, (label, value) in zip((k1,k2,k3,k4), cards):
            slot.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)
        chart_type = st.radio("Behavior view", ["3D RFM", "2D Recency vs Value"], horizontal=True)
        if chart_type == "3D RFM":
            fig = px.scatter_3d(view, x="Recency", y="Frequency", z="Monetary", color="Segment", hover_data=["CustomerID"], log_z=True, title="Customer behavioral space")
        else:
            fig = px.scatter(view, x="Recency", y="Monetary", size="Frequency", color="Segment", hover_data=["CustomerID", "Frequency"], log_y=True, title="Recency versus customer value")
        fig.update_layout(height=620, legend_title_text="Business segment", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#e9eef7"), margin=dict(l=0,r=0,t=55,b=0))
        fig.update_scenes(xaxis_title="Recency (days)", yaxis_title="Frequency", zaxis_title="Monetary (£)", bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="#26344b"), yaxis=dict(gridcolor="#26344b"), zaxis=dict(gridcolor="#26344b"))
        st.plotly_chart(fig, use_container_width=True)
        left, right = st.columns(2)
        with left:
            counts = view.groupby("Segment", as_index=False).size().rename(columns={"size": "Customers"})
            bar = px.bar(counts, y="Segment", x="Customers", color="Segment", orientation="h", title="Customers by segment")
            bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color="#e9eef7"), margin=dict(l=0,r=0,t=55,b=0), showlegend=False)
            st.plotly_chart(bar, use_container_width=True)
        with right:
            st.dataframe(view.groupby("Segment")[RFM_COLUMNS].agg(["count", "median", "mean"]).round(2), use_container_width=True, height=300)
        st.download_button("Download filtered customer scores", view.to_csv(index=False), "customer_segments_filtered.csv", "text/csv")

    with scoring:
        st.markdown('<div class="section-label">Decision support</div>', unsafe_allow_html=True)
        st.subheader("Score a new customer")
        st.caption("Use the same RFM definitions and units used during training: days, unique invoices, and GBP.")
        a, b, c = st.columns(3)
        recency = a.number_input("Recency (days)", min_value=0.0, max_value=5000.0, value=30.0, step=1.0)
        frequency = b.number_input("Frequency (unique invoices)", min_value=1.0, max_value=10000.0, value=4.0, step=1.0)
        monetary = c.number_input("Monetary (GBP)", min_value=0.0, max_value=10_000_000.0, value=250.0, step=25.0)
        if st.button("Score customer", type="primary"):
            cluster = int(model.predict(pd.DataFrame([[recency, frequency, monetary]], columns=RFM_COLUMNS))[0])
            segment = mapping[cluster]
            st.success(f"Assigned **{segment}** (model cluster {cluster})")
            st.info(f"Suggested next action: {ACTION_PLAYBOOK[segment]}")
            st.dataframe(pd.DataFrame([{"Recency": recency, "Frequency": frequency, "Monetary": monetary, "Cluster": cluster, "Segment": segment}]), use_container_width=True, hide_index=True)

    with evidence:
        st.markdown('<div class="section-label">Trust and transparency</div>', unsafe_allow_html=True)
        st.subheader("Model selection evidence")
        comparison_path = ARTIFACT_DIR / "model_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            st.plotly_chart(px.line(comparison, x="k", y="silhouette", markers=True, title="Silhouette score by candidate cluster count"), use_container_width=True)
            st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
        st.markdown("""**Interpretation guardrails**

        - Cluster IDs are arbitrary and are mapped to business names from relative RFM profiles.
        - Silhouette and seed stability describe this historical sample; they do not measure campaign lift.
        - Monetary is gross positive purchase value, not profit, net revenue, or predicted lifetime value.
        - The next production validation would be temporal segment migration and a randomized retention test.
        """)

if __name__ == "__main__":
    main()
