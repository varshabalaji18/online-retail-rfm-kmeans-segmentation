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
    st.title("Retail Customer Intelligence")
    st.markdown("**RFM behavior → K-Means segments → retention priorities**")
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
        k1.metric("Customers in view", f"{len(view):,}")
        k2.metric("Gross value", f"£{view.Monetary.sum():,.0f}")
        k3.metric("Median recency", f"{view.Recency.median():.0f} days")
        k4.metric("Median frequency", f"{view.Frequency.median():.0f} invoices")
        chart_type = st.radio("Behavior view", ["3D RFM", "2D Recency vs Value"], horizontal=True)
        if chart_type == "3D RFM":
            fig = px.scatter_3d(view, x="Recency", y="Frequency", z="Monetary", color="Segment", hover_data=["CustomerID"], log_z=True, title="Customer behavioral space (log-scaled monetary axis)")
        else:
            fig = px.scatter(view, x="Recency", y="Monetary", size="Frequency", color="Segment", hover_data=["CustomerID", "Frequency"], log_y=True, title="Recency versus customer value")
        fig.update_layout(height=620, legend_title_text="Business segment")
        st.plotly_chart(fig, use_container_width=True)
        left, right = st.columns(2)
        with left:
            counts = view.groupby("Segment", as_index=False).size().rename(columns={"size": "Customers"})
            st.plotly_chart(px.bar(counts, y="Segment", x="Customers", color="Segment", orientation="h", title="Customers by segment"), use_container_width=True)
        with right:
            st.dataframe(view.groupby("Segment")[RFM_COLUMNS].agg(["count", "median", "mean"]).round(2), use_container_width=True, height=300)
        st.download_button("Download filtered customer scores", view.to_csv(index=False), "customer_segments_filtered.csv", "text/csv")

    with scoring:
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
