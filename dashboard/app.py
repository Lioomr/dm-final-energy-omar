from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_dark"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
PREPARED_DATA_PATH = PROJECT_ROOT / "outputs" / "prepared_hourly_energy.csv"

from src.data_preparation import (  # noqa: E402
    DEFAULT_DATA_PATH,
    available_forecast_features,
    available_model_features,
    prepare_energy_dataset,
    reduce_memory_usage,
    summarize_dataset,
)


st.set_page_config(
    page_title="Smart Energy Analytics",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
/* ── Metric cards ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
    border: 1px solid #06B6D4;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 0 18px rgba(6, 182, 212, 0.18);
}
[data-testid="metric-container"] label {
    color: #9CA3AF !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #06B6D4 !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

/* ── Section subheaders ───────────────────────────── */
h2, h3 {
    border-left: 4px solid #06B6D4;
    padding-left: 12px;
    margin-top: 2rem !important;
}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1F2937 !important;
    border-right: 1px solid #374151;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label {
    color: #F9FAFB !important;
}

/* ── Info / warning boxes ─────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Radio nav bar ────────────────────────────────── */
[data-testid="stRadio"] > div {
    gap: 8px;
}
[data-testid="stRadio"] label {
    background: #1F2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 6px 16px;
    font-weight: 500;
    transition: border-color 0.2s;
}
[data-testid="stRadio"] label:hover {
    border-color: #06B6D4;
}

/* ── Dataframes ───────────────────────────────────── */
[data-testid="stDataFrameResizable"] {
    border: 1px solid #374151;
    border-radius: 8px;
}

/* ── Dividers ─────────────────────────────────────── */
hr { border-color: #374151; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Preparing energy data...")
def load_prepared_data(use_full_data: bool, sample_rows: int) -> pd.DataFrame:
    if PREPARED_DATA_PATH.exists():
        df = pd.read_csv(PREPARED_DATA_PATH, parse_dates=["datetime"])
        if not use_full_data:
            estimated_hours = max(int(sample_rows / 60), 24)
            df = df.head(min(estimated_hours, len(df)))
        return reduce_memory_usage(df)

    nrows = None if use_full_data else sample_rows
    df = prepare_energy_dataset(
        data_path=DEFAULT_DATA_PATH,
        nrows=nrows,
        frequency="h",
    )
    return reduce_memory_usage(df)


@st.cache_data(show_spinner="Running baseline models...")
def run_dashboard_models(
    df: pd.DataFrame,
    feature_columns: list[str],
    forecast_feature_columns: list[str],
):
    from src.modeling import (  # noqa: WPS433
        pca_feature_summary,
        run_clustering_baselines,
        train_classification_baselines,
        train_regression_baselines,
    )

    clustering_metrics, clustered_df = run_clustering_baselines(
        df,
        features=feature_columns,
        max_rows=12000,
    )
    regression_metrics, _ = train_regression_baselines(
        df,
        features=forecast_feature_columns,
        target="global_active_power",
    )
    classification_metrics, _, report_text = train_classification_baselines(
        df,
        features=feature_columns,
        target="high_consumption",
    )
    pca_summary, _ = pca_feature_summary(
        df,
        features=feature_columns,
        n_components=3,
    )
    return clustering_metrics, clustered_df, regression_metrics, classification_metrics, report_text, pca_summary


@st.cache_data(show_spinner="Running clustering...")
def run_clustering_section(df: pd.DataFrame, feature_columns: list[str]):
    from src.modeling import cluster_submetering_breakdown, run_clustering_baselines, summarize_clusters  # noqa: WPS433

    clustering_metrics, clustered_df = run_clustering_baselines(
        df,
        features=feature_columns,
        max_rows=8000,
    )
    cluster_profile = summarize_clusters(clustered_df)
    submetering_df = cluster_submetering_breakdown(clustered_df)
    return clustering_metrics, clustered_df, cluster_profile, submetering_df


@st.cache_data(show_spinner="Running regression...")
def run_regression_section(
    df: pd.DataFrame,
    forecast_feature_columns: list[str],
):
    from src.modeling import regression_predictions, train_regression_baselines  # noqa: WPS433

    regression_metrics, _ = train_regression_baselines(
        df,
        features=forecast_feature_columns,
        target="global_active_power",
    )
    pred_df = regression_predictions(df, features=forecast_feature_columns)
    return regression_metrics, pred_df


@st.cache_data(show_spinner="Running classification...")
def run_classification_section(df: pd.DataFrame, feature_columns: list[str]):
    from src.modeling import rf_feature_importance, train_classification_baselines  # noqa: WPS433

    classification_metrics, models, report_text = train_classification_baselines(
        df,
        features=feature_columns,
        target="high_consumption",
    )
    feature_importance_df = rf_feature_importance(models, feature_columns)
    return classification_metrics, report_text, feature_importance_df


@st.cache_data(show_spinner="Running PCA summary...")
def run_pca_section(df: pd.DataFrame, feature_columns: list[str]):
    from src.modeling import pca_feature_summary  # noqa: WPS433

    pca_summary, _ = pca_feature_summary(
        df,
        features=feature_columns,
        n_components=3,
    )
    return pca_summary


st.markdown("""
<div style="background: linear-gradient(90deg, #0284C7 0%, #06B6D4 60%, #0891B2 100%);
            padding: 22px 32px; border-radius: 14px; margin-bottom: 8px;">
    <h1 style="color: white; margin: 0; font-size: 2rem; letter-spacing: -0.5px;">
        ⚡ Smart Energy Consumption Analytics
    </h1>
    <p style="color: rgba(255,255,255,0.82); margin: 8px 0 0 0; font-size: 0.95rem;">
        Household power consumption &nbsp;·&nbsp; Clustering &nbsp;·&nbsp;
        Regression &nbsp;·&nbsp; Classification
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Global Filters")
    use_full_data = st.toggle("Use full dataset", value=False)
    sample_rows = st.slider(
        "Development sample rows",
        min_value=50000,
        max_value=300000,
        value=100000,
        step=50000,
        disabled=use_full_data,
    )

energy_df = load_prepared_data(use_full_data=use_full_data, sample_rows=sample_rows)
feature_columns = available_model_features(energy_df)
forecast_feature_columns = available_forecast_features(energy_df)

with st.sidebar:
    min_date = energy_df["datetime"].dt.date.min()
    max_date = energy_df["datetime"].dt.date.max()
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    mask = energy_df["datetime"].dt.date.between(start_date, end_date)
    filtered_df = energy_df.loc[mask].copy()
else:
    filtered_df = energy_df.copy()

summary = summarize_dataset(filtered_df)
full_summary = summarize_dataset(energy_df)

st.info(
    "Working CSV coverage: "
    f"{full_summary['start']:%Y-%m-%d} to {full_summary['end']:%Y-%m-%d}. "
    "The full public source dataset may contain a longer period, so final reporting should use the local CSV dates unless the dataset file is replaced."
)

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{summary['rows']:,}")
metric_cols[1].metric("Average Power", f"{summary['mean_global_active_power']:.3f} kW")
metric_cols[2].metric("Peak Power", f"{summary['max_global_active_power']:.3f} kW")
metric_cols[3].metric("Missing Values", f"{summary['missing_values']:,}")

view_options = [
        "Overview",
        "Clustering",
        "Regression",
        "Classification",
        "Business Insights",
]

selected_view = st.radio(
    "Dashboard section",
    options=view_options,
    horizontal=True,
    label_visibility="collapsed",
)

if selected_view == "Overview":
    st.subheader("Consumption Overview")
    st.plotly_chart(
        px.line(
            filtered_df,
            x="datetime",
            y="global_active_power",
            labels={"global_active_power": "Global active power (kW)", "datetime": "Date"},
        ),
        width="stretch",
    )

    left, right = st.columns(2)
    hourly_profile = filtered_df.groupby("hour", as_index=False)["global_active_power"].mean()
    left.plotly_chart(
        px.bar(
            hourly_profile,
            x="hour",
            y="global_active_power",
            title="Average Power by Hour",
            labels={"global_active_power": "Average power (kW)"},
        ),
        width="stretch",
    )

    metering_cols = [
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
        "unmetered_energy_wh",
    ]
    metering_totals = filtered_df[metering_cols].sum().reset_index()
    metering_totals.columns = ["source", "energy_wh"]
    right.plotly_chart(
        px.pie(
            metering_totals,
            names="source",
            values="energy_wh",
            title="Estimated Energy Share",
        ),
        width="stretch",
    )

    st.subheader("Weekday vs Weekend Consumption")
    wk_df = filtered_df.copy()
    wk_df["day_type"] = wk_df["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    st.plotly_chart(
        px.box(
            wk_df,
            x="hour",
            y="global_active_power",
            color="day_type",
            title="Power Distribution by Hour: Weekday vs Weekend",
            labels={
                "global_active_power": "Global active power (kW)",
                "hour": "Hour of day",
                "day_type": "Day type",
            },
            color_discrete_map={"Weekday": "#1f77b4", "Weekend": "#ff7f0e"},
        ),
        width="stretch",
    )

elif selected_view == "Clustering":
    clustering_metrics, clustered_df, cluster_profile, submetering_df = run_clustering_section(
        filtered_df,
        feature_columns,
    )
    st.subheader("Clustering: Usage Behavior Segments")
    st.write(
        "The PCA plot shows the same clustering result in two reduced dimensions. "
        "This is easier to interpret than plotting power against current intensity, because those two measurements are almost perfectly correlated."
    )
    st.dataframe(clustering_metrics, width="stretch")

    st.markdown("**K-Means Cluster Profile**")
    st.dataframe(cluster_profile, width="stretch")

    st.plotly_chart(
        px.scatter(
            clustered_df,
            x="pca_1",
            y="pca_2",
            color="kmeans_cluster",
            hover_data=[
                "datetime",
                "global_active_power",
                "global_intensity",
                "sub_metering_total_wh",
            ],
            labels={
                "pca_1": "PCA component 1",
                "pca_2": "PCA component 2",
            },
            title="K-Means Clusters Projected with PCA",
        ),
        width="stretch",
    )

    st.subheader("Sub-metering Breakdown by Cluster")
    st.write("Average energy (Wh/hour) per appliance category in each cluster — shows *what* is being used in each usage pattern.")
    st.plotly_chart(
        px.bar(
            submetering_df,
            x="kmeans_cluster",
            y="avg_wh",
            color="source",
            barmode="group",
            title="Average Sub-metering Energy by Cluster",
            labels={"avg_wh": "Average energy (Wh)", "kmeans_cluster": "Cluster", "source": "Metering source"},
        ),
        width="stretch",
    )

    st.subheader("Cluster Temporal Heatmap")
    st.write("Average power by hour and day of week per cluster — reveals *when* each usage pattern occurs.")
    import plotly.graph_objects as go  # noqa: WPS433
    from plotly.subplots import make_subplots  # noqa: WPS433

    def _cluster_label(row):
        if row["avg_power_kw"] == cluster_profile["avg_power_kw"].max():
            return "Peak Usage"
        elif row["weekend_share"] >= 0.5:
            return "Low Usage — Weekend"
        else:
            return "Low Usage — Weekday"

    cluster_label_map = {
        int(row["kmeans_cluster"]): _cluster_label(row)
        for _, row in cluster_profile.iterrows()
    }

    label_cols = st.columns(len(cluster_label_map))
    label_icons = {"Peak Usage": "🔴", "Low Usage — Weekend": "🟡", "Low Usage — Weekday": "🔵"}
    for col, (cid, label) in zip(label_cols, sorted(cluster_label_map.items())):
        col.info(f"{label_icons.get(label, '')} **Cluster {cid}** — {label}")

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    clusters = sorted(clustered_df["kmeans_cluster"].unique())
    heatmap_fig = make_subplots(
        rows=1,
        cols=len(clusters),
        subplot_titles=[f"Cluster {c} — {cluster_label_map.get(c, '')}" for c in clusters],
        shared_yaxes=True,
    )
    for col_idx, cluster_id in enumerate(clusters, start=1):
        subset = clustered_df[clustered_df["kmeans_cluster"] == cluster_id]
        pivot = (
            subset.groupby(["hour", "day_of_week"])["global_active_power"]
            .mean()
            .unstack(level="day_of_week")
            .reindex(index=range(24), columns=range(7), fill_value=0)
        )
        heatmap_fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=day_labels,
                y=list(range(24)),
                colorscale="YlOrRd",
                showscale=(col_idx == len(clusters)),
                colorbar={"title": "Avg kW"} if col_idx == len(clusters) else None,
            ),
            row=1,
            col=col_idx,
        )
    heatmap_fig.update_layout(
        height=450,
        yaxis={"autorange": "reversed", "title": "Hour of day"},
    )
    st.plotly_chart(heatmap_fig, width="stretch")

elif selected_view == "Regression":
    regression_metrics, pred_df = run_regression_section(
        filtered_df,
        forecast_feature_columns,
    )
    st.subheader("Regression: Consumption Prediction")
    st.dataframe(regression_metrics.sort_values("rmse"), width="stretch")
    st.plotly_chart(
        px.bar(
            regression_metrics,
            x="model_name",
            y="rmse",
            title="Regression RMSE Comparison",
            labels={"model_name": "Model", "rmse": "RMSE"},
        ),
        width="stretch",
    )

    st.subheader("Actual vs Predicted — Polynomial Ridge (Test Set)")
    st.write("Gaps between lines show where the model struggles: winter evening peaks and unusual weekends.")
    st.plotly_chart(
        px.line(
            pred_df.melt(id_vars="datetime", value_vars=["actual", "predicted"],
                         var_name="series", value_name="power_kw"),
            x="datetime",
            y="power_kw",
            color="series",
            labels={"power_kw": "Global active power (kW)", "datetime": "Date", "series": ""},
            color_discrete_map={"actual": "#1f77b4", "predicted": "#ff7f0e"},
        ).update_traces(opacity=0.8),
        width="stretch",
    )

elif selected_view == "Classification":
    classification_metrics, report_text, feature_importance_df = run_classification_section(
        filtered_df,
        feature_columns,
    )
    st.subheader("Classification: High vs Normal Consumption")
    st.dataframe(classification_metrics.sort_values("accuracy", ascending=False), width="stretch")
    st.text(report_text)

    if not feature_importance_df.empty:
        st.subheader("Random Forest — Feature Importance")
        st.write("Which features most strongly predict whether an hour is high consumption?")
        st.plotly_chart(
            px.bar(
                feature_importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance for High Consumption Classification",
                labels={"importance": "Importance score", "feature": "Feature"},
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            width="stretch",
        )

elif selected_view == "Business Insights":
    pca_summary = run_pca_section(filtered_df, feature_columns)
    st.subheader("Business Insights")
    peak_hour = (
        filtered_df.groupby("hour")["global_active_power"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    high_share = filtered_df["high_consumption"].mean() * 100

    insight_cols = st.columns(2)
    insight_cols[0].metric("Peak Average Hour", f"{peak_hour}:00")
    insight_cols[1].metric("High Consumption Share", f"{high_share:.1f}%")

    st.write(
        "The strongest dashboard story is to identify peak hours and inspect which metering "
        "sources contribute most. These results can support practical recommendations such as "
        "shifting heavy appliance usage away from peak periods."
    )

    st.dataframe(pca_summary, width="stretch")
