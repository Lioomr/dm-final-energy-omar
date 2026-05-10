from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


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
    page_icon="SE",
    layout="wide",
)


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
        detect_anomalies,
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
    anomaly_df = detect_anomalies(
        df,
        features=feature_columns,
        contamination=0.01,
        max_rows=50000,
    )
    pca_summary, _ = pca_feature_summary(
        df,
        features=feature_columns,
        n_components=3,
    )
    return clustering_metrics, clustered_df, regression_metrics, classification_metrics, report_text, anomaly_df, pca_summary


@st.cache_data(show_spinner="Running clustering...")
def run_clustering_section(df: pd.DataFrame, feature_columns: list[str]):
    from src.modeling import run_clustering_baselines, summarize_clusters  # noqa: WPS433

    clustering_metrics, clustered_df = run_clustering_baselines(
        df,
        features=feature_columns,
        max_rows=8000,
    )
    cluster_profile = summarize_clusters(clustered_df)
    return clustering_metrics, clustered_df, cluster_profile


@st.cache_data(show_spinner="Running regression...")
def run_regression_section(
    df: pd.DataFrame,
    forecast_feature_columns: list[str],
):
    from src.modeling import train_regression_baselines  # noqa: WPS433

    regression_metrics, _ = train_regression_baselines(
        df,
        features=forecast_feature_columns,
        target="global_active_power",
    )
    return regression_metrics


@st.cache_data(show_spinner="Running classification...")
def run_classification_section(df: pd.DataFrame, feature_columns: list[str]):
    from src.modeling import train_classification_baselines  # noqa: WPS433

    classification_metrics, _, report_text = train_classification_baselines(
        df,
        features=feature_columns,
        target="high_consumption",
    )
    return classification_metrics, report_text


@st.cache_data(show_spinner="Running anomaly detection...")
def run_anomaly_section(df: pd.DataFrame, feature_columns: list[str]):
    from src.modeling import detect_anomalies  # noqa: WPS433

    return detect_anomalies(
        df,
        features=feature_columns,
        contamination=0.01,
        max_rows=20000,
    )


@st.cache_data(show_spinner="Running PCA summary...")
def run_pca_section(df: pd.DataFrame, feature_columns: list[str]):
    from src.modeling import pca_feature_summary  # noqa: WPS433

    pca_summary, _ = pca_feature_summary(
        df,
        features=feature_columns,
        n_components=3,
    )
    return pca_summary


st.title("Smart Energy Consumption Analytics")
st.caption("Household power consumption analysis, prediction, clustering, classification, and anomaly detection.")

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
        "Anomaly Detection",
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

elif selected_view == "Clustering":
    clustering_metrics, clustered_df, cluster_profile = run_clustering_section(
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

elif selected_view == "Regression":
    regression_metrics = run_regression_section(
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

elif selected_view == "Classification":
    classification_metrics, report_text = run_classification_section(
        filtered_df,
        feature_columns,
    )
    st.subheader("Classification: High vs Normal Consumption")
    st.dataframe(classification_metrics.sort_values("accuracy", ascending=False), width="stretch")
    st.text(report_text)

elif selected_view == "Anomaly Detection":
    anomaly_df = run_anomaly_section(
        filtered_df,
        feature_columns,
    )
    st.subheader("Anomaly Detection")
    st.plotly_chart(
        px.scatter(
            anomaly_df,
            x="datetime",
            y="global_active_power",
            color="anomaly_label",
            labels={
                "global_active_power": "Global active power (kW)",
                "anomaly_label": "Anomaly",
            },
        ),
        width="stretch",
    )
    st.dataframe(
        anomaly_df.sort_values("anomaly_score").head(20)[
            ["datetime", "global_active_power", "global_intensity", "anomaly_score"]
        ],
        width="stretch",
    )

elif selected_view == "Business Insights":
    anomaly_df = run_anomaly_section(filtered_df, feature_columns)
    pca_summary = run_pca_section(filtered_df, feature_columns)
    st.subheader("Business Insights")
    peak_hour = (
        filtered_df.groupby("hour")["global_active_power"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    high_share = filtered_df["high_consumption"].mean() * 100
    anomaly_share = anomaly_df["anomaly_label"].mean() * 100

    insight_cols = st.columns(3)
    insight_cols[0].metric("Peak Average Hour", f"{peak_hour}:00")
    insight_cols[1].metric("High Consumption Share", f"{high_share:.1f}%")
    insight_cols[2].metric("Detected Anomaly Share", f"{anomaly_share:.1f}%")

    st.write(
        "The strongest dashboard story is to identify peak hours, inspect which metering "
        "sources contribute most, then use anomaly detection to flag unusual periods for "
        "review. These results can support practical recommendations such as shifting "
        "heavy appliance usage away from peak periods and investigating repeated spikes."
    )

    st.dataframe(pca_summary, width="stretch")
