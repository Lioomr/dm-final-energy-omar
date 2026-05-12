from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass
class RegressionResult:
    model_name: str
    mae: float
    rmse: float
    r2: float


def _sample_frame(df: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(max_rows, random_state=random_state).sort_index()


def _clean_model_matrix(df: pd.DataFrame, features: list[str], target: str | None = None):
    columns = features + ([target] if target else [])
    model_df = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    x = model_df[features]
    if target:
        return x, model_df[target]
    return x


def train_regression_baselines(
    df: pd.DataFrame,
    features: list[str],
    target: str = "global_active_power",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Compare simple regression models for energy prediction."""
    x, y = _clean_model_matrix(df, features, target)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=False,
    )

    models: dict[str, Pipeline] = {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "Polynomial Ridge": Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }

    rows: list[RegressionResult] = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append(
            RegressionResult(
                model_name=model_name,
                mae=float(mean_absolute_error(y_test, predictions)),
                rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
                r2=float(r2_score(y_test, predictions)),
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows]), models


def train_classification_baselines(
    df: pd.DataFrame,
    features: list[str],
    target: str = "high_consumption",
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, Pipeline], str]:
    """Train interpretable classifiers for high-vs-normal consumption."""
    x, y = _clean_model_matrix(df, features, target)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y if y.nunique() > 1 else None,
        random_state=42,
    )

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }

    rows = []
    reports = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_test, predictions)),
            }
        )
        reports.append(
            f"{model_name}\n"
            + classification_report(y_test, predictions, zero_division=0)
        )

    return pd.DataFrame(rows), models, "\n\n".join(reports)


def run_clustering_baselines(
    df: pd.DataFrame,
    features: list[str],
    max_rows: int = 20000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run K-Means and DBSCAN on a sampled feature matrix."""
    sampled = _sample_frame(df, max_rows=max_rows)
    x = _clean_model_matrix(sampled, features)
    scaled = StandardScaler().fit_transform(x)

    kmeans_rows = []
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        score = silhouette_score(scaled, labels) if len(set(labels)) > 1 else np.nan
        kmeans_rows.append({"algorithm": "K-Means", "clusters": k, "silhouette": score})

    dbscan = DBSCAN(eps=1.5, min_samples=20)
    dbscan_labels = dbscan.fit_predict(scaled)
    dbscan_cluster_count = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_score = (
        silhouette_score(scaled, dbscan_labels)
        if dbscan_cluster_count > 1
        else np.nan
    )
    dbscan_row = {
        "algorithm": "DBSCAN",
        "clusters": dbscan_cluster_count,
        "silhouette": dbscan_score,
        "noise_points": int((dbscan_labels == -1).sum()),
    }

    pca = PCA(n_components=2, random_state=42)
    pca_coordinates = pca.fit_transform(scaled)

    clustered = sampled.loc[x.index].copy()
    clustered["kmeans_cluster"] = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    ).fit_predict(scaled)
    clustered["dbscan_cluster"] = dbscan_labels
    clustered["pca_1"] = pca_coordinates[:, 0]
    clustered["pca_2"] = pca_coordinates[:, 1]

    metrics = pd.concat(
        [pd.DataFrame(kmeans_rows), pd.DataFrame([dbscan_row])],
        ignore_index=True,
    )
    return metrics, clustered


def summarize_clusters(
    clustered_df: pd.DataFrame,
    cluster_column: str = "kmeans_cluster",
) -> pd.DataFrame:
    """Create an interpretable profile table for each cluster."""
    profile_columns = [
        "global_active_power",
        "global_intensity",
        "sub_metering_total_wh",
        "unmetered_energy_wh",
        "hour",
        "is_weekend",
    ]
    available_columns = [
        column for column in profile_columns if column in clustered_df.columns
    ]

    profile = (
        clustered_df.groupby(cluster_column)[available_columns]
        .mean()
        .rename(
            columns={
                "global_active_power": "avg_power_kw",
                "global_intensity": "avg_intensity_a",
                "sub_metering_total_wh": "avg_metered_wh",
                "unmetered_energy_wh": "avg_unmetered_wh",
                "hour": "avg_hour",
                "is_weekend": "weekend_share",
            }
        )
        .round(3)
    )
    profile["records"] = clustered_df.groupby(cluster_column).size()
    return profile.reset_index()


def cluster_submetering_breakdown(
    clustered_df: pd.DataFrame,
    cluster_column: str = "kmeans_cluster",
) -> pd.DataFrame:
    """Return average Wh per appliance category per cluster in long format."""
    source_columns = {
        "sub_metering_1": "Kitchen",
        "sub_metering_2": "Laundry",
        "sub_metering_3": "HVAC / Water Heater",
        "unmetered_energy_wh": "Other / Unmetered",
    }
    available = {col: label for col, label in source_columns.items() if col in clustered_df.columns}
    rows = []
    for col, label in available.items():
        means = clustered_df.groupby(cluster_column)[col].mean()
        for cluster_id, avg in means.items():
            rows.append({"kmeans_cluster": cluster_id, "source": label, "avg_wh": round(avg, 3)})
    return pd.DataFrame(rows)


def detect_anomalies(
    df: pd.DataFrame,
    features: list[str],
    contamination: float = 0.01,
    max_rows: int = 100000,
) -> pd.DataFrame:
    """Detect unusual consumption periods with Isolation Forest."""
    sampled = _sample_frame(df, max_rows=max_rows)
    x = _clean_model_matrix(sampled, features)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=150,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    labels = pipeline.fit_predict(x)
    scores = pipeline.decision_function(x)

    anomalies = sampled.loc[x.index].copy()
    anomalies["anomaly_label"] = np.where(labels == -1, 1, 0)
    anomalies["anomaly_score"] = scores
    return anomalies


def pca_feature_summary(
    df: pd.DataFrame,
    features: list[str],
    n_components: int = 3,
    max_rows: int = 50000,
) -> tuple[pd.DataFrame, PCA]:
    """Summarize variance captured by PCA components."""
    sampled = _sample_frame(df, max_rows=max_rows)
    x = _clean_model_matrix(sampled, features)
    scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(scaled)

    summary = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return summary, pca
