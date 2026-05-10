from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "DataSet" / "household_power_consumption.csv"

RAW_NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

NUMERIC_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

MODEL_FEATURE_COLUMNS = [
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_total_wh",
    "unmetered_energy_wh",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

FORECAST_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1_power",
    "lag_2_power",
    "lag_24_power",
    "rolling_3_power_mean",
    "rolling_24_power_mean",
]


def _clean_column_name(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def load_raw_energy_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load the raw CSV with missing-value markers handled consistently."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return pd.read_csv(
        data_path,
        nrows=nrows,
        na_values=["?"],
        low_memory=False,
    )


def clean_energy_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean data types and create a single datetime column."""
    df = raw_df.copy()
    df.columns = [_clean_column_name(column) for column in df.columns]

    required_columns = {"date", "time", *NUMERIC_COLUMNS}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["datetime"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )

    # Time interpolation is appropriate here because measurements are ordered
    # minute-by-minute. Remaining edge gaps are filled in both directions.
    df = df.set_index("datetime")
    df[NUMERIC_COLUMNS] = (
        df[NUMERIC_COLUMNS]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    df = df.reset_index()

    return add_energy_features(df)


def add_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable time and consumption features."""
    prepared = df.copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"])

    prepared["hour"] = prepared["datetime"].dt.hour
    prepared["day_of_week"] = prepared["datetime"].dt.dayofweek
    prepared["month"] = prepared["datetime"].dt.month
    prepared["year"] = prepared["datetime"].dt.year
    prepared["is_weekend"] = prepared["day_of_week"].isin([5, 6]).astype(int)

    prepared["sub_metering_total_wh"] = prepared[
        ["sub_metering_1", "sub_metering_2", "sub_metering_3"]
    ].sum(axis=1)

    # Global active power is kW sampled once per minute.
    # kW * 1000 converts to W, then / 60 estimates Wh for that minute.
    prepared["active_energy_wh"] = prepared["global_active_power"] * 1000 / 60
    prepared["unmetered_energy_wh"] = (
        prepared["active_energy_wh"] - prepared["sub_metering_total_wh"]
    ).clip(lower=0)

    return prepared


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features for short-term forecasting."""
    featured = df.copy().sort_values("datetime").reset_index(drop=True)
    featured["lag_1_power"] = featured["global_active_power"].shift(1)
    featured["lag_2_power"] = featured["global_active_power"].shift(2)
    featured["lag_24_power"] = featured["global_active_power"].shift(24)
    featured["rolling_3_power_mean"] = (
        featured["global_active_power"].shift(1).rolling(window=3, min_periods=1).mean()
    )
    featured["rolling_24_power_mean"] = (
        featured["global_active_power"].shift(1).rolling(window=24, min_periods=1).mean()
    )

    lag_columns = [
        "lag_1_power",
        "lag_2_power",
        "lag_24_power",
        "rolling_3_power_mean",
        "rolling_24_power_mean",
    ]
    featured[lag_columns] = featured[lag_columns].bfill().ffill()
    return featured


def add_consumption_label(
    df: pd.DataFrame,
    target_column: str = "global_active_power",
    quantile: float = 0.75,
) -> pd.DataFrame:
    """Create a high-consumption label for classification experiments."""
    labelled = df.copy()
    threshold = labelled[target_column].quantile(quantile)
    labelled["high_consumption"] = (labelled[target_column] >= threshold).astype(int)
    labelled.attrs["high_consumption_threshold"] = float(threshold)
    return labelled


def resample_energy_data(df: pd.DataFrame, frequency: str = "h") -> pd.DataFrame:
    """Aggregate minute-level data into a smaller time grain."""
    indexed = df.copy()
    indexed["datetime"] = pd.to_datetime(indexed["datetime"])
    indexed = indexed.set_index("datetime").sort_index()

    aggregations = {
        "global_active_power": "mean",
        "global_reactive_power": "mean",
        "voltage": "mean",
        "global_intensity": "mean",
        "sub_metering_1": "sum",
        "sub_metering_2": "sum",
        "sub_metering_3": "sum",
        "sub_metering_total_wh": "sum",
        "active_energy_wh": "sum",
        "unmetered_energy_wh": "sum",
    }

    hourly = indexed.resample(frequency).agg(aggregations).dropna().reset_index()
    return add_energy_features(hourly)


def prepare_energy_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
    nrows: int | None = None,
    frequency: str | None = "h",
    label_quantile: float = 0.75,
) -> pd.DataFrame:
    """End-to-end preparation helper used by the notebook and dashboard."""
    raw_df = load_raw_energy_data(data_path=data_path, nrows=nrows)
    clean_df = clean_energy_data(raw_df)

    if frequency:
        clean_df = resample_energy_data(clean_df, frequency=frequency)

    clean_df = add_time_series_features(clean_df)
    return add_consumption_label(clean_df, quantile=label_quantile)


def available_model_features(df: pd.DataFrame) -> list[str]:
    """Return model features that are present in the provided dataframe."""
    return [column for column in MODEL_FEATURE_COLUMNS if column in df.columns]


def available_forecast_features(df: pd.DataFrame) -> list[str]:
    """Return non-leaky forecasting features available in the dataframe."""
    return [column for column in FORECAST_FEATURE_COLUMNS if column in df.columns]


def reduce_memory_usage(df: pd.DataFrame, exclude: Iterable[str] = ("datetime",)) -> pd.DataFrame:
    """Downcast numeric columns after preparation to reduce RAM use."""
    optimized = df.copy()
    excluded = set(exclude)

    for column in optimized.columns:
        if column in excluded:
            continue
        if pd.api.types.is_float_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="float")
        elif pd.api.types.is_integer_dtype(optimized[column]):
            optimized[column] = pd.to_numeric(optimized[column], downcast="integer")

    return optimized


def summarize_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Create a compact summary for quick notebook and dashboard checks."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start": df["datetime"].min(),
        "end": df["datetime"].max(),
        "missing_values": int(df.isna().sum().sum()),
        "mean_global_active_power": float(df["global_active_power"].mean()),
        "max_global_active_power": float(df["global_active_power"].max()),
        "high_consumption_threshold": df.attrs.get("high_consumption_threshold"),
    }
