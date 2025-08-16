import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("../../data/interim/data_processed.pkl")

df["Stage1.Output.Measurement0.U.Actual"].plot()

import pandas as pd
import numpy as np


def clean_series(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """
    Clean a time series by removing extreme values (outliers).
    Outliers are replaced with NaN and filled by linear interpolation.

    Parameters
    ----------
    series : pd.Series
        Input time series data.
    z_thresh : float, optional
        Z-score threshold for detecting outliers (default=3.0).

    Returns
    -------
    pd.Series
        Cleaned time series with outliers replaced by interpolated values.
    """
    # Compute z-scores
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std

    # Identify outliers
    outliers = np.abs(z_scores) > z_thresh

    # Replace outliers with NaN
    cleaned = series.copy()
    cleaned[outliers] = np.nan

    # Fill missing values by linear interpolation
    cleaned = cleaned.interpolate(method="linear")

    return cleaned


# Clean the time series data
df["Stage1.Output.Measurement0.U.Actual"] = clean_series(
    df["Stage1.Output.Measurement0.U.Actual"]
)

df["Stage1.Output.Measurement0.U.Actual"].plot()

import pandas as pd


def add_features(
    df: pd.DataFrame, target_col: str = "Stage1.Output.Measurement0.U.Actual"
) -> pd.DataFrame:
    """
    Add engineered features to help in prediction of the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a datetime index and sensor readings.
    target_col : str
        The column to predict (default: Stage1.Output.Measurement0.U.Actual).

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered features.
    """

    df_feat = df.copy()

    # ---------------------------
    # 1. Time-based features
    # ---------------------------
    df_feat["hour"] = df_feat.index.hour
    df_feat["minute"] = df_feat.index.minute
    df_feat["second"] = df_feat.index.second

    # ---------------------------
    # 2. Lag features (for target)
    # ---------------------------
    for lag in [1, 2, 3, 5, 10]:
        df_feat[f"{target_col}_lag{lag}"] = df_feat[target_col].shift(lag)

    # ---------------------------
    # 3. Rolling statistics (for target)
    # ---------------------------
    for window in [3, 5, 10, 30]:
        df_feat[f"{target_col}_rollmean{window}"] = (
            df_feat[target_col].rolling(window).mean()
        )
        df_feat[f"{target_col}_rollstd{window}"] = (
            df_feat[target_col].rolling(window).std()
        )
        df_feat[f"{target_col}_rollmin{window}"] = (
            df_feat[target_col].rolling(window).min()
        )
        df_feat[f"{target_col}_rollmax{window}"] = (
            df_feat[target_col].rolling(window).max()
        )

    # ---------------------------
    # 4. Interaction features between machines
    # ---------------------------
    if (
        "Machine1.MotorAmperage.U.Actual" in df_feat.columns
        and "Machine2.MotorAmperage.U.Actual" in df_feat.columns
    ):
        df_feat["MotorAmperage_Ratio_1_2"] = df_feat[
            "Machine1.MotorAmperage.U.Actual"
        ] / (df_feat["Machine2.MotorAmperage.U.Actual"] + 1e-6)

    if (
        "Machine1.MaterialTemperature.U.Actual" in df_feat.columns
        and "Machine2.MaterialTemperature.U.Actual" in df_feat.columns
    ):
        df_feat["MaterialTemp_Diff_1_2"] = (
            df_feat["Machine1.MaterialTemperature.U.Actual"]
            - df_feat["Machine2.MaterialTemperature.U.Actual"]
        )

    # ---------------------------
    # 5. Fill NaN values from lags/rolling
    # ---------------------------
    df_feat = df_feat.fillna(method="bfill").fillna(method="ffill")

    return df_feat


df_with_features = add_features(df, target_col="Stage1.Output.Measurement0.U.Actual")
print(df_with_features.head())
