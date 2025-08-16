import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/raw/continuous_factory_process.csv")

df.info(verbose=True)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame:
    1. Select the first 71 columns
    2. Drop columns containing 'Setpoint'
    3. Convert 'time_stamp' to datetime
    4. Set 'time_stamp' as index
    """
    # Step 1: Select first 71 columns
    df = df.iloc[:, :71]

    # Step 2: Drop 'Setpoint' columns
    df = df.loc[:, ~df.columns.str.contains("Setpoint")]

    # Step 3: Convert 'time_stamp' to datetime
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    # Step 4: Set 'time_stamp' as index
    df = df.set_index("time_stamp")

    return df


processed_df = preprocess_df(df)

processed_df.info()

# Save the processed DataFrame to a CSV file
processed_df.to_pickle("../../data/interim/data_processed.pkl")
