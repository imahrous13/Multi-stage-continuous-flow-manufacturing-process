import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_pickle("../../data/interim/data_with_features.pkl")


def analyze_time_series_models(df: pd.DataFrame, target_column: str):
    """
    Analyzes multiple regression models for a time series prediction problem.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column to predict.
    """

    # --- 1. Data Preparation ---
    # Assuming the index is already a datetime or represents the time series order
    # Drop rows with missing target values
    df = df.dropna(subset=[target_column])

    # Identify feature columns (all columns except the target)
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    # --- 2. Time Series Split ---
    # Use a simple time-based split (e.g., last 20% for testing)
    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # --- 3. Model Selection ---
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    results = {}

    print("Starting model training and evaluation...")

    # --- 4. Training and Prediction ---
    for model_name, model in models.items():
        print(f"  - Training {model_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # --- 5. Evaluation ---
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[model_name] = {"MAE": mae, "R2 Score": r2, "Predictions": predictions}

        print(f"    - {model_name} finished. MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    # --- 6. Visualization ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(15, 6 * len(models)))

    if len(models) == 1:
        axes = [axes]  # Ensure axes is iterable for single model case

    for i, (model_name, metrics) in enumerate(results.items()):
        ax = axes[i]

        # Plotting the actual vs. predicted values
        ax.plot(
            y_test.index,
            y_test.values,
            label="Actual Values",
            color="blue",
            linewidth=2,
        )
        ax.plot(
            y_test.index,
            metrics["Predictions"],
            label="Predicted Values",
            color="red",
            linestyle="--",
            linewidth=2,
        )

        ax.set_title(f"Actual vs. Predicted Values for {model_name}", fontsize=16)
        ax.set_xlabel("Time Series Index", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    # Create a summary table for easy comparison
    summary_df = pd.DataFrame(
        {
            "MAE": [results[m]["MAE"] for m in models.keys()],
            "R2 Score": [results[m]["R2 Score"] for m in models.keys()],
        },
        index=models.keys(),
    )

    print("\n--- Model Performance Summary ---")
    print(summary_df.sort_values(by="R2 Score", ascending=False))

    return results


# Example usage (assuming 'df' is your final dataframe)
# df = pd.read_csv('your_data.csv') # Or however you load your data
analyze_time_series_models(df, "Stage1.Output.Measurement0.U.Actual")
