import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("../../data/interim/data_processed.pkl")
df.info(verbose=True)


def plot_machine_properties(df):
    """
    Plot machine properties by grouping the same property across all machines.
    Example: RawMaterial.Property1 -> Machine1, Machine2, Machine3 together.
    """
    # Extract all machine-related properties
    machine_cols = [col for col in df.columns if col.startswith("Machine")]

    # Get property names without machine prefixes
    properties = sorted(set(col.split(".", 1)[1] for col in machine_cols))

    for prop in properties:
        # Get columns for this property across all machines
        prop_cols = [col for col in machine_cols if col.endswith(prop)]
        if not prop_cols:
            continue

        plt.figure(figsize=(12, 6))
        for col in prop_cols:
            plt.plot(df.index, df[col], label=col.split(".")[0], linewidth=1)

        plt.title(f"{prop}", fontsize=14)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend(title="Machine", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


plot_machine_properties(df)


def plot_stage_outputs(df):
    """
    Create separate line plots for each Stage1.Output.Measurement column.
    """
    stage_cols = [col for col in df.columns if col.startswith("Stage1.Output")]

    for col in stage_cols:
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df[col], label=col, color="tab:blue", linewidth=1)
        plt.title(f"{col}", fontsize=14)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


plot_stage_outputs(df)
