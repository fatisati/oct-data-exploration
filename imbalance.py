import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_label_distribution_stacked(input_folder, label_col, label_mapping=None):
    """
    Reads train, test, and val CSV files, calculates the percentage of each label,
    and creates a single stacked bar plot showing proportions for each split.

    Args:
        input_folder (str): Path to the folder containing train.csv, test.csv, and val.csv.
        label_col (str): Name of the column containing labels.
        label_mapping (dict): Optional dictionary to map long labels to shorter ones.

    Returns:
        None
    """
    # Initialize a dictionary to store data for each split
    splits = {}
    total_samples = {}

    for split_name in ['train', 'test', 'val']:
        file_path = os.path.join(input_folder, f"{split_name}.csv")
        if os.path.exists(file_path):
            # Read the CSV file
            data = pd.read_csv(file_path)
            if label_col not in data.columns:
                raise ValueError(f"Column '{label_col}' not found in {split_name}.csv")
            
            # Calculate label counts and percentages
            label_counts = data[label_col].value_counts()
            label_percentages = label_counts / label_counts.sum() * 100
            splits[split_name] = label_percentages
            total_samples[split_name] = label_counts.sum()
        else:
            print(f"Warning: {split_name}.csv not found in the folder.")
            splits[split_name] = pd.Series()
            total_samples[split_name] = 0

    # Create a DataFrame with proportions for each split
    split_df = pd.DataFrame(splits).fillna(0).sort_index()

    # Apply label mapping if provided
    if label_mapping:
        split_df.index = split_df.index.map(lambda x: label_mapping.get(x, x))

    # Plot stacked bar chart
    labels = split_df.index  # Unique labels (mapped if mapping provided)
    splits = split_df.columns  # Split names (train, test, val)

    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(splits))  # Track the bottom position for stacking

    # Colors for each label
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    for i, label in enumerate(labels):
        values = split_df.loc[label]
        ax.bar(
            splits,
            values,
            bottom=bottom,
            color=colors[i],
            label=label
        )

        # Add percentage and sample count text next to each bar segment
        for j, (val, split_name) in enumerate(zip(values, splits)):
            if val > 0:  # Only display if percentage > 0
                count = int(val * total_samples[split_name] / 100)
                ax.text(j, bottom[j] + val / 2, f"{count} ({val:.1f}%)", ha='center', va='center', fontsize=10)
        bottom += values  # Update bottom for stacking

    # Add total sample count on top of each split bar
    for j, split_name in enumerate(splits):
        ax.text(
            j,
            bottom[j] + 2,
            f"Total: {total_samples[split_name]}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    # Add legend
    ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Customize plot
    ax.set_title("Label Distribution Across Splits", fontsize=16)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Splits", fontsize=12)
    ax.set_ylim(0, max(bottom) + 10)  # Adjust ylim to fit total counts
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.tight_layout()

    # Show plot
    plt.show()
