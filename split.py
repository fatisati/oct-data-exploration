import pandas as pd
from sklearn.model_selection import train_test_split
import os

def get_image_path(dicom_path):
    return f"{os.path.basename(dicom_path)}_middle_slice_rgb.jpg"


def split_and_save_filtered(df, stratify_col, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, min_samples=30):
    """
    Filters labels with fewer than a minimum number of samples, splits the DataFrame into train, test, 
    and validation sets stratified by a specified column, and saves them.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - stratify_col (str): Column name to stratify on.
    - output_folder (str): Folder to save the splits.
    - train_ratio (float): Proportion of training data (default 0.7).
    - val_ratio (float): Proportion of validation data (default 0.15).
    - test_ratio (float): Proportion of testing data (default 0.15).
    - min_samples (int): Minimum number of samples required for a label to be included.

    Returns:
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ensure ratios sum to 1
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    
    # Filter labels with fewer than `min_samples`
    label_counts = df[stratify_col].value_counts()
    valid_labels = label_counts[label_counts >= min_samples].index
    print(valid_labels)
    filtered_df = df[df[stratify_col].isin(valid_labels)]
    
    print(f"Filtered dataset has {len(filtered_df)} samples and {len(valid_labels)} valid labels.")
    
    # Split into train + remaining
    train_df, remaining_df = train_test_split(
        filtered_df,
        test_size=(1 - train_ratio),
        stratify=filtered_df[stratify_col],
        random_state=42
    )
    
    # Adjust validation and test ratios
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    
    # Split remaining into validation and test
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=(1 - val_test_ratio),
        stratify=remaining_df[stratify_col],
        random_state=42
    )
    
    # Save the splits
    train_df.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_folder, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_folder, 'test.csv'), index=False)
    
    print(f"Data saved successfully to {output_folder}:")
    print(f"- Train set: {len(train_df)} samples")
    print(f"- Validation set: {len(val_df)} samples")
    print(f"- Test set: {len(test_df)} samples")

# Example usage
# df = pd.read_csv('your_data.csv')
# split_and_save_filtered(df, stratify_col='disease_en', output_folder='data_splits_filtered')
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(data_folder, class_col='disease_en', output_path="class_distribution_plot.png"):
    """
    Reads split CSV files, counts the number of samples per class for each split,
    and plots a bar plot sorted by 'train' count with sample numbers on top of each bar.

    Args:
        data_folder (str): Path to the folder containing split CSV files.
        output_path (str): Path to save the output plot image.

    Returns:
        None
    """
    split_counts = {}

    # Read all CSV files in the folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            split_name = file_name.replace(".csv", "")  # Extract split name (e.g., 'train', 'test', 'val')
            file_path = os.path.join(data_folder, file_name)
            data = pd.read_csv(file_path)
            # Count class occurrences
            counts = data[class_col].value_counts()
            split_counts[split_name] = counts

    # Combine counts into a DataFrame
    combined_counts = pd.DataFrame(split_counts).fillna(0).astype(int)

    # Reshape for plotting
    combined_counts_reset = combined_counts.reset_index().melt(
        id_vars='index', var_name='Split', value_name='Count'
    )
    combined_counts_reset.rename(columns={'index': 'Class'}, inplace=True)

    # Sort by 'train' count
    if 'train' in combined_counts:
        train_sorted = combined_counts['train'].sort_values(ascending=False).index
        combined_counts_reset['Class'] = pd.Categorical(
            combined_counts_reset['Class'], categories=train_sorted, ordered=True
        )

    # Create the bar plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=combined_counts_reset, x='Class', y='Count', hue='Split', dodge=True)

    # Add counts on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10)

    # Customize plot
    plt.title("Number of Samples for Each Class in Each Split (Sorted by Train Count)", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Split", fontsize=10)
    plt.tight_layout()

    # Save and show the plot
    # plt.savefig(output_path, dpi=300)
    plt.show()

import os
import shutil
import pandas as pd

def organize_images(data_folder, image_in_folder, image_out_folder):
    file_col = 'Filename'
    label_col = 'disease_en'
    """
    Organizes images into subfolders based on split and label by copying them from
    the input image folder to the output image folder.

    Args:
        data_folder (str): Path to the folder containing CSV files (e.g., train.csv, val.csv).
        image_in_folder (str): Path to the folder containing all images.
        image_out_folder (str): Path to the folder where organized images will be copied.

    Returns:
        None
    """
    # Ensure the output image folder exists
    os.makedirs(image_out_folder, exist_ok=True)

    # Process each CSV file in the data folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            split_name = file_name.replace(".csv", "")  # Extract split name (e.g., 'train', 'test', 'val')
            file_path = os.path.join(data_folder, file_name)
            
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Process each row in the CSV
            for _, row in data.iterrows():
                image_file = get_image_path(row[file_col])
                label = row[label_col]

                # Source and destination paths
                src_path = os.path.join(image_in_folder, image_file)
                dest_folder = os.path.join(image_out_folder, split_name, label)
                dest_path = os.path.join(dest_folder, image_file)

                # Ensure the destination folder exists
                os.makedirs(dest_folder, exist_ok=True)

                # Copy the image file
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                else:
                    print(f"Warning: Image file {src_path} does not exist.")

    print("Image organization complete.")
