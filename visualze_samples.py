import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import random
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from collections import defaultdict

def visualize_samples_from_imagefolder(dataset, i=1, class_names=None, figsize=(15, 5), class_range=None):
    """
    Visualize `i` samples from each class in a PyTorch ImageFolder dataset within a specified class range.
    
    Args:
        dataset (ImageFolder): A PyTorch ImageFolder dataset object.
        i (int): Number of samples to visualize per class.
        class_names (list, optional): List of class names corresponding to class indices. If None, uses dataset.classes.
        figsize (tuple): Size of the figure for visualization.
        class_range (tuple, optional): Range of classes to visualize (start, end). If None, shows all classes.
    
    Returns:
        None: Displays the plot.
    """
    # Default to class names from the dataset
    class_names = class_names or dataset.classes

    # Determine which classes to include based on the range
    if class_range:
        start, end = class_range
        selected_classes = range(start, end + 1)
    else:
        selected_classes = range(len(class_names))
    
    # Group file paths by class
    class_to_indices = {cls: [] for cls in selected_classes}
    for idx, (path, label) in enumerate(dataset.samples):
        if label in class_to_indices:
            class_to_indices[label].append(idx)
    
    # Randomly select `i` samples from each selected class
    selected_indices = {cls: random.sample(class_to_indices[cls], i) for cls in class_to_indices if len(class_to_indices[cls]) >= i}
    
    # Load the selected samples
    selected_samples = {cls: [dataset[idx][0] for idx in indices] for cls, indices in selected_indices.items()}
    
    # Plot the selected samples
    fig, axes = plt.subplots(i, len(selected_samples), figsize=figsize)
    for col, (cls, images) in enumerate(selected_samples.items()):
        for row, img in enumerate(images):
            ax = axes[row, col] if i > 1 else axes[col]
            img = ToTensor()(img) if not isinstance(img, torch.Tensor) else img
            ax.imshow(img.permute(1, 2, 0))  # Rearrange dimensions for RGB
            ax.axis('off')  # Turn off axis for clean visuals
            if row == 0:
                ax.set_title(class_names[cls], fontsize=8, rotation=15)
    
    # Remove white spaces between images
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.tight_layout(pad=0)
    plt.show()

def visualize_augmentations(dataset, augmentations, i=3, j=5, figsize=(15, 5)):
    """
    Visualize `i` samples from the dataset with `j` augmentations applied to each sample.
    
    Args:
        dataset (Dataset): A PyTorch dataset object.
        augmentations (transforms.Compose): A composed set of data augmentations.
        i (int): Number of samples to visualize (rows).
        j (int): Number of augmentations per sample (columns).
        figsize (tuple): Figure size for the output grid.
    
    Returns:
        None: Displays the plot.
    """
    # Create a grid for visualization
    fig, axes = plt.subplots(i, j, figsize=figsize)

    # Randomly select `i` samples from the dataset
    sample_indices = random.sample(range(len(dataset)), i)
    
    for row_idx, sample_idx in enumerate(sample_indices):
        # Get the original image from the dataset
        original_img, label = dataset[sample_idx]
        
        for col_idx in range(j):
            # Apply augmentations to the original image
            augmented_img = augmentations(original_img)
            
            # Plot the augmented image
            ax = axes[row_idx, col_idx] if i > 1 else axes[col_idx]
            ax.imshow(augmented_img.permute(1, 2, 0), cmap="gray")  # Adjust for RGB/Grayscale
            ax.axis('off')
            
            # Add label to the first column
            if col_idx == 0:
                ax.set_title(f"Class {label}", fontsize=8, rotation=15)
    
    # Adjust layout
    plt.tight_layout(pad=0.5)
    plt.show()

def show_class_samples_with_transformations(dataset, max_classes=10, transformation_dict=None):
    """
    Displays random samples from each class with transformations applied.

    Parameters:
    - dataset: A PyTorch ImageFolder dataset.
    - max_classes: Maximum number of classes to show.
    - transformation_dict: Dictionary with transformation names and transformations.
    """
    # Get unique classes and their indices in the dataset
    class_to_idx = dataset.class_to_idx
    max_classes = min(max_classes, len(class_to_idx))  # Ensure max_classes doesn't exceed dataset classes
    print(max_classes)
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        class_to_indices[label].append(idx)

    # Limit to max_classes and select one random sample per class
    chosen_samples = {}
    for label, indices in random.sample(list(class_to_indices.items()), min(max_classes, len(class_to_indices))):
        random_idx = random.choice(indices)  # Randomly pick one index for the class
        chosen_samples[label] = dataset[random_idx][0]  # Add the image to chosen samples

    # Convert the chosen samples into a sorted list (by class index)
    sorted_samples = sorted(chosen_samples.items(), key=lambda x: x[0])
    images = [img for _, img in sorted_samples]
    class_names = [dataset.classes[label] for label, _ in sorted_samples]

    # If no transformations are provided, use only the original images
    if transformation_dict is None:
        transformation_dict = {"Original": None}

    # Prepare grid dimensions
    n_rows = len(transformation_dict)
    n_cols = len(images)

    # Create the figure
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=((n_cols + 1) * 3, n_rows * 3), dpi=100)
    axes = axes if n_rows > 1 else [axes]  # Ensure axes is always iterable

    for row_idx, (transform_name, transform) in enumerate(transformation_dict.items()):
        for col_idx, (img, class_name) in enumerate(zip(images, class_names)):
            ax = axes[row_idx][col_idx + 1] if n_rows > 1 else axes[col_idx + 1]

            # Apply transformation if it's not None
            transformed_img = transform(img) if transform else img

            # Convert image to numpy for plotting
            if not isinstance(transformed_img, torch.Tensor):
                transformed_img = ToTensor()(transformed_img)  # Convert PIL image to tensor
            transformed_img = (
                transformed_img.permute(1, 2, 0).numpy()
                if transformed_img.ndimension() == 3
                else transformed_img.numpy()
            )

            # Plot the image
            ax.imshow(transformed_img)
            if row_idx == 0:
                ax.set_title(class_name, fontsize=10)  # Add title for the first row
            ax.axis("off")

        # Add transformation label in the first column
        label_ax = axes[row_idx][0] if n_rows > 1 else axes[0]
        label_ax.text(
            0.5, 0.5, transform_name, fontsize=12, ha="center", va="center", rotation=0
        )
        label_ax.axis("off")

    plt.tight_layout()
    plt.show()
