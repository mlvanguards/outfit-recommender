import matplotlib.pyplot as plt
from datasets import load_dataset

data = load_dataset("valex95/fashion-dataset")

# Print a sample data point
print(data["train"][0])

# Get label names from the dataset
label_names = data["train"].features["label"].names


def plot_single_image(index=0):
    """
    Plot a single image from the dataset by its index.

    Args:
        index: Index of the image to plot (default: 0)
    """
    # Get the sample at the specified index
    sample = data["train"][index]

    # Get the image and its label
    image = sample["image"]
    label_id = sample["label"]
    label_name = label_names[label_id]

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(
        f"Category: {label_name}\nImage ID: {sample['image_id'][:8]}...\nUser: {sample['user_id']}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Print some information about the image
    print("Image details:")
    print(f"  Category: {label_name}")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  User ID: {sample['user_id']}")
    print(f"  Image dimensions: {image.size}")


# Example usage
if __name__ == "__main__":
    print("\nDataset structure:")
    print(data)

    # Plot the first image in the dataset
    plot_single_image(10)
