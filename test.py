import matplotlib.pyplot as plt
from datasets import load_dataset

data = load_dataset("ashraq/fashion-product-images-small")

# Get the first image
image = data["train"][0]["image"]

# Print image dimensions
print(f"Image dimensions (width x height): {image.size}")
print(f"Total number of pixels: {image.size[0] * image.size[1]}")

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.title(f"Image Size: {image.size[0]}x{image.size[1]} pixels")
plt.show()
