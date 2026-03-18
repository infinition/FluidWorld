import os
import numpy as np
import torchvision.datasets as datasets

# Create the data directory if it does not exist
os.makedirs('data', exist_ok=True)

# Download the official MNIST dataset
print("Downloading MNIST...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True)

# Extract images and labels as NumPy arrays
images = mnist_train.data.numpy()  # Format: (60000, 28, 28) - type: uint8
labels = mnist_train.targets.numpy()  # Format: (60000,) - type: int64

# Save in .npz format with the correct keys
output_path = './data/mnist_labeled.npz'
np.savez_compressed(output_path, images=images, labels=labels)

print(f"Success! Generated file: {output_path}")
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")