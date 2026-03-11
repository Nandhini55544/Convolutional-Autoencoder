# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Overview and Dataset

In practical scenarios, images often contain noise that degrades the performance of computer vision models. A convolutional autoencoder learns compressed representations of images and reconstructs them, which can be used to remove noise.

* **Dataset:** MNIST (28×28 grayscale images of handwritten digits)
* **Noise:** Gaussian noise will be added to simulate real-world scenarios

## Implementation Steps

### Step 1: Setup Environment

Import required libraries: PyTorch, torchvision, matplotlib, and others for data handling and visualization.

### Step 2: Load Dataset

Download the MNIST dataset and apply transformations to convert images to tensors suitable for training.

### Step 3: Introduce Noise

Add Gaussian noise to the training and testing images using a custom noise-adding function.

### Step 4: Define Autoencoder Architecture

* **Encoder:** Convolutional layers (Conv2D) with ReLU activations and MaxPooling
* **Decoder:** Transposed convolutional layers (ConvTranspose2D) with ReLU and Sigmoid activations to reconstruct the image

### Step 5: Prepare Training

* Initialize the autoencoder model
* Define Mean Squared Error (MSE) as the loss function
* Choose Adam optimizer for training

### Step 6: Model Training

Train the autoencoder using the noisy images as input and the original clean images as the target. Track the loss over epochs to monitor learning.

### Step 7: Evaluate and Visualize

* Compare the original, noisy, and denoised images
* Visualize results to assess the model’s performance in removing noise

## PROGRAM
### Name: Nandhini M
### Register Number: 212224040211

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder (Compress image)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 28x28 -> 14x14

            nn.Conv2d(16, 32, 3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # 14x14 -> 7x7
        )

        # Decoder (Reconstruct image)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # 7x7 -> 14x14
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 2, stride=2),   # 14x14 -> 28x28
            nn.Sigmoid()  # Output pixels between 0 and 1
        )

    def forward(self, x):

        x = self.encoder(x)   # Encode image (compress)
        x = self.decoder(x)   # Decode image (reconstruct)

        return x
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
summary(model, input_size=(1, 28, 28))
def train(model, loader, criterion, optimizer, epochs=5):
    
    model.train()  # set model to training mode
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, _ in loader:
            
            images = images.to(device)
            
            # Add noise to images
            noisy_images = add_noise(images).to(device)
            
            # Forward pass
            outputs = model(noisy_images)
            
            # Calculate loss between reconstructed image and original image
            loss = criterion(outputs, images)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")
def visualize_denoising(model, loader, num_images=3):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: Nandhini M")
    print("Register Number: 212224040211")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

## OUTPUT

### Model Summary

<img width="515" height="403" alt="image" src="https://github.com/user-attachments/assets/efeb0aad-7bf3-42e4-a2b1-f8cb6e4b896a" />

### Original vs Noisy Vs Reconstructed Image

<img width="1199" height="634" alt="image" src="https://github.com/user-attachments/assets/a10d5015-063f-4a66-a981-d363c3b9f501" />


## RESULT

The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
