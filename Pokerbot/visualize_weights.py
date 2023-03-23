# visualize_weights.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleFeedForward
from torch.utils.data import DataLoader

# Hyperparameters
input_size = 784
num_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and its state_dict
model = SimpleFeedForward(input_size, num_classes).to(device)
model.load_state_dict(torch.load("mnist_model.pth"))

# Get the weights and biases of the model's layers
fc1_weights = model.fc1.weight.data.cpu().numpy()
fc1_biases = model.fc1.bias.data.cpu().numpy()
fc2_weights = model.fc2.weight.data.cpu().numpy()
fc2_biases = model.fc2.bias.data.cpu().numpy()

# Visualize the weights and biases
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0, 0].hist(fc1_weights.flatten(), bins=100)
axs[0, 0].set_title("FC1 Weights")
axs[0, 1].hist(fc1_biases, bins=100)
axs[0, 1].set_title("FC1 Biases")
axs[1, 0].hist(fc2_weights.flatten(), bins=100)
axs[1, 0].set_title("FC2 Weights")
axs[1, 1].hist(fc2_biases, bins=100)
axs[1, 1].set_title("FC2 Biases")

plt.show()
