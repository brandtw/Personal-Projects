# load.py
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import SimpleFeedForward

# Hyperparameters
input_size = 784
num_classes = 10
batch_size = 64

# Dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root="data/", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="data/", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#checks the accuracy of the model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            data = data.reshape(data.shape[0], -1)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# Define the visualize_predictions function
def visualize_predictions(loader, model, num_images=5):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    with torch.no_grad():
        for i, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)
            data = data.reshape(data.shape[0], -1)

            scores = model(data)
            _, predictions = scores.max(1)

            for j, ax in enumerate(axes):
                ax.imshow(data[j].reshape(28, 28).cpu().numpy(), cmap="gray")
                ax.set_title(f"Pred: {predictions[j]}, True: {targets[j]}")
                ax.axis("off")

            break

    model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleFeedForward(input_size, num_classes).to(device)
model.load_state_dict(torch.load("mnist_model.pth"))

train_acc = check_accuracy(train_loader, model)
test_acc = check_accuracy(test_loader, model)
print(f"Train accuracy: {train_acc * 100:.2f}%, Test accuracy: {test_acc * 100:.2f}%")

visualize_predictions(test_loader, model)
plt.show()
