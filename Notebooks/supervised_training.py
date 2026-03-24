import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

os.makedirs("data", exist_ok=True)
os.makedirs("supervised_checkpoints", exist_ok=True)



class CIFAR10Subset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base[real_idx]


def get_cifar10_loaders(selected_indices, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        ),
    ])

    train_base = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test
    )

    train_subset = CIFAR10Subset(train_base, selected_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def build_resnet18():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_supervised(
        selected_indices,
        epochs=500,
        batch_size=128,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_loaders(selected_indices, batch_size=batch_size)

    model = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_curve = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_curve.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} ::: Loss {epoch_loss:.4f}")

        # Save checkpoint (relative path)
        torch.save(model.state_dict(), f"supervised_checkpoints/epoch_{epoch+1}.pth")

    runtime = time.time() - start_time

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Final Test Accuracy: {accuracy:.3f}")
    print(f"Training time elapsed: {runtime:.2f} seconds")

    return accuracy, loss_curve, runtime



if __name__ == "__main__":
    selected_indices = list(range(50000))  # full CIFAR-10 training set
    train_supervised(selected_indices=selected_indices, epochs=500)
