import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

def get_data_loaders(batch_size, data_dir):
    train_dataset = MNIST(data_dir, download=True, train=True, transform=transforms.ToTensor())
    test_dataset = MNIST(data_dir, download=True, train=False, transform=transforms.ToTensor())

    train_ds, val_ds = random_split(train_dataset, [50000,10000])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader 

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_1=256, hidden_size_2=128):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, xb):
        xb = self.flatten(xb)
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        return self.fc3(xb)

def train_and_validate(model, train_loader, val_loader, 
                       optimizer, criterion, device, epochs):
    
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward pass
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        avg_train = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train)
        train_accs.append(correct/len(train_loader.dataset))
        

        model.eval()
        running_loss, correct = 0.0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds==labels).sum().item()

        avg_val = running_loss / len(val_loader.dataset)
        acc_val = correct / len(val_loader.dataset)
        val_losses.append(avg_val)
        val_accs.append(acc_val)


        print(f"Epoch {epoch}/{epochs} "
              f"Train Loss: {avg_train:.4f} "
              f"Val Loss: {avg_val:.4f} "
              f"Val Acc: {acc_val:.4f} ")

    return train_losses, train_accs, val_losses, val_accs

def test(model, test_loader, device):

    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST and save the model.")
    parser.add_argument('--epochs', type=int, default=5,  help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--save-path',  type=str,   default='../../saved_models/mnist_model_fp32.pth', help='Path to save the trained model')
    parser.add_argument('--data-dir',   type=str,   default='../../data',     help='Directory for MNIST data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_data_loaders(batch_size = args.batch_size, data_dir = args.data_dir)
    model = MLP(input_size=28*28, output_size=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, args.epochs)
    test(model, test_loader, device)

    # Save state_dict
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    main()