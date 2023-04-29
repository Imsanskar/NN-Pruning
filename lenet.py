import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
total_epoch = 0

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc_1 = nn.Linear(400, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor):
        y = self.pool_1(F.relu(self.conv_1(x)))
        y = self.pool_2(F.relu(self.conv_2(y)))
        y = torch.flatten(y, 1)
        y = F.relu(self.fc_1(y))
        y = F.relu(self.fc_2(y))
        y = self.fc_3(y)
        
        return y

def train(model, epochs, train_dataloader, optimizer, loss_fn = None):
    # initially train the model with all the parameters
    total_data = len(train_dataloader)
    if loss_fn == None:
        loss_fn = F.cross_entropy
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_dataloader):
            pred = model(data.to(device))
            loss = loss_fn(pred, label.to(device))

            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f"Epoch: {epoch} Loss {i} / {total_data}: {torch.linalg.norm(loss)}")

            optimizer.zero_grad()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((32, 32))
    ])

    # dataset
    train_dataset = datasets.MNIST("./data", train = True, download=True, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    testloader = DataLoader(test_dataset, shuffle=True)

    PATH = "models/letnet.pt"
    checkpoint = torch.load(PATH)

    model = LeNet(10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer
    loss_fn = torch.nn.CrossEntropyLoss() # Cross Entropy Loss function

    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    total_epoch = checkpoint['epoch']

    model.to(device = device)
    
    epoch = 10
    train(model, epoch, trainloader, optimizer, loss_fn)

    torch.save({
            'epoch': total_epoch + epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),            
            }, PATH)