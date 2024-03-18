import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


# DataLoader
class CustomMINST(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def _getitem_(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target


class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx >= len(self):
            raise StopIteration

        batch_indices = self.indices[self.iter_idx:self.iter_idx +self.batch_size]
        data = [self.dataset[idx] for idx in batch_indices]
        data = zip(*data)
        inputs, targets = [torch.stack(tensors) for tensors in data]
        self.iter_idx += self.batch_size
        return inputs, targets

    def __len__(self):
        return len(self.dataset)


# 神经网络 两个全连接层
class SimpleNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.rula(self.fc1(x))
        x = self.fc2(x)
        return x


# 设置超参数，优化器 损失函数
input_size = 784  # 28*28
hidden_size = 500
num_classes = 10
learning_rate = 0.001
batch_size = 30
epochs = 5

model = SimpleNeuralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 训练网络并测试
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)\tLoss:{:.6f]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # 加 batch loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('/nTest set:Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# 用cpu训练了
device = torch.device("cpu")
model = model.to(device)

# train_dataset test_dataset
train_dataset = CustomMINST(r"C:\Users\86182\Desktop\MNIST_Dataset\MNIST_Dataset\train_images")
test_dataset = CustomMINST(r"C:\Users\86182\Desktop\MNIST_Dataset\MNIST_Dataset\train_images")

train_loader = CustomDataLoader(train_dataset, batch_size, shuffle=True)
test_loader = CustomDataLoader(test_dataset, batch_size, shuffle=False)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)




