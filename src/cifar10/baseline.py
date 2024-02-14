import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import numpy as np
import matplotlib.pyplot as plt

class Baseline_Net(nn.Module):
    def __init__(self):
        super().__init__()
        #特征提取/编码器
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #分类器（特征组合器）
        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 对每个batch做一维展平，以便线性层读取
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, trainloader, device, criterion, optimizer, epoch_losses):
    model.train()
    log_interval = 100
    for epoch in range(10): 

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 每轮前后向传播前优化器保存的梯度清零
            optimizer.zero_grad()

            # 前向传播算loss，反向传播求梯度，优化器优化做参数更新
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % log_interval == log_interval - 1:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_interval:.3f}')
                epoch_losses.append(running_loss / log_interval)
                running_loss = 0.0

    print('Finished Training')

def test(model, testloader, device):

    correct = 0
    total = 0
    model.eval()
    # 关闭torch梯度计算模型，节省显存
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 类别概率里取最高的作为预测标签
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def seed_torch(seed=214):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#如果你用gpu训练的话


def main(model):
    seed_torch(10)

    #特征工程：输入图像先转化为张量矩阵，随后进行减均值除方差的标准化
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    

    criterion = nn.CrossEntropyLoss()#用交叉熵衡量预测概率分布与真实分布之间差异
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)#使用随机梯度下降对参数做更新

    epoch_losses = []
    train(model, trainloader, device, criterion, optimizer, epoch_losses)
    test(model, testloader, device)

    # 绘制loss曲线
    plt.plot(range(len(epoch_losses)), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

if __name__ == '__main__':
    model = Baseline_Net()
    main(model)

#Accuracy of the network on the 10000 test images: 55 %