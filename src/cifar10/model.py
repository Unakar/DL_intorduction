import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from config import Config

class Baseline_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BnModel(nn.Module): # Add Batch Normalization 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class KaimingInitModel(nn.Module): # Using Kaiming Initialization 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc1.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc3.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DoubleChannelModel(nn.Module): # Doubling the Channel
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class BetterBaselineModel(nn.Module): # Better Baseline Model 
    def __init__(self):
        super().__init__()
        channel1 = 256
        channel2 = 256
        self.conv1 = nn.Conv2d(3, channel1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(channel1)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(channel1, channel2, 5)
        self.bn2 = nn.BatchNorm2d(channel2)
        self.dropout2 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(channel2 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.dropout3 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ResNet(nn.Module): # Resnet18
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False)#这里改成true会自动下载预训练权重
        if Config.PRETRAINED:
            self.model.load_state_dict(torch.load(Config.RESNET_PRETRAINED_PATH))
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(512, 10)
        self.model.fc.requires_grad_(True)

    def forward(self, x):
        x = self.model(x)
        return x