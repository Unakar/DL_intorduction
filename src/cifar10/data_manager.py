import torch
import torchvision
import torchvision.transforms as transforms
from config import Config
from augmentation import AutoAugment

class Dataset_Manager:
    def __init__(self) -> None:
        self.transform = self.get_transform()
        self.transform_test = self.get_transform_test()

        trainset = torchvision.datasets.CIFAR10(root=Config.DATASET_PATH, train=True,
                                                download=False, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=Config.BATCH_SIZE,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=Config.DATASET_PATH, train=False,
                                            download=False, transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=Config.BATCH_SIZE,
                                                shuffle=False, num_workers=2)
    def get_transform(self):
        res = []
        res.append(transforms.RandomHorizontalFlip(p=0.5))
        res.extend([transforms.Pad(2, padding_mode='constant'),
                        transforms.RandomCrop([32,32])])
        res.append(transforms.RandomApply([AutoAugment()], p=0.6))
        res.append(transforms.ToTensor())
        res += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(res)
    
    def get_transform_test(self):
        res = [transforms.ToTensor()]
        res += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(res)

    def get_trainloader(self):
        return self.trainloader
    
    def get_testloader(self):
        return self.testloader


