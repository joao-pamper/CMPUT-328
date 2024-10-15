import timeit
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets, models
import numpy as np
import random


# Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(100)


def get_config_dict(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need different configs for part 1 and 2.
    """

    config = {
        "batch_size": 64,
        "lr": 0.0015,
        "num_epochs": 5,
        "weight_decay": 0.0001,  # set to 0 if you do not want L2 regularization
        "save_criteria": "accuracy",  # Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)
    }

    return config


# Part 1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input is 32X32 images
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        # output is 16 32x32
        self.pool1 = nn.MaxPool2d(2, 2)
        # output is 16 16x16
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        # output is 32 16x16
        self.pool2 = nn.MaxPool2d(2, 2)
        # output is 32 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Part 2
class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()

        self.model = models.efficientnet_b0(pretrained=True)

        print("Model summary:", self.model)

        # Replace the final layer to match CIFAR-10's 10 classes
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, 10)

    def forward(self, x):
        x = self.model(x)
        return x


def load_dataset(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need to define different dataset splits/transformations/augmentations for part 2.

    returns:
    train_dataset, valid_dataset: Dataset for training your model
    test_transforms: Default is None. Edit if you would like transformations applied to the test set.

    """
    test_transforms = None

    if pretrain == 0:
        full_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif pretrain == 1:
        full_dataset = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        test_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    return train_dataset, valid_dataset, test_transforms
