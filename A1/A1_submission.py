import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from itertools import product


def train_LR(epoch, data_loader, model, optimizer, device, log_interval):
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # cross entropy applies soft max
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )


def eval(data_loader, model, dataset, device):
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += F.cross_entropy(output, target)
    loss /= len(data_loader.dataset)
    print(
        dataset
        + "set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            loss,
            correct,
            len(data_loader.dataset),
            100.0 * correct / len(data_loader.dataset),
        )
    )
    return 100.0 * correct / len(data_loader.dataset)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
        self.fc1 = nn.Linear(10240, 64)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_MNIST_dataloader(batch_size_train, batch_size_test):
    MNIST_training = datasets.MNIST(
        "/MNIST_dataset/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    MNIST_test_set = datasets.MNIST(
        "/MNIST_dataset/",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # Separating the first 48,000 samples for training and the last 12,000 for validation
    MNIST_training_set = Subset(MNIST_training, range(0, 48000))
    MNIST_validation_set = Subset(MNIST_training, range(48000, 60000))

    train_loader = DataLoader(
        MNIST_training_set, batch_size=batch_size_train, shuffle=True
    )

    validation_loader = DataLoader(
        MNIST_validation_set, batch_size=batch_size_train, shuffle=True
    )

    test_loader = DataLoader(MNIST_test_set, batch_size=batch_size_test, shuffle=True)

    return train_loader, validation_loader, test_loader


def logistic_regression(device):

    n_epochs = 10
    batch_size_train = 200
    batch_size_test = 1000
    log_interval = 100
    learning_rate = 0.0015
    weight_decay = 5e-05

    random_seed = 1
    torch.manual_seed(random_seed)

    train_loader, validation_loader, test_loader = get_MNIST_dataloader(
        batch_size_train, batch_size_test
    )

    logistic_model = LogisticRegression().to(device)
    optimizer = optim.SGD(
        logistic_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,  # L2 regularizer added with weight decay
    )

    eval(validation_loader, logistic_model, "Validation", device)
    for epoch in range(1, n_epochs + 1):
        train_LR(
            epoch,
            train_loader,
            logistic_model,
            optimizer,
            device,
            log_interval,
        )
        eval(validation_loader, logistic_model, "Validation", device)

    eval(test_loader, logistic_model, "Test", device)

    results = dict(
        model=logistic_model,
    )

    return results


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        self.fc1 = nn.Linear(32 * 32 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        output = None

        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_loss(self, output, target):
        loss = F.cross_entropy(output, target)

        return loss


def tune_hyper_parameter(target_metric, device):

    from FNN_main import get_dataloaders, train as train_FNN, validation, test

    n_epochs = 6
    batch_size_train = 200
    batch_size_test = 1000
    log_interval = 100

    param_space = {
        "learning rate": [
            0.0075,
            0.0025,
            0.0015,
        ],  # First search:  log - 0.0050 fnn - 0.0001
        # Second search: log - 0.0015 fnn - 0.0015
        "weight decay": [
            0.00010,
            0.00005,
            0.00001,
        ],  # First search:  log - 0.00010 fnn - 0.00001
        # Second search: log - 0.00005 fnn - 0.00010
    }

    param_combinations = list(product(*param_space.values()))

    best_params = [
        {
            "logistic_regression": {
                "learning_rate": None,
                "weight_decay": None,
            }
        },
        {
            "FNN": {
                "learning_rate": None,
                "weight_decay": None,
            }
        },
    ]
    best_metric = [
        {"logistic_regression": {"accuracy": None}},
        {"FNN": {"accuracy": None}},
    ]

    # GRID SEARCH
    best_accuracy_lr = 0
    best_accuracy_fnn = 0
    for params in param_combinations:

        # PART I

        train_loader, validation_loader, test_loader = get_MNIST_dataloader(
            batch_size_train, batch_size_test
        )

        learning_rate, weight_decay = params

        print(
            "\nLR - LEARNING RATE {} AND WEIGHT DECAY {}".format(
                learning_rate,
                weight_decay,
            )
        )

        logistic_model = LogisticRegression().to(device)

        # Define optimizer with current parameters
        optimizer = optim.Adam(
            logistic_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Train the model
        for epoch in range(1, n_epochs + 1):
            train_LR(
                epoch, train_loader, logistic_model, optimizer, device, log_interval
            )
            eval(validation_loader, logistic_model, "Validation", device)

        # Evaluate the model
        accuracy_lr = eval(test_loader, logistic_model, "Test", device)

        print(
            "\nLR - ACCURACY {}, LEARNING RATE {} AND WEIGHT DECAY {}".format(
                accuracy_lr,
                learning_rate,
                weight_decay,
            )
        )
        # Store the best model and hyperparameters
        if accuracy_lr > best_accuracy_lr:
            print(
                "\nBEST ACCURACY UPDATED FOR LR OLD = {}, NEW = {} ".format(
                    best_accuracy_lr,
                    accuracy_lr,
                )
            )
            best_accuracy_lr = accuracy_lr
            best_params[0]["logistic_regression"]["learning_rate"] = learning_rate
            best_params[0]["logistic_regression"]["weight_decay"] = weight_decay
            best_metric[0]["logistic_regression"]["accuracy"] = best_accuracy_lr

        # PART II
        class BatchSizeTune:
            train = 128
            val = 128
            test = 1000

        batch_size_fnn = BatchSizeTune()

        train_loader, validation_loader, test_loader = get_dataloaders(batch_size_fnn)

        print(
            "\nFNN - LEARNING RATE {} AND WEIGHT DECAY {}".format(
                learning_rate,
                weight_decay,
            )
        )

        # Instantiate model
        fnn_model = FNN("ce", 10).to(device)

        # Define optimizer with current parameters
        optimizer = optim.Adam(
            fnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Train the model
        with torch.no_grad():
            validation(fnn_model, validation_loader, device)
        for epoch in range(n_epochs):
            print(f"\nepoch {epoch + 1} / {n_epochs}\n")

            train_FNN(fnn_model, optimizer, train_loader, device)

            with torch.no_grad():
                validation(fnn_model, validation_loader, device)

        # Test the model
        with torch.no_grad():
            accuracy_fnn = test(fnn_model, test_loader, device)

        print(
            "\nFNN - ACCURACY {}, LEARNING RATE {} AND WEIGHT DECAY {}".format(
                accuracy_fnn,
                learning_rate,
                weight_decay,
            )
        )
        # Store the best model and hyperparameters
        if accuracy_fnn > best_accuracy_fnn:
            print(
                "\nBEST ACCURACY UPDATED FOR FNN OLD = {}, NEW = {} ".format(
                    best_accuracy_fnn,
                    accuracy_fnn,
                )
            )
            best_accuracy_fnn = accuracy_fnn
            best_params[1]["FNN"]["learning_rate"] = learning_rate
            best_params[1]["FNN"]["weight_decay"] = weight_decay
            best_metric[1]["FNN"]["accuracy"] = best_accuracy_fnn

    return best_params, best_metric
