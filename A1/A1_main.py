import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

from A1_submission import logistic_regression, LogisticRegression

torch.multiprocessing.set_sharing_strategy("file_system")


def compute_score(acc, acc_thresh):
    min_thres, max_thres = acc_thresh
    if acc <= min_thres:
        score = 0.0
    elif acc >= max_thres:
        score = 100.0
    else:
        score = float(acc - min_thres) / (max_thres - min_thres) * 100
    return score


def test(
    model,
    device,
):
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    num_correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            output = model(data)
            predicted = torch.argmax(output, dim=1)
            total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

    acc = float(num_correct) / total
    return acc


class Args:
    """
    command-line arguments
    """

    """
    'logistic': run logistic regression on the specified dataset (part 1)
    'tune': run hyper parameter tuning (part 3)
    """
    mode = "logistic"

    """
    metric with respect to which hyper parameters are to be tuned
    'acc': validation classification accuracy
    'loss': validation loss
    """
    target_metric = "acc"
    # target_metric = 'loss'

    """
    set to 0 to run on cpu
    """
    gpu = 1


def main():
    args = Args()
    try:
        import paramparse

        paramparse.process(args)
    except ImportError:
        pass

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    acc_thresh = dict(
        logistic=(0.83, 0.93),
    )

    if args.mode == "logistic":
        start = timeit.default_timer()
        results = logistic_regression(device)
        model = results["model"]

        if model is None:
            print("model is None")
            return

        stop = timeit.default_timer()
        run_time = stop - start

        accuracy = test(
            model,
            device,
        )

        score = compute_score(accuracy, acc_thresh[args.mode])
        result = OrderedDict(accuracy=accuracy, score=score, run_time=run_time)
        print(f"result on {args.mode}:")
        for key in result:
            print(f"\t{key}: {result[key]}")


if __name__ == "__main__":
    main()
