import torch
from torchmetrics.functional import accuracy


def calculate_accuracies(y_pred: torch.tensor, y_test: torch.tensor,
                         n_valid_windows_per_trial: torch.tensor,
                         valid_windows_cumsum: torch.tensor, n_classes: int,
                         max_windows_per_trial: int):
    n_trials = len(n_valid_windows_per_trial)

    # repeat labels according to the number of valid windows
    y_test_repeated = torch.repeat_interleave(y_test, n_valid_windows_per_trial)

    # calculate mean prediction per trial
    y_pred_mean = [
        y_pred[valid_windows_cumsum[i - 1] if i > 0 else 0:
               valid_windows_cumsum[i]].mean(0)[0]
        for i in range(n_trials)
    ]

    # calculate window-wise and trial-wise accuracy
    if n_classes > 2:
        window_wise_acc = accuracy(
            y_pred.squeeze(-2), y_test_repeated, task="multiclass",
            num_classes=n_classes)
        trial_wise_acc = accuracy(
            torch.stack(y_pred_mean), y_test, task="multiclass",
            num_classes=n_classes)
    else:
        window_wise_acc = accuracy(y_pred.squeeze(), y_test_repeated,
                                   task="binary")
        trial_wise_acc = accuracy(torch.stack(y_pred_mean).squeeze(),
                                  y_test, task="binary")

    return trial_wise_acc.item(), window_wise_acc.item()
