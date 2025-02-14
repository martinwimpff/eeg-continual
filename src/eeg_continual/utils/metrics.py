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

    # accuracy per window
    nan_tensor = torch.full((n_trials, max_windows_per_trial), float("nan"))

    for i in range(n_trials):
        start_idx = valid_windows_cumsum[i - 1] if i > 0 else 0
        end_idx = valid_windows_cumsum[i]
        preds = y_pred[start_idx:end_idx, 0]

        if n_classes > 2:
            preds = preds.argmax(-1)  # multiclass
        else:
            preds = torch.round(preds.squeeze())  # binary

        # write predictions to nan_tensor
        nan_tensor[i, :n_valid_windows_per_trial[i]] = preds

    # mask invalid windows and calculate accuracy per window
    nan_mask = torch.isnan(nan_tensor)
    comparison_result = (
                nan_tensor == y_test.unsqueeze(-1).expand(-1, max_windows_per_trial)).float()
    comparison_result.masked_fill_(nan_mask, float("nan"))

    # calculate mean accuracy per window over all trials
    per_window = torch.nanmean(comparison_result, dim=0)

    return trial_wise_acc, window_wise_acc, per_window
