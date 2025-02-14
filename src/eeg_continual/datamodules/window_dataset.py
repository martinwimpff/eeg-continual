import torch
from torch.utils.data import Dataset, TensorDataset


class WindowDataSet(Dataset):
    """
    This is a wrapper class for TensorDatasets.
    It ensures easy window loading while remaining memory-efficient
    """
    def __init__(self, dataset: TensorDataset, window_length: float = 1.0,
                 f_update: int = 25, sfreq: int = 250):
        self.dataset = dataset
        self.window_length = window_length
        self.f_update = f_update
        self.sfreq = sfreq
        self.window_shift_samples = int(self.sfreq / self.f_update)
        self.window_length_samples = int(self.window_length * self.sfreq)
        self.n_valid_windows_per_trial = self._get_n_windows_per_trial()
        self.valid_windows_cumsum = torch.cumsum(self.n_valid_windows_per_trial, dim=0)

    def __len__(self):
        return torch.sum(self.n_valid_windows_per_trial).item()

    def _get_n_windows_per_trial(self):
        trial_lengths = self.dataset.tensors[-1]
        n_valid_windows = torch.floor(
            (self.f_update * (trial_lengths - self.window_length)) + 1).int()
        return n_valid_windows

    def __getitem__(self, idx):
        # find trial
        if idx < self.valid_windows_cumsum[0]:
            trial_idx = 0
        else:
            trial_idx = torch.where(idx >= self.valid_windows_cumsum)[0][-1].item() + 1

        # find window idx
        if trial_idx > 0:
            window_idx = idx - self.valid_windows_cumsum[trial_idx - 1].item()
        else:
            window_idx = idx

        window = self.dataset.tensors[0][
                 trial_idx, :, window_idx * self.window_shift_samples:
                               window_idx * self.window_shift_samples +
                               self.window_length_samples]

        return window, self.dataset.tensors[1][trial_idx]
