import numpy as np
from scipy import linalg


def align(x: np.ndarray, trial_lengths: np.ndarray, sfreq: int, f_update: int,
          window_length: int):
    window_shift_samples = int(sfreq / f_update)
    window_length_samples = int(window_length * sfreq)
    windows = []
    for trial_idx, trial in enumerate(x):
        n_valid_windows = np.floor(
            f_update * (trial_lengths[trial_idx] - window_length) + 1).astype("int32")
        for i in range(n_valid_windows):
            windows.append(trial[:, i * window_shift_samples:i * window_shift_samples + window_length_samples])
    windows = np.array(windows)
    window_covmats = np.matmul(windows, windows.transpose(0, 2, 1))
    R = window_covmats.mean(0)
    R_op = linalg.inv(linalg.sqrtm(R))
    x = np.matmul(R_op, x)
    return x
