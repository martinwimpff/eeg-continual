import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

DATA_PATH = Path(__file__).resolve().parents[3].joinpath("data")


def get_n_sessions(subject_id: int) -> int:
    all_files = os.listdir(DATA_PATH)
    return len([file for file in all_files if file.startswith(f"S{subject_id}_")])


def load_data(subject_id: int, session_id: int, min_trial_length: float = 1.0,
              l_freq: float = 8.0, h_freq: float = 30.0, paradigm: str = "LR"):

    subject_path = DATA_PATH.joinpath(f"S{subject_id}_Session_{session_id}.mat")
    mat = loadmat(subject_path)

    trial_data = mat["BCI"]["TrialData"][0, 0]  # 1 x 450
    rows = []
    for trial in trial_data[0]:
        row = [arr.item() for arr in trial]
        rows.append(row)
    df = pd.DataFrame(rows, columns=["tasknumber", "runnumber", "trialnumber",
                                     "targetnumber", "triallength", "targethitnumber",
                                     "resultind", "result", "forcedresult", "artifact"])

    if paradigm == "LR":
        target_numbers = [1, 2]
    elif paradigm == "UD":
        target_numbers = [3, 4]
    else:
        raise NotImplementedError

    # filter by tasknumber
    assert paradigm in ["LR", "UD"]
    tasknumber = 1 if paradigm == "LR" else 2
    df = df[df["tasknumber"] == tasknumber]

    # filter data (min trial length and artifact rejection)
    filtered_df = df[
        (df['artifact'] == 0) &
        (df['triallength'] > min_trial_length)
        ]

    indices = filtered_df.index.to_numpy()
    labels = filtered_df["targetnumber"].values - np.min(target_numbers)
    triallengths = filtered_df["triallength"].values
    filtered_df.reset_index(inplace=True)
    filtered_df.rename(columns={"index": "trial_idx"}, inplace=True)

    # pad trial
    data = mat["BCI"]["data"][0][0]
    padded_data = np.zeros((len(indices), 62, 11041), dtype=np.float32)
    for i, trial in enumerate(data[0, indices]):
        length = trial.shape[-1]
        padded_data[i, :, :length] = trial

    # get channel info
    chaninfo = mat["BCI"]["chaninfo"][0][0].item()
    ch_names = chaninfo[0][0]
    ch_names = [ch.item() for ch in ch_names]

    # create mne object
    info = mne.create_info(ch_names, sfreq=1000.0, ch_types=["eeg"] * 62)
    info.rename_channels({
        "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2", "FZ": "Fz", "FCZ": "FCz",
        "CZ": "Cz", "CPZ": "CPz", "PZ": "Pz", "POZ": "POz", "CB1": "PO9", "OZ": "Oz",
        "CB2": "PO10"
    })
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    epochs = mne.EpochsArray(padded_data, info, tmin=-4.0)

    # channel selection
    epochs.pick_channels([
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "TP8",
    ])

    # filter and resample
    epochs.filter(l_freq=l_freq, h_freq=h_freq)
    epochs.resample(250.0)

    return epochs, labels, triallengths, filtered_df
