from abc import ABC

import numpy as np

from eeg_continual.utils.alignment import align
from eeg_continual.utils.load_data import get_n_sessions, load_data


class StiegerBaseDataset(ABC):
    all_subject_ids = list(range(1, 62))
    subject_id: int = None
    data_dict = dict()
    info = None
    max_trial_length = 6.04
    sfreq = 250
    f_update = 25
    window_length = 1
    max_n_folds: int = None
    n_sessions: int = None

    def __init__(self, min_trial_length: float = 1.0, l_freq: float = None,
                 h_freq: float = None, paradigm: str = "LR", data_mode: str = "first",
                 alignment: str = False):
        self.min_trial_length = min_trial_length
        self.l_freq, self.h_freq = l_freq, h_freq
        self.max_windows_per_trial = int(self.f_update * (
                self.max_trial_length - self.window_length)) + 1
        assert data_mode in ["first", "exemplar-free", "joint"]
        self.data_mode = data_mode
        assert paradigm in ["LR", "UD"]
        self.paradigm = paradigm
        self.alignment = alignment

    def setup_subject(self, subject_id: int):
        self.subject_id = subject_id

    def setup_fold(self, fold_idx: int = 0):
        raise NotImplementedError

    def _clean_data_dict(self):
        # delete old data from data dict to save memory
        if f"subject_{self.subject_id}" in self.data_dict.keys():
            keys = list(self.data_dict[f"subject_{self.subject_id}"].keys())
            for key in keys:
                del self.data_dict[f"subject_{self.subject_id}"][key]

    def _get_data(self, subject_ids: list[int], session_ids: list[int]):
        x, targets, lengths = [], [], []
        for subject_id in subject_ids:
            for session_id in session_ids:
                df = self.data_dict[f"subject_{subject_id}"][f"session_{session_id}"][3]
                indices = df.index.to_numpy()
                eeg = self.data_dict[f"subject_{subject_id}"][f"session_{session_id}"][
                    0].get_data(item=indices, tmin=0.0, tmax=self.max_trial_length)
                trial_lengths = self.data_dict[f"subject_{subject_id}"][
                    f"session_{session_id}"][2][indices]

                # align on a session level
                if self.alignment:
                    eeg = align(eeg, trial_lengths, sfreq=self.sfreq,
                                f_update=self.f_update, window_length=self.window_length)

                x.append(eeg)
                targets.append(self.data_dict[f"subject_{subject_id}"][
                                   f"session_{session_id}"][1][indices])
                lengths.append(trial_lengths)

        x = np.concatenate(x, axis=0, dtype="float32")
        targets = np.concatenate(targets, axis=0, dtype="int32")
        lengths = np.concatenate(lengths, axis=0, dtype="float32")

        return x, targets, lengths


class Stieger21WithinDataset(StiegerBaseDataset):

    def __init__(self, min_trial_length: float = 1.0, l_freq: float = None,
                 h_freq: float = None, paradigm: str = "LR", data_mode: str = "joint",
                 alignment: str = False, **kwargs):
        super(Stieger21WithinDataset, self).__init__(
            min_trial_length=min_trial_length, l_freq=l_freq, h_freq=h_freq,
            paradigm=paradigm, data_mode=data_mode, alignment=alignment)

        assert self.data_mode in ["exemplar-free", "joint"]

    def setup_subject(self, subject_id: int):
        self._clean_data_dict()  # delete all sessions of prev. subject
        self.subject_id = subject_id
        self.n_sessions = get_n_sessions(self.subject_id)
        self.max_n_folds = self.n_sessions - 1

        self.data_dict.update({f"subject_{self.subject_id}": {f"session_{i}": load_data(
            self.subject_id, i, self.min_trial_length, self.l_freq, self.h_freq,
            self.paradigm) for i in range(1, self.n_sessions + 1)}})
        if self.info is None:
            self.info = self.data_dict[f"subject_{self.subject_id}"]["session_1"][0].info

    def setup_fold(self, fold_idx: int = 0):
        assert fold_idx in range(self.max_n_folds)
        if self.data_mode == "exemplar-free":
            train_session_ids = [fold_idx + 1]
        elif self.data_mode == "joint":
            train_session_ids = list(range(1, fold_idx + 2))
        else:
            raise NotImplementedError
        test_session_ids = [fold_idx + 2]

        train_data = self._get_data([self.subject_id], train_session_ids)
        test_data = self._get_data([self.subject_id], test_session_ids)

        return train_data, test_data


class Stieger21LOSODataset(StiegerBaseDataset):
    def __init__(self, min_trial_length: float = 1.0, l_freq: float = None,
                 h_freq: float = None, paradigm: str = "LR", data_mode: str = "first",
                 alignment: str = False, **kwargs):
        super(Stieger21LOSODataset, self).__init__(
            min_trial_length=min_trial_length, l_freq=l_freq, h_freq=h_freq,
            paradigm=paradigm, data_mode=data_mode, alignment=alignment)

        assert self.data_mode == "first"

        # load first session of all subjects
        self.data_dict.update(
            {f"subject_{subject_id}": {"session_1": load_data(
                subject_id, 1, self.min_trial_length, self.l_freq, self.h_freq,
                self.paradigm)} for subject_id in self.all_subject_ids})
        self.info = self.data_dict["subject_1"]["session_1"][0].info

    def setup_subject(self, subject_id: int):
        super(Stieger21LOSODataset, self).setup_subject(subject_id)
        self.max_n_folds = 1

    def setup_fold(self, fold_idx: int = 0):
        assert fold_idx in range(self.max_n_folds)
        train_subject_ids = list(set(self.all_subject_ids) - {self.subject_id})
        train_data = self._get_data(train_subject_ids, [1])
        test_data = self._get_data([self.subject_id], [1])

        return train_data, test_data


class Stieger21TTADataset(StiegerBaseDataset):
    def __init__(self, min_trial_length: float = 1.0, l_freq: float = None,
                 h_freq: float = None, paradigm: str = "LR", data_mode: str = None,
                 alignment: str = False, **kwargs):
        assert not alignment
        super(Stieger21TTADataset, self).__init__(
            min_trial_length=min_trial_length, l_freq=l_freq, h_freq=h_freq,
            paradigm=paradigm, data_mode=data_mode, alignment=alignment)

    def setup_subject(self, subject_id: int):
        Stieger21WithinDataset.setup_subject(self, subject_id)

    def setup_fold(self, fold_idx: int = 0):
        assert fold_idx in range(self.max_n_folds)
        test_data = self._get_data([self.subject_id], [fold_idx + 2])

        return test_data
