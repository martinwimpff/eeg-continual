import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from .stieger_datasets import Stieger21WithinDataset, Stieger21LOSODataset, \
    Stieger21TTADataset
from .window_dataset import WindowDataSet


class Stieger21BaseDataModule(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, preprocessing_dict: dict):
        super(Stieger21BaseDataModule, self).__init__()
        self.batch_size = preprocessing_dict.get("batch_size", NotImplementedError)
        self.preprocessing_dict = preprocessing_dict
        self.dataset = Stieger21WithinDataset(**self.preprocessing_dict)

    def setup_subject(self, subject_id: int):
        self.dataset.setup_subject(subject_id)

    def setup_fold(self, fold_idx: int):
        train_data, test_data = self.dataset.setup_fold(fold_idx)
        self.train_dataset = TensorDataset(*[
            torch.tensor(array) for array in train_data])
        self.test_dataset = TensorDataset(*[
            torch.tensor(array) for array in test_data])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            WindowDataSet(self.test_dataset, self.dataset.window_length,
                          self.dataset.f_update, self.dataset.sfreq),
            batch_size=self.batch_size * 16, shuffle=False)


class Stieger21WithinDataModule(Stieger21BaseDataModule):
    def __init__(self, preprocessing_dict: dict):
        super(Stieger21WithinDataModule, self).__init__(preprocessing_dict)
        self.dataset = Stieger21WithinDataset(**self.preprocessing_dict)


class Stieger21LOSODataModule(Stieger21BaseDataModule):
    def __init__(self, preprocessing_dict: dict):
        super(Stieger21LOSODataModule, self).__init__(preprocessing_dict)
        self.dataset = Stieger21LOSODataset(**self.preprocessing_dict)


class Stieger21TTADataModule(Stieger21BaseDataModule):
    def __init__(self, preprocessing_dict: dict):
        super(Stieger21TTADataModule, self).__init__(preprocessing_dict)
        self.dataset = Stieger21TTADataset(**self.preprocessing_dict)

    def setup_fold(self, fold_idx: int):
        test_data = self.dataset.setup_fold(fold_idx)
        self.test_dataset = TensorDataset(*[
            torch.tensor(array) for array in test_data])

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            WindowDataSet(self.test_dataset, self.dataset.window_length,
                          self.dataset.f_update, self.dataset.sfreq),
            batch_size=1, shuffle=False)
