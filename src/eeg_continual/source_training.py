from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
import torch
import yaml

from eeg_continual.datamodules import Stieger21LOSODataModule
from eeg_continual.models import BaseNet
from eeg_continual.utils.metrics import calculate_accuracies
from eeg_continual.utils.print_results import print_results
from eeg_continual.utils.seed import seed_everything

CONFIG_DIR = Path(__file__).resolve().parents[2].joinpath("configs")
CKPT_DIR = Path(__file__).resolve().parents[2].joinpath("ckpts")
DEFAULT_CONFIG = "basenet.yaml"


def train_and_test(config: dict):
    model_cls = BaseNet
    datamodule_cls = Stieger21LOSODataModule
    n_subjects = config.get("n_subjects", 1)
    n_classes = 2

    results_df = pd.DataFrame(columns=[
        "subject_id", "session_id", "test_acc", "test_acc_ww"])

    datamodule = datamodule_cls(config.get("preprocessing"))
    for subject_id in range(1, n_subjects + 1):
        seed_everything(config.get("seed", 0))
        datamodule.setup_subject(subject_id)
        datamodule.setup_fold(0)

        trainer = Trainer(
            max_epochs=config.get("max_epochs"),
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy="auto",
            enable_checkpointing=False
        )

        # train model
        model = model_cls(
            **config.get("model_kwargs"),
            max_epochs=config.get("max_epochs"), n_classes=n_classes)
        trainer.fit(model, datamodule=datamodule)

        # save checkoint and config
        if config.get("log_model", False):
            trainer.save_checkpoint(CKPT_DIR.joinpath("source", f"subject_{subject_id:02}.ckpt"))
            if subject_id == 1:
                with open(CKPT_DIR.joinpath("source", "config.yaml"), 'w') as yaml_file:
                    yaml.dump(config, yaml_file, default_flow_style=False)


        # get predictions and true labels
        y_pred = torch.cat(trainer.predict(model, datamodule.predict_dataloader()))
        y_test = datamodule.test_dataset.tensors[1]

        # get valid window information
        n_valid_windows_per_trial = datamodule.predict_dataloader().dataset.n_valid_windows_per_trial
        valid_windows_cumsum = datamodule.predict_dataloader().dataset.valid_windows_cumsum

        # calculate metrics
        trial_wise_acc, window_wise_acc = calculate_accuracies(
            y_pred, y_test, n_valid_windows_per_trial, valid_windows_cumsum,
            n_classes, datamodule.dataset.max_windows_per_trial
        )

        # write results to dataframe
        results_df.loc[len(results_df)] = [subject_id, 1, trial_wise_acc, window_wise_acc]

    print_results(results_df)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(CONFIG_DIR.joinpath(args.config)) as f:
        config = yaml.safe_load(f)

    train_and_test(config)
