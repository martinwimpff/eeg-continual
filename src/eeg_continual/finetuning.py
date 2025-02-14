from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
import torch
import yaml

from eeg_continual.datamodules import Stieger21WithinDataModule
from eeg_continual.models import BaseNet, Finetuner
from eeg_continual.utils.metrics import calculate_accuracies
from eeg_continual.utils.print_results import print_results
from eeg_continual.utils.seed import seed_everything

CONFIG_DIR = Path(__file__).resolve().parents[1].joinpath("configs")
CKPT_DIR = Path(__file__).resolve().parents[2].joinpath("ckpts")
DEFAULT_CONFIG = "finetune.yaml"


def finetune(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load source config
    with open(CKPT_DIR.joinpath("source", "config.yaml")) as yaml_file:
        source_config = yaml.safe_load(yaml_file)

    # get preprocessing dict
    preprocessing_dict = source_config.get("preprocessing")

    # set data_mode -> which data setting is used for finetuning
    data_mode = config.get("data_mode", "joint")
    preprocessing_dict["data_mode"] = data_mode

    # get classes and run name
    model_cls = BaseNet
    n_subjects = config.get("n_subjects", 1)
    n_classes = 2
    datamodule = Stieger21WithinDataModule(preprocessing_dict)

    results_df = pd.DataFrame(columns=[
        "subject_id", "session_id", "test_acc", "test_acc_ww"])
    for subject_id in range(1, n_subjects + 1):
        datamodule.setup_subject(subject_id)

        # load source model
        ckpt_path = CKPT_DIR.joinpath("source", f"subject_{subject_id:02}.ckpt")
        source_model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)

        n_folds = config.get("n_folds", datamodule.dataset.max_n_folds)
        for fold_idx in range(n_folds):
            seed_everything(source_config.get("seed", 0))
            datamodule.setup_fold(fold_idx)

            trainer = Trainer(
                max_epochs=config.get("max_epochs"),
                num_sanity_val_steps=0,
                accelerator="auto",
                strategy="auto",
                enable_checkpointing=False
            )

            if config.get("adaptation_mode", "independent") == "independent" or fold_idx == 0:
                # restart from source
                model = Finetuner(deepcopy(source_model.model),
                                  max_epochs=config.get("max_epochs"),
                                  n_classes=n_classes,
                                  **config.get("finetuner_kwargs"))
                trainer.fit(model, datamodule=datamodule)
            else:
                # continue from previous checkpoint
                trainer.fit(model, datamodule=datamodule)

            # save checkpoint
            if config.get("log_model", False):
                trainer.save_checkpoint(
                    CKPT_DIR.joinpath("finetuning", f"subject_{subject_id:02}-"
                                                    f"session_{(fold_idx+1):02}.ckpt"))

            # get predictions and true labels
            y_pred = torch.cat(
                trainer.predict(model, datamodule.predict_dataloader()))
            y_test = datamodule.test_dataset.tensors[1]

            # get valid window information
            n_valid_windows_per_trial = datamodule.predict_dataloader().dataset.n_valid_windows_per_trial
            valid_windows_cumsum = datamodule.predict_dataloader().dataset.valid_windows_cumsum

            trial_wise_acc, window_wise_acc = calculate_accuracies(
                y_pred, y_test, n_valid_windows_per_trial, valid_windows_cumsum,
                n_classes, datamodule.dataset.max_windows_per_trial
            )

            # write results to dataframe
            results_df.loc[len(results_df)] = [subject_id, fold_idx + 2,
                                               trial_wise_acc, window_wise_acc]

    print_results(results_df)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(CONFIG_DIR.joinpath(args.config)) as f:
        config = yaml.safe_load(f)

    finetune(config)
