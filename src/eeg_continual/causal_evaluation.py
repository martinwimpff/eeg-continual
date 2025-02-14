from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import yaml

from eeg_continual.datamodules import Stieger21TTADataModule
from eeg_continual.models import BaseNet
from eeg_continual.utils.metrics import calculate_accuracies
from eeg_continual.utils.seed import seed_everything
from eeg_continual.tta import Alignment, Norm

CONFIG_DIR = Path(__file__).resolve().parents[2].joinpath("configs")
CKPT_DIR = Path(__file__).resolve().parents[2].joinpath("ckpts")
DEFAULT_CONFIG = "causal_eval.yaml"


def causal_evaluation(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load source config
    with open(CKPT_DIR.joinpath("source", "config.yaml")) as yaml_file:
        source_config = yaml.safe_load(yaml_file)

    # get preprocessing dict
    preprocessing_dict = source_config.get("preprocessing")
    alignment = preprocessing_dict.get("alignment", False)
    preprocessing_dict["alignment"] = False

    n_subjects = config.get("n_subjects", 1)
    n_classes = 2
    datamodule = Stieger21TTADataModule(preprocessing_dict)
    results_df = pd.DataFrame(columns=[
        "subject_id", "session_id", "test_acc", "test_acc_ww"])
    for subject_id in range(1, n_subjects + 1):
        # setup datamodule for target subject
        datamodule.setup_subject(subject_id)

        n_folds = config.get("n_folds", datamodule.dataset.max_n_folds)
        for fold_idx in range(n_folds):
            # load fientuned model
            ckpt_path = CKPT_DIR.joinpath("finetuning", f"subject_{subject_id:02}-"
                                                f"session_{(fold_idx+1):02}.ckpt")
            finetuned_model = BaseNet.load_from_checkpoint(ckpt_path, map_location=device)


            seed_everything(source_config.get("seed", 0))
            datamodule.setup_fold(fold_idx)

            # put finetuned model in TTA cls
            tta_cls = Norm if config.get("norm") else Alignment
            tta_model = tta_cls(
                deepcopy(finetuned_model.model), config=dict(
                    alignment=alignment))

            # run OTTA
            y_pred = []
            with torch.no_grad():
                for batch in datamodule.predict_dataloader():
                    x, y = batch
                    output = tta_model(x.to(device))
                    if n_classes > 2:
                        y_pred.append(torch.softmax(output, dim=-1))
                    else:
                        y_pred.append(torch.sigmoid(output))
            y_pred = torch.cat(y_pred).detach().cpu()
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
            results_df.loc[len(results_df)] = [
                subject_id, fold_idx + 2, trial_wise_acc, window_wise_acc]

    # save csv
    results_df.to_csv(CKPT_DIR.joinpath("results.csv"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(CONFIG_DIR.joinpath(args.config)) as f:
        config = yaml.safe_load(f)

    causal_evaluation(config)
