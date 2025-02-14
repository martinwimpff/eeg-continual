# Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study

This is the official repository for the paper 
[Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study
](https://arxiv.org/pdf/2502.06828).

## Installation
Run ``pip install .`` to install the ``eeg-continual`` package.

## Usage

Reproducing the results is a 3-step process.

To enhance reproducability, we have included [checkpoints](ckpts) for the first subject.

### 1. Cross-subject source training

Run [source_training.py](src/eeg_continual/source_training.py) ``--config basenet.yaml``
to run the source training.

*Note: You may want to change the [configuration file](configs/basenet.yaml) before to 
customize training (number of subjects, number of epochs, model logging, ..).*

### 2. Subject-specific fine-tuning

Run [finetuning.py](src/eeg_continual/finetuning.py) ``--config finetuning.yaml``
to run the subject-specific fine-tuning.

*Note: You need to have checkpoints in the [ckpt folder](ckpts/source) before starting fine-tuning.*
*Also: check the [configuration file](configs/finetune.yaml)*

### 3. Causal evaluation with OTTA

Run [causal_evaluation.py](src/eeg_continual/causal_evaluation.py) ``--config causal_eval.yaml``
to run the final evaluation.

*Note: You need to have checkpoints in the [ckpt folder](ckpts/finetuning) before starting the evaluation.*
*Also: check the [configuration file](configs/causal_eval.yaml)*

If there are any questions, feel free to [contact me](mailto:martin.wimpff@iss.uni-stuttgart.de).

## Citation
If you find this repository useful, please cite our work:
```
@article{wimpff2025fine,
  title={Fine-Tuning Strategies for Continual Online EEG Motor Imagery Decoding: Insights from a Large-Scale Longitudinal Study},
  author={Wimpff, Martin and Aristimunha, Bruno and Chevallier, Sylvain and Yang, Bin},
  journal={arXiv preprint arXiv:2502.06828},
  year={2025}
}
```

