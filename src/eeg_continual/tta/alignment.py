import numpy as np
from scipy import linalg
import torch
from torch import nn

from .base import TTAMethod


class Alignment(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, reference: np.ndarray = None):
        super(Alignment, self).__init__(model, config)
        self.reference = reference
        self.counter = 0

    @torch.no_grad()
    def forward_and_adapt(self, x):
        x_aligned = self.align_data(x, self.config.get("alignment"))
        outputs = self.model(x_aligned)
        return outputs

    def align_data(self, x, alignment):
        covmat = torch.matmul(x, x.transpose(1, 2)).detach().cpu().numpy()[0]
        self.counter += 1
        if self.reference is not None:
            weights = [1-1/self.counter, 1/self.counter]
            if alignment == "euclidean":
                self.reference = np.average([self.reference, covmat], axis=0, weights=weights)
            else:
                raise NotImplementedError
        else:
            self.reference = covmat
        R_op = linalg.inv(linalg.sqrtm(self.reference))
        x_aligned = torch.matmul(
            torch.tensor(R_op, dtype=torch.float32, device=x.device), x)
        return x_aligned

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
