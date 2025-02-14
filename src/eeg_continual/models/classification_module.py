import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy

from eeg_continual.utils.lr_scheduler import linear_warmup_cosine_decay


class ClassificationModule(pl.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 optimizer: str = "adam",
                 scheduler: bool = False,
                 max_epochs: int = 1000,
                 warmup_epochs: int = 20,
                 window_length: float = 1.0,
                 f_update: int = 25,
                 n_classes: int = 2,
                 **kwargs):
        super(ClassificationModule, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x: torch.tensor, trial_lengths=None):
        return self.model(x, trial_lengths)

    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y, trial_lengths = batch

        # forward pass and prediction
        y_hat = self.forward(x, trial_lengths)  # b x n_windows x n_classes (or 1 if binary)
        if self.hparams.n_classes > 2:
            y_hat = torch.softmax(y_hat, dim=-1)
        else:
            y_hat = torch.sigmoid(y_hat)

        batch_size, n_windows, _ = y_hat.shape

        # compute number of valid windows for each trial
        n_valid_windows = torch.floor((self.hparams.f_update *
                                       (trial_lengths - self.hparams.window_length)) + 1).long()

        # create mask for valid windows
        indices = torch.arange(n_windows, device=x.device).unsqueeze(0).expand(
            batch_size, n_windows)
        mask = indices < n_valid_windows.unsqueeze(1)

        # mask predictions and labels
        y_expanded = y.unsqueeze(-1).repeat(1, n_windows)  # batch_size x n_windows
        y_masked = (y_expanded * mask).reshape(-1)
        y_hat_masked = y_hat * mask.unsqueeze(-1)

        # compute mean predictions for each trial
        y_hat_mean = y_hat_masked.sum(dim=1) / n_valid_windows.unsqueeze(-1)

        # calculate accuracy and loss
        if self.hparams.n_classes > 2:  # multiclass
            acc = accuracy(y_hat_mean, y, task="multiclass",
                           num_classes=self.hparams.n_classes)

            # ensure zero loss for invalid windows by setting the first class to 1
            y_hat_masked[..., 0] = torch.where(~mask, torch.tensor(1.0, device=x.device), y_hat_masked[..., 0])

            loss = F.cross_entropy(y_hat_masked.reshape(-1, self.hparams.n_classes),
                                   y_masked.long().to(x.device))
        else:  # binary
            acc = accuracy(y_hat_mean.squeeze(-1), y, task="binary")
            pred, target = y_hat_masked.reshape(-1), y_masked.float()
            loss = -(target * torch.log(pred + 1e-12) + (1 - target) * torch.log(1 - pred + 1e-12)).mean()

        # log results
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        if self.hparams.n_classes > 2:
            return torch.softmax(self.forward(x), dim=-1)  # b x n_windows x n_classes
        else:
            return torch.sigmoid(self.forward(x))  # b x n_windows x 1


class Finetuner(ClassificationModule):
    def __init__(self,
                 model: torch.nn.Module,
                 lr: float = 0.001,
                 weight_decay: float = 0.0,
                 optimizer: str = "adam",
                 scheduler: bool = False,
                 max_epochs: int = 50,
                 warmup_epochs: int = 0,
                 window_length: float = 1.0,
                 f_update: int = 25,
                 n_classes: int = 2,
                 **kwargs):
        super(Finetuner, self).__init__(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            window_length=window_length,
            f_update=f_update,
            n_classes=n_classes,
            **kwargs)
