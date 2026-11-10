"""Early stopping with scArches scPoli semantics (``scarches.utils.monitor.EarlyStopping``)."""

from __future__ import annotations

import copy
import logging

import numpy as np
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

# scArches metric names -> names logged by scpolivi
SCARCHES_METRIC_NAMES = {
    "val_loss": "validation_loss",
    "val_unweighted_loss": "unweighted_loss_validation",
    "val_cvae_loss": "cvae_loss_validation",
    "val_prototype_loss": "prototype_loss_validation",
    "val_labeled_loss": "labeled_loss_validation",
    "val_unlabeled_loss": "unlabeled_loss_validation",
    "epoch_loss": "train_loss",
}


def translate_metric(name: str) -> str:
    return SCARCHES_METRIC_NAMES.get(name, name)


class SCPoliEarlyStopping(Callback):
    """Patience-based stopping plus learning-rate reduction on the same metric.

    Inactive during the first ``pretraining_epochs`` epochs. The first ``patience`` checks
    after that never stop training (as in scArches). The best weights and prototypes
    (module ``state_dict``, deep-copied) are tracked and, with ``reload_best``, restored at
    the end of training.
    """

    def __init__(
        self,
        pretraining_epochs: int,
        early_stopping_metric: str = "val_unweighted_loss",
        mode: str = "min",
        threshold: float = 0.0,
        patience: int = 20,
        reduce_lr: bool = True,
        lr_patience: int = 13,
        lr_factor: float = 0.1,
        reload_best: bool = False,
    ):
        super().__init__()
        self.pretraining_epochs = pretraining_epochs
        self.metric = translate_metric(early_stopping_metric)
        self.mode = mode
        self.threshold = threshold
        self.patience = patience
        self.reduce_lr = reduce_lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.reload_best = reload_best

        self.epoch = 0
        self.wait = 0
        self.wait_lr = 0
        sign = np.inf if mode == "min" else -np.inf
        self.best_performance = sign
        self.best_performance_state = sign
        self.best_state_dict = None
        self.best_epoch = None
        self.stopped_epoch = None

    def _improvement(self, best, current):
        return best - current if self.mode == "min" else current - best

    def _step(self, value: float) -> tuple[bool, bool]:
        if self.patience == 0:
            return True, False
        self.epoch += 1
        if self.epoch < self.patience:
            return True, False
        if self.wait >= self.patience:
            return False, False
        lr_update = False
        if self.reduce_lr and self.wait_lr >= self.lr_patience:
            lr_update = True
            self.wait_lr = 0
        improvement = self._improvement(self.best_performance, value)
        if improvement > 0:
            self.best_performance = value
        if improvement < self.threshold:
            self.wait += 1
            self.wait_lr += 1
        else:
            self.wait = 0
            self.wait_lr = 0
        return True, lr_update

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.current_epoch < self.pretraining_epochs:
            return
        if self.metric not in trainer.callback_metrics:
            raise KeyError(
                f"Early stopping metric {self.metric!r} was not logged. "
                f"Available: {sorted(trainer.callback_metrics)}"
            )
        value = float(trainer.callback_metrics[self.metric])
        if self._improvement(self.best_performance_state, value) > 0:
            self.best_performance_state = value
            self.best_state_dict = copy.deepcopy(pl_module.module.state_dict())
            self.best_epoch = trainer.current_epoch
        cont, lr_update = self._step(value)
        if lr_update:
            for opt in trainer.optimizers:
                for group in opt.param_groups:
                    group["lr"] *= self.lr_factor
            logger.info("Early stopping: learning rate reduced by %s.", self.lr_factor)
        if not cont:
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True
            logger.info(
                "Stopping early: no improvement of more than %s in %s epochs.",
                self.threshold,
                self.patience,
            )

    def on_train_end(self, trainer, pl_module):
        if self.reload_best and self.best_state_dict is not None:
            pl_module.module.load_state_dict(self.best_state_dict)
            logger.info("Restored best state from epoch %s.", self.best_epoch)
