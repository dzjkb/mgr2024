import math
from typing import Any

from attrs import frozen
import pytorch_lightning as pl


@frozen
class CallbacksConfig:
    initial_latent_weight: float | None = None  # if `None` beta warmup is disabled
    target_latent_weight: float | None = None  # required if `initial_latent_loss` is not `None`
    latent_weight_warmup_length: int | None = (
        None  # required if `initial_latent_loss` is not `None`
    )
    encoder_training_len: int | None = None
    discriminator_warmup_len: int | None = None


def init_callbacks(cfg: CallbacksConfig) -> list[pl.Callback]:
    callbacks: list[pl.Callback] = [
        pl.callbacks.ModelCheckpoint(monitor="loss/validation_loss", filename="best"),
        pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=0, save_on_train_epoch_end=True),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg.initial_latent_weight:
        assert cfg.target_latent_weight is not None and cfg.latent_weight_warmup_length is not None
        callbacks.append(
            BetaWarmupCallback(
                initial_value=cfg.initial_latent_weight,
                target_value=cfg.target_latent_weight,
                warmup_len=cfg.latent_weight_warmup_length,
            )
        )

    if cfg.encoder_training_len:
        callbacks.append(EncoderTrainingPhaseCallback(duration=cfg.encoder_training_len, disc_warmup=cfg.discriminator_warmup_len or 0))

    return callbacks


class BetaWarmupCallback(pl.Callback):

    def __init__(
        self,
        initial_value: float,
        target_value: float,
        warmup_len: int,  # in batches
    ) -> None:
        super().__init__()
        self.state = {"training_steps": 0}
        self.warmup_len = warmup_len
        self.initial_value = initial_value
        self.target_value = target_value

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.state["training_steps"] += 1
        if self.state["training_steps"] >= self.warmup_len:
            pl_module.latent_loss_weight = self.target_value  # type: ignore
            return

        ratio = self.state["training_steps"] / self.warmup_len
        weight = math.log(self.initial_value) * (1 - ratio) + math.log(self.target_value) * ratio
        pl_module.latent_loss_weight = math.exp(weight)  # type: ignore

    def state_dict(self) -> dict[str, Any]:
        return self.state.copy()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state.update(state_dict)


class EncoderTrainingPhaseCallback(pl.Callback):

    def __init__(
        self,
        duration: int,  # in batches
        disc_warmup: int,
    ) -> None:
        super().__init__()
        self.state = {"training_steps": 0}
        self.duration = duration
        self.disc_warmup = disc_warmup

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.state["training_steps"] >= self.duration:
            pl_module.discrimination_phase = True  # type: ignore
            if self.state["training_steps"] < self.duration + self.disc_warmup:
                pl_module.discriminator_warmup_phase = True  # type: ignore
            else:
                pl_module.discriminator_warmup_phase = False  # type: ignore
        self.state["training_steps"] += 1

    def state_dict(self) -> dict[str, Any]:
        return self.state.copy()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state.update(state_dict)
