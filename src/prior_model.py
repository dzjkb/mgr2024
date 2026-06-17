import pytorch_lightning as pl
from attrs import frozen
from torch import optim, nn, Tensor
from zuko.flows import RealNVP


@frozen
class PriorConfig:
    latent_size: int
    layers: int
    lr: float
    betas: tuple[float, float] = (0.5, 0.9)


class Prior(pl.LightningModule):
    def __init__(
        self,
        latent_size: int,
        layers: int,
        lr: float,
        betas: tuple[float, float],
    ) -> None:
        super().__init__()
        self.flow = RealNVP(latent_size, transforms=layers, hidden_features=[latent_size])
        self.lr = lr
        self.betas = betas

    def forward(self, x: Tensor) -> Tensor:
        return -self.flow().log_prob(x).mean()

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        # opt = self.optimizers()
        loss = self.forward(batch[0])
        self.log("prior_train_loss", loss.detach(), on_step=False, on_epoch=True)
        return loss
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        loss = self.forward(batch[0])
        self.log("prior_validation_loss", loss.detach(), on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.flow.parameters(), lr=self.lr, betas=self.betas, fused=True)
        return optimizer
