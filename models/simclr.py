import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam 
import pytorch_lightning as pl
from pl_bolts.models.self_supervised.resnets import resnet50
from losses.nt_xent_loss import nt_xent_loss
from models.projection_head import ProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers import LARS

class SimCLR(pl.LightningModule):
    def __init__(self, batch_size, num_samples, world_size=1, warmup_epochs=10, max_epochs=100, lars_lr=0.1, \
                    lars_eta=1e-3, opt_weight_decay=1e-6, loss_temperature=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.nt_xent_loss = nt_xent_loss
        self.encoder = resnet50()
        self.projection_head = ProjectionHead()

        self.encoder.conv1 = nn.Conv2d(
            3, 64, 
            kernel_size=3,
            stride=1,
            padding=1, 
            bias=False 
        )
        # self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bn', 'bias']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.}
        ]
     
    def forward(self, x):
        if isinstance(x, list):
            x = x[0]

        result = self.encoder(x)
        if isinstance(result, list):
            result = result[-1]
        return result

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

    def configure_optimizers(self):
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        ) 
        optimizer = LARS(parameters, lr=self.hparams.lars_lr)

        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        self.hparams.max_epochs = self.hparams.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_delay = LinearWarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )

        scheduler = {
            'scheduler': linear_warmup_cosine_delay,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        print(img1.shape)
        print(img2.shape)

        # (b, 3, 32, 32) --> (b, 2048, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]
            
        print(h1.shape)
        print(h2.shape)


        # (b, 2048, 2, 2) --> (b, 128)
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # result = pl.TrainResult(minimize=loss)
        # result.log('train_loss', loss, on_epoch=True)
        # return result
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('avg_val_loss', loss) 
        # return result
        return loss
