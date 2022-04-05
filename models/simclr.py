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
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

class SimCLR(pl.LightningModule):
    def __init__(self, batch_size, num_samples, world_size=1, warmup_epochs=10, max_epochs=100, lars_lr=0.1, \
                    lars_eta=1e-3, learning_rate=1e-3, opt_weight_decay=1e-6, loss_temperature=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.nt_xent_loss = nt_xent_loss
        self.encoder = resnet50(return_all_feature_maps=True)
        self.projection_head = ProjectionHead()
        self.training_losses = []
        self.validation_losses = []

        self.encoder.conv1 = nn.Conv2d(
            3, 64, 
            kernel_size=3,
            stride=1,
            padding=1, 
            bias=False 
        )
        


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
        
        result = self.projection_head.avgpool(result)
        result = self.projection_head.flatten(result)

        return result

    def setup(self, stage):
        global_batch_size = self.trainer.world_size * self.hparams.batch_size
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size
        print(self.hparams.num_samples)
        print(global_batch_size)
        print(self.train_iters_per_epoch)

    def configure_optimizers(self):
        parameters = self.exclude_from_wt_decay(
            self.named_parameters(),
            weight_decay=self.hparams.opt_weight_decay
        ) 
        optimizer = LARS(parameters, lr=self.hparams.learning_rate, momentum=0.9, weight_decay=self.hparams.opt_weight_decay, trust_coefficient=0.001)

        self.hparams.warmup_epochs = self.hparams.warmup_epochs * self.train_iters_per_epoch
        self.hparams.max_epochs = self.hparams.max_epochs * self.train_iters_per_epoch

        linear_warmup_cosine_delay = LinearWarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0,
            eta_min=0
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            linear_warmup_decay(self.hparams.warmup_epochs, self.hparams.max_epochs, cosine=True),
        )
        

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def shared_step(self, batch, batch_idx):

        # print(batch)
        # 1/0
        # #  if self.dataset == "stl10":
        # unlabeled_batch = batch[0]
        # batch = unlabeled_batch

        (img1, img2, _), y = batch
        
        # (img1, img2), y = batchx

        # (b, 3, 32, 32) --> (b, 2048, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

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
        # self.training_losses.append(loss.detach().item())
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # result = pl.EvalResult(checkpoint_on=loss)
        # result.log('avg_val_loss', loss) 
        # return result
        # self.validation_losses.append(loss.detach().item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
