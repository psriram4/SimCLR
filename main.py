import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.datamodules import STL10DataModule
from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

from models.simclr import SimCLR
from transforms.stl10_transforms import SimCLRTrainTransform, SimCLRTestTransform

from tqdm import tqdm
import torch
import os


# def to_device(batch, device):
#     (img1, _), y = batch
#     img1 = img1.to(device)
#     y = y.to(device)
#     return img1, y

# online_finetuner = SSLOnlineEvaluator(z_dim=2048*2*2, num_classes=10)
# online_finetuner.to_device = to_device

# lr_logger = LearningRateLogger()
# callbacks = [online_finetuner, lr_logger]
# callbacks = [online_finetuner]

batch_size = 8
stl10_height = 32

dm = STL10DataModule(os.getcwd(), batch_size=batch_size)
dm.train_transforms = SimCLRTrainTransform(stl10_height)
dm.val_transforms = SimCLRTestTransform(stl10_height)

dm.prepare_data()
dm.setup()

train_samples = len(dm.train_dataloader())

model = SimCLR(batch_size=batch_size, num_samples=train_samples)
trainer = pl.Trainer(progress_bar_refresh_rate=10, accelerator="gpu", devices=1)
trainer.fit(model, dm)


