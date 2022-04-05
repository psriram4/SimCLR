import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.datamodules import STL10DataModule
from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from models.simclr import SimCLR
from transforms.stl10_transforms import SimCLRTrainTransform, SimCLRTestTransform

from tqdm import tqdm
import torch
import os

data_present = True

batch_size = 8
stl10_height = 96

dm = STL10DataModule(data_dir=".", batch_size=batch_size, train_transforms=SimCLRTrainTransform(stl10_height), \
    val_transforms=SimCLRTestTransform(stl10_height))

if not data_present:
    dm.prepare_data()

train_samples = len(dm.train_dataloader())

bar = TQDMProgressBar(refresh_rate=10)
checkpoint = ModelCheckpoint(dirpath="./saved_weights", filename="simclr-model")
model = SimCLR(batch_size=batch_size, num_samples=train_samples)
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(callbacks=[bar, checkpoint], accelerator="gpu", devices=1, logger=logger, max_epochs=1)
trainer.fit(model, dm)

print("Training done.")


