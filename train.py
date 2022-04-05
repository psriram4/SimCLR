import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from pl_bolts.datamodules import STL10DataModule
from data.stl10_datamodule import STL10DataModule
from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from models.simclr import SimCLR
from transforms.stl10_transforms import SimCLRTrainTransform, SimCLREvalTransform, SimCLRTestTransform

from tqdm import tqdm
import torch
import os

data_present = True

batch_size = 8
stl10_height = 96


dm = STL10DataModule(data_dir=".", batch_size=batch_size, train_val_split=100, num_workers=8)
normalization = stl10_normalization()

dm.train_dataloader = dm.train_dataloader_mixed
dm.val_dataloader = dm.val_dataloader_mixed
dm.train_transforms = SimCLRTrainTransform(input_height=stl10_height, normalize=normalization)
dm.val_transforms = SimCLREvalTransform(input_height=stl10_height, normalize=normalization)

if not data_present:
    dm.prepare_data()

train_samples = dm.num_unlabeled_samples

# print(train_samples)
# 1/0

bar = TQDMProgressBar(refresh_rate=10)
# checkpoint = ModelCheckpoint(dirpath="./saved_weights", filename="simclr-model")
checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
model = SimCLR(batch_size=batch_size, num_samples=train_samples)
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(callbacks=[bar, checkpoint], accelerator="gpu", devices=1, logger=logger, max_epochs=100, max_steps=-1)
trainer.fit(model, dm)

print("Training done.")


