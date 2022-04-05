import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.datamodules import STL10DataModule
from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner

from models.simclr import SimCLR
from transforms.stl10_transforms import SimCLRTrainTransform, SimCLRTestTransform, SimCLRFinetuneTransform

from tqdm import tqdm
import torch
import os

data_present = True

batch_size = 64
stl10_height = 96
ckpt_path = "./saved_weights/simclr-model.ckpt"


dm = STL10DataModule(data_dir=".", batch_size=batch_size, num_workers=8)

dm.train_dataloader = dm.train_dataloader_labeled
dm.val_dataloader = dm.val_dataloader_labeled
num_samples = 1

dm.train_transforms = SimCLRFinetuneTransform(
    normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=False
)
dm.val_transforms = SimCLRFinetuneTransform(
    normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=True
)
dm.test_transforms = SimCLRFinetuneTransform(
    normalize=stl10_normalization(), input_height=dm.size()[-1], eval_transform=True
)

model = SimCLR(batch_size=batch_size, num_samples=1).load_from_checkpoint(ckpt_path, strict=False)

print("Model loaded.")

tuner = SSLFineTuner(
    model,
    in_features=2048,
    num_classes=dm.num_classes,
    epochs=100,
    hidden_dim=None,
    dropout=0.0,
    learning_rate=0.3,
    weight_decay=1e-6,
    nesterov=False,
    scheduler_type="cosine",
    gamma=0.1,
    final_lr=0.0,
)

trainer = pl.Trainer(
    gpus=1,
    num_nodes=1,
    precision=16,
    max_epochs=100,
    accelerator="ddp",
    sync_batchnorm=False,
)

trainer.fit(tuner, dm)
trainer.test(datamodule=dm)