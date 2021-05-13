import os
import sys

sys.path.append(os.path.join(os.environ["ALFRED_ROOT"]))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "models"))

import os
import torch
import pprint
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from models.utils.helper_utils import optimizer_to
from models.model.vlnce.loader import DataModuleArguments, DataModule
from models.model.vlnce import Module, ModuleArguments


class Arguments(ModuleArguments, DataModuleArguments):
    # decay_epoch: int = 10
    epoch: int = 20
    seed: int = 42
    fast: bool = False


if __name__ == "__main__":
    args = Arguments()

    pl.seed_everything(args.seed, workers=True)

    # load model
    logger = TensorBoardLogger("tb_logs", name="vlnce")
    trainer = pl.Trainer(
        gpus=1,
        log_gpu_memory="min_max",
        max_epochs=args.epoch,
        log_every_n_steps=50,
        fast_dev_run=args.fast,
        logger=logger,
    )
    dataloader = DataModule(args)
    model = Module(args)
    trainer.fit(model, dataloader)
