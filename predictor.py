# -*- coding: utf-8 -*-
import os
import sys
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

import utils
from data import Data
from model import Model


def predict(model_file, data_file, prediction_file, gpu_id=0):
    # For reproducibility
    pl.seed_everything(seed=42, workers=True) # 
    
    model = Model.load_from_checkpoint(model_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model.hparams.configs["pretrained_model_dir"]
    )

    # Data
    test_data = Data(
        data_file=data_file,
        config=model.model.config,
        tokenizer=tokenizer,
        label_encoders=model.label_encoders,
        batch_size=100,
        shuffle=False,
        num_workers=20,
        use_gold_data=False,
        negative_sampling_rate=1.0,
    )

    test_dataloader = test_data.get_dataloader()

    trainer = pl.Trainer(gpus=[gpu_id], precision=16, logger=False, deterministic=True)

    predictions = trainer.predict(
        model, dataloaders=test_dataloader, return_predictions=True
    )

    utils.serialize_objects(predictions, prediction_file)


if __name__ == "__main__":

    model_file = sys.argv[1] + ".ckpt"

    data_file = sys.argv[2] 

    prediction_file = (
        os.path.splitext(model_file)[0]
        + "_"
        + os.path.splitext(os.path.basename(data_file))[0]
        + ".preds"
    )

    predict(
        model_file=model_file,
        data_file=data_file,
        prediction_file=prediction_file,
        gpu_id=0, 
    )
