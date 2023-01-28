# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from data import Data
from model import Model


def train():
    # For reproducibility
    pl.seed_everything(seed=42, workers=True)

    folder_name = "data" 
    # Configs
    configs = {
        "pretrained_model_dir": "./bigbird-roberta-base",
        "label_encoders_file": "{}/label_encoders.json".format(folder_name),
        "data": {
            "train": {
                "data_file": "{}/train.pkl".format(folder_name),
                "batch_size": 4, # 
                "shuffle": True,
                "num_workers": 20,
                "use_gold_data": True,
                "negative_sampling_rate": 1.0,
            },
        },
        "optimizer": {"lr": 3e-5, "weight_decay": 0.01, "eps": 1e-08},
        "scheduler": {
            "name": "linear",
            "num_training_steps": -1,
            "num_warmup_steps": 0.1,
        },

        # 1 GPU 
        "trainer": {
            "gpus": [0],
            "accumulate_grad_batches": 8,
            "max_epochs": 20,
            "precision": 16,
            "deterministic": True,
        },

    }

    # Model
    model = Model(configs)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model.hparams.configs["pretrained_model_dir"]
    )

    # Data
    train_data = Data(
        config=model.model.config,
        tokenizer=tokenizer,
        label_encoders=model.label_encoders,
        **model.hparams.configs["data"]["train"]
    )

    train_dataloader = train_data.get_dataloader()

    # Training
    trainer = pl.Trainer(
        callbacks=[ModelCheckpoint(dirpath="checkpoints/3task/", save_top_k=-1)], **model.hparams.configs["trainer"]
    )

    trainer.fit(model, train_dataloader=train_dataloader)


if __name__ == "__main__":
    train()
