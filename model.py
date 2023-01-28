# -*- coding: utf-8 -*-
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1, Accuracy
from transformers import AdamW, BigBirdModel, BigBirdPreTrainedModel, get_scheduler
from transformers.activations import get_activation

import utils


class Activation(nn.Module):
    def __init__(self, activation_name):
        super().__init__()

        self.activation_fn = get_activation(activation_name)

    def forward(self, x):
        return self.activation_fn(x)


class QuestionAnsweringModel(BigBirdPreTrainedModel):
    def __init__(self, configs, label_encoders):
        super().__init__(configs)

        # Transformer model
        self.bert = BigBirdModel(configs, add_pooling_layer=False)

        self.dropout = nn.Dropout(configs.hidden_dropout_prob)

        self.activation = Activation(configs.hidden_act)

        # Answer extraction
        self.answer_extraction_head = nn.Sequential(
            # nn.Linear(configs.hidden_size, configs.hidden_size),
            nn.Linear(configs.hidden_size * 2, configs.hidden_size),
            self.activation,
            self.dropout,
            nn.Linear(configs.hidden_size, 2),
        )

        # Answer classification
        self.question_type_dim = 25

        self.question_type_embedder = nn.Embedding(
            num_embeddings=len(label_encoders["questions"]),
            embedding_dim=self.question_type_dim,
        )

        self.answer_classification_head = nn.Sequential(
            nn.Linear(
                configs.hidden_size + self.question_type_dim, configs.hidden_size
            ),
            self.activation,
            self.dropout,
            nn.Linear(configs.hidden_size, len(label_encoders["answers"])),
        )

        # Supporting fact classification
        self.supporting_fact_classification_head = nn.Sequential(
            nn.Linear(configs.hidden_size * 2, configs.hidden_size),
            self.activation,
            self.dropout,
            nn.Linear(configs.hidden_size, len(label_encoders["supporting_facts"])),
        )

        # Relation extraction
        self.entity_type_dim = 50

        self.entity_type_embedder = nn.Embedding(
            num_embeddings=len(label_encoders["entities"]),
            embedding_dim=self.entity_type_dim,
        )

        self.entity_embedding_head = nn.Sequential(
            nn.Linear(
                # configs.hidden_size * 2 + self.entity_type_dim, configs.hidden_size
                configs.hidden_size * 4 + self.entity_type_dim,
                configs.hidden_size,
            ),
            self.activation,
        )

        self.relation_extraction_head = nn.Sequential(
            nn.Linear(configs.hidden_size * 2, configs.hidden_size),
            self.activation,
            self.dropout,
            nn.Linear(configs.hidden_size, len(label_encoders["relations"])),
        )

        # MUST HAVE FOR INITIALIZATION
        self.init_weights()

    def forward(self, batch, *args, **kwargs):
        outputs = {}

        # Transformer model
        answer_start_positions = batch["transformer_features"].pop(
            "start_positions", None
        )
        answer_end_positions = batch["transformer_features"].pop("end_positions", None)

        transformer_outputs = self.bert(**batch["transformer_features"])

        subtoken_embeddings = self.dropout(transformer_outputs[0])

        token_embeddings = subtoken_embeddings[batch["words_masks"]]

        # Answer classification
        question_type_embeddings = self.question_type_embedder(batch["question_types"])

        question_type_embeddings = self.dropout(question_type_embeddings)

        answer_classification_embeddings = torch.cat(
            (subtoken_embeddings[:, 0, :], question_type_embeddings), dim=-1
        )

        answer_classification_logits = self.answer_classification_head(
            answer_classification_embeddings
        )

        outputs[
            "answer_classification_preds"
        ] = answer_classification_logits.detach().argmax(dim=-1)

        # Supporting fact classification
        supporting_fact_starts, supporting_fact_ends = batch[
            "supporting_fact_spans"
        ].split(1, dim=-1)

        supporting_fact_start_embeddings = token_embeddings[
            supporting_fact_starts.squeeze(dim=-1)
        ]
        supporting_fact_end_embeddings = token_embeddings[
            supporting_fact_ends.squeeze(dim=-1)
        ]

        supporting_fact_embeddings = torch.cat(
            (supporting_fact_start_embeddings, supporting_fact_end_embeddings), dim=-1
        )

        supporting_fact_logits = self.supporting_fact_classification_head(
            supporting_fact_embeddings
        )

        outputs["supporting_fact_preds"] = supporting_fact_logits.detach().argmax(
            dim=-1
        )

        # Relation extraction
        entity_span_starts, entity_span_ends = batch["entity_spans"].split(1, dim=-1)

        entity_start_embeddings = token_embeddings[entity_span_starts.squeeze(dim=-1)]
        entity_end_embeddings = token_embeddings[entity_span_ends.squeeze(dim=-1)]

        entity_type_embeddings = self.entity_type_embedder(batch["entity_types"])

        entity_type_embeddings = self.dropout(entity_type_embeddings)

        entity_supporting_fact_embeddings = supporting_fact_embeddings[
            batch["entity_supporting_fact_indices"]
        ]

        # entity_embeddings = torch.cat(
        #     (entity_start_embeddings, entity_end_embeddings, entity_type_embeddings),
        #     dim=-1,
        # )
        entity_embeddings = torch.cat(
            (
                entity_start_embeddings,
                entity_end_embeddings,
                entity_type_embeddings,
                entity_supporting_fact_embeddings,
            ),
            dim=-1,
        )

        entity_embeddings = self.entity_embedding_head(entity_embeddings)

        left_entity_indices, right_entity_indices = batch["relation_pairs"].split(
            1, dim=-1
        )

        left_entity_embeddings = entity_embeddings[left_entity_indices.squeeze(dim=-1)]
        right_entity_embeddings = entity_embeddings[
            right_entity_indices.squeeze(dim=-1)
        ]

        relation_embeddings = torch.cat(
            (left_entity_embeddings, right_entity_embeddings), dim=-1
        )

        relation_logits = self.relation_extraction_head(relation_embeddings)

        outputs["relation_preds"] = relation_logits.detach().argmax(dim=-1)

        # Answer extraction
        padded_entity_embeddings = F.pad(entity_embeddings, (0, 0, 1, 0))

        subtoken_entity_embeddings = torch.cat(
            (
                subtoken_embeddings,
                padded_entity_embeddings[batch["subword_entity_indices"] + 1],
            ),
            dim=-1,
        )

        answer_extraction_logits = self.answer_extraction_head(
            subtoken_entity_embeddings
        )

        (
            answer_extraction_start_logits,
            answer_extraction_end_logits,
        ) = answer_extraction_logits.split(1, dim=-1)

        answer_extraction_start_logits = answer_extraction_start_logits.squeeze(dim=-1)
        answer_extraction_end_logits = answer_extraction_end_logits.squeeze(dim=-1)

        outputs["answer_extraction_start_logits"] = (
            answer_extraction_start_logits.detach()
            + (batch["context_tokens_masks"] - 1) * 1e6
        )
        outputs["answer_extraction_end_logits"] = (
            answer_extraction_end_logits.detach()
            + (batch["context_tokens_masks"] - 1) * 1e6
        )

        # Compute loss if has gold labels
        if "answer_labels" in batch:
            # Answer extraction
            outputs["answer_extraction_loss"] = (
                F.cross_entropy(answer_extraction_start_logits, answer_start_positions)
                + F.cross_entropy(answer_extraction_end_logits, answer_end_positions)
            ) / 2

            # Answer classification
            outputs["answer_classification_loss"] = F.cross_entropy(
                answer_classification_logits, batch["answer_labels"]
            )

            # Supporting fact classification
            outputs["supporting_fact_loss"] = F.cross_entropy(
                supporting_fact_logits, batch["supporting_fact_labels"]
            )

            # Relation extraction
            outputs["relation_loss"] = F.cross_entropy(
                relation_logits, batch["relation_labels"]
            )

        return outputs


class Model(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

        self.save_hyperparameters()

        self.label_encoders = utils.read_json(
            self.hparams.configs["label_encoders_file"]
        )

        self.model = QuestionAnsweringModel.from_pretrained(
            self.hparams.configs["pretrained_model_dir"],
            label_encoders=self.label_encoders,
        )

        # Metrics
        self.answer_start_acc = Accuracy()
        self.answer_end_acc = Accuracy()

        self.answer_classification_acc = Accuracy()

        self.supporting_fact_f1 = F1(
            ignore_index=self.label_encoders["supporting_facts"]["false"]
        )

        self.relation_f1 = F1(ignore_index=self.label_encoders["relations"]["@@NONE@@"])

        # Dynamic Weight Average (https://arxiv.org/abs/1803.10704)
        self.losses = defaultdict(list)

    def training_step(self, batch, *args, **kwargs):
        answer_start_positions = batch["transformer_features"]["start_positions"]
        answer_end_positions = batch["transformer_features"]["end_positions"]

        outputs = self.model(batch, *args, **kwargs)

        # Overall loss
        loss = 0.0

        # Answer extraction
        loss += outputs["answer_extraction_loss"]

        self.answer_start_acc(
            outputs["answer_extraction_start_logits"].argmax(dim=-1),
            answer_start_positions,
        )
        self.answer_end_acc(
            outputs["answer_extraction_end_logits"].argmax(dim=-1), answer_end_positions
        )

        self.log("answer_start_acc", self.answer_start_acc, on_step=True, on_epoch=True)
        self.log("answer_end_acc", self.answer_end_acc, on_step=True, on_epoch=True)

        # Answer classification
        loss += outputs["answer_classification_loss"]

        self.answer_classification_acc(
            outputs["answer_classification_preds"], batch["answer_labels"]
        )

        self.log(
            "answer_classification_acc",
            self.answer_classification_acc,
            on_step=True,
            on_epoch=True,
        )

        # Supporting fact classification
        loss += outputs["supporting_fact_loss"]

        self.supporting_fact_f1(
            outputs["supporting_fact_preds"], batch["supporting_fact_labels"]
        )

        self.log(
            "supporting_fact_f1", self.supporting_fact_f1, on_step=True, on_epoch=True
        )

        # Relation extraction
        loss += outputs["relation_loss"]

        self.relation_f1(outputs["relation_preds"], batch["relation_labels"])

        self.log("relation_f1", self.relation_f1, on_step=True, on_epoch=True)

        return loss

    def predict_step(self, batch, *args, **kwargs):
        outputs = self.model(batch, *args, **kwargs)

        outputs["sample_indices"] = batch["sample_indices"]

        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                outputs[k] = v.cpu()

        return outputs

    @property
    def num_training_steps(self):
        # https://github.com/PyTorchLightning/lightning-transformers/blob/fac1e28cd7b8e73e1fdad1c77f9ffdcd55859d9b/lightning_transformers/core/model.py#L56
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches > 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            dataset_size = int(
                len(self.train_dataloader()) * self.trainer.limit_train_batches
            )
        else:
            dataset_size = len(self.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)

        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices

        max_estimated_steps = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs

        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps

        return max_estimated_steps

    def compute_warmup(self, num_training_steps, num_warmup_steps):
        if num_training_steps < 0:
            num_training_steps = self.num_training_steps

        if isinstance(num_warmup_steps, float):
            num_warmup_steps = int(num_warmup_steps * num_training_steps)

        return num_training_steps, num_warmup_steps

    def configure_optimizers(self):
        model = self.model

        # https://github.com/huggingface/transformers/blob/3b1f5caff26c08dfb74a76de1163f4becde9e828/examples/pytorch/question-answering/run_qa_no_trainer.py#L628
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.configs["optimizer"]["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.configs["optimizer"]["lr"],
            eps=self.hparams.configs["optimizer"]["eps"],
        )

        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=self.hparams.configs["scheduler"]["num_training_steps"],
            num_warmup_steps=self.hparams.configs["scheduler"]["num_warmup_steps"],
        )

        scheduler = get_scheduler(
            name=self.hparams.configs["scheduler"]["name"],
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }