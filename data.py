# -*- coding: utf-8 -*-
import math

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import utils
from preprocess import NEGATIVE_CLASS


class Data(Dataset):
    def __init__(
        self,
        data_file,
        config,
        tokenizer,
        label_encoders,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        use_gold_data=True,
        negative_sampling_rate=1.0,
    ):
        super().__init__()

        assert 0.0 <= negative_sampling_rate <= 1.0

        if negative_sampling_rate < 1.0:
            assert (
                shuffle
            ), "It looks like you are using negative sampling for evaluation data"

            logger.warning(
                "(PLEASE ONLY USE FOR TRAINING DATA) "
                "Negative sampling is enabled for data: {}",
                data_file,
            )

        self.data_file = data_file
        self.config = config
        self.tokenizer = tokenizer
        self.label_encoders = label_encoders
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.use_gold_data = use_gold_data
        self.negative_sampling_rate = negative_sampling_rate

        # https://github.com/huggingface/transformers/blob/9a9314f6d9f352351490a95bd55e2d097409b5f6/src/transformers/models/big_bird/modeling_big_bird.py#L2033
        self.min_tokens = (5 + 2 * config.num_random_blocks) * config.block_size + 1

        self.samples = []

        for sample_index, sample in tqdm(
            enumerate(utils.deserialize_objects(data_file))
        ):
            # Useful for prediction
            sample["sample_index"] = sample_index

            self.samples.append(sample)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def get_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, batch):
        # For generating prediction
        sample_indices = []

        # For padding
        max_tokens = self.min_tokens

        # For accumulating spans & relations
        num_tokens = 0
        num_entities = 0
        num_supporting_facts = 0

        # For transformer model and answer extraction
        transformer_features = []
        words_masks = []
        context_tokens_masks = []
        subword_entity_indices = []

        # For answer classification
        question_types = []
        answer_labels = []

        # For supporting fact classification
        supporting_fact_spans = []
        supporting_fact_labels = []

        # For entity embeddings
        entity_spans = []
        entity_supporting_fact_indices = []
        entity_types = []

        # For relation extraction
        relation_pairs = []
        relation_masks = []
        relation_labels = []

        for sample in batch:
            # For generating prediction
            sample_indices.append(sample["sample_index"])

            # For padding
            max_tokens = max(max_tokens, len(sample["transformer_features"].input_ids))

            # For transformer model and answer extraction
            transformer_features.append(sample["transformer_features"])
            words_masks.append(sample["words_mask"])
            context_tokens_masks.append(sample["context_tokens_mask"])
            subword_entity_indices_ts = torch.tensor(sample["subword_entity_indices"])
            subword_entity_indices_masks = subword_entity_indices_ts == -1
            subword_entity_indices_ts += num_entities
            subword_entity_indices_ts[subword_entity_indices_masks] = -1
            subword_entity_indices.append(subword_entity_indices_ts.tolist())

            # For answer classification
            question_types.append(sample["question_type"])

            # For supporting fact classification
            supporting_fact_spans.append(
                torch.tensor(sample["supporting_fact_spans"]) + num_tokens
            )

            # For entity embeddings
            entity_spans.append(torch.tensor(sample["entity_spans"]) + num_tokens)
            entity_supporting_fact_indices.append(
                torch.tensor(sample["entity_supporting_fact_indices"])
                + num_supporting_facts
            )
            entity_types.extend(sample["entity_types"])

            # For relation extraction
            relation_pairs.append(torch.tensor(sample["relation_pairs"]) + num_entities)

            # For training data, add label information for computing loss
            if self.use_gold_data:
                # For answer extraction
                answer_start, answer_end = sample["answer_start"], sample["answer_end"]

                sample["transformer_features"]["start_positions"] = answer_start
                sample["transformer_features"]["end_positions"] = answer_end

                # For answer classification
                answer_labels.append(sample["answer_label"])

                # For supporting fact classification
                supporting_fact_labels.extend(sample["supporting_fact_labels"])

                # For relation extraction

                # Code for negative sampling tricks
                positive_relation_masks = (
                    torch.tensor(sample["relation_labels"])
                    != self.label_encoders["relations"][NEGATIVE_CLASS]
                )
                negative_relation_indices = torch.nonzero(
                    ~positive_relation_masks, as_tuple=True
                )[0]
                selected_negative_relation_indices = negative_relation_indices[
                    torch.randperm(len(negative_relation_indices))
                ][
                    : math.ceil(
                        len(negative_relation_indices) * self.negative_sampling_rate
                    )
                ]

                positive_relation_masks[selected_negative_relation_indices] = True

                relation_masks.append(positive_relation_masks)

                relation_labels.extend(sample["relation_labels"])

            # For accumulating spans & relations
            num_tokens += sample["num_tokens"]
            num_entities += sample["num_entities"]
            num_supporting_facts += len(sample["supporting_fact_spans"])

        features = {
            "sample_indices": torch.tensor(sample_indices),
            "transformer_features": self.tokenizer.pad(
                encoded_inputs=transformer_features,
                padding="max_length",
                max_length=max_tokens,
                return_tensors="pt",
            ),
            "words_masks": torch.tensor(
                [
                    words_mask + [False] * (max_tokens - len(words_mask))
                    for words_mask in words_masks
                ]
            ),
            "context_tokens_masks": torch.tensor(
                [
                    context_tokens_mask + [0] * (max_tokens - len(context_tokens_mask))
                    for context_tokens_mask in context_tokens_masks
                ]
            ),
            "subword_entity_indices": torch.tensor(
                [
                    subword_entity_index
                    + [-1] * (max_tokens - len(subword_entity_index))
                    for subword_entity_index in subword_entity_indices
                ]
            ),
            "question_types": torch.tensor(question_types),
            "supporting_fact_spans": torch.cat(supporting_fact_spans, dim=0),
            "entity_spans": torch.cat(entity_spans, dim=0),
            "entity_supporting_fact_indices": torch.cat(
                entity_supporting_fact_indices, dim=0
            ),
            "entity_types": torch.tensor(entity_types),
            "relation_pairs": torch.cat(relation_pairs, dim=0),
        }

        # For training data, add label information for computing loss
        if self.use_gold_data:
            # For answer classification
            features["answer_labels"] = torch.tensor(answer_labels)

            # For supporting fact classification
            features["supporting_fact_labels"] = torch.tensor(supporting_fact_labels)

            # For relation extraction
            relation_masks = torch.cat(relation_masks, dim=0)

            assert self.negative_sampling_rate < 1.0 or relation_masks.all()

            features["relation_pairs"] = features["relation_pairs"][relation_masks]
            features["relation_labels"] = torch.tensor(relation_labels)[relation_masks]

        return features