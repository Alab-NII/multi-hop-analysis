# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from itertools import permutations

from loguru import logger
from tqdm import tqdm
from transformers import BigBirdTokenizerFast

import utils

NEGATIVE_CLASS = "@@NONE@@"


class Example(object):
    def __init__(self,
                 qas_id,
                 qas_type,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 sup_para_id,
                 ques_entities_text,
                 ctx_entities_text,
                 para_start_end_position,
                 sent_start_end_position,
                 ques_entity_start_end_position,
                 ctx_entity_start_end_position,
                 question_text,
                 question_word_to_char_idx,
                 ctx_text,
                 ctx_word_to_char_idx,
                 # 
                 relations,
                 evidences,
                 evidence_ids,
                 q_ner_labels,
                 ctx_ner_labels,
                 #
                 edges=None,
                 orig_answer_text=None,
                 answer_in_ques_entity_ids=None,
                 answer_in_ctx_entity_ids=None,
                 answer_candidates_in_ctx_entity_ids=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.sup_para_id = sup_para_id
        self.ques_entities_text = ques_entities_text
        self.ctx_entities_text = ctx_entities_text
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.ques_entity_start_end_position = ques_entity_start_end_position
        self.ctx_entity_start_end_position = ctx_entity_start_end_position
        self.question_word_to_char_idx = question_word_to_char_idx
        self.ctx_text = ctx_text
        self.ctx_word_to_char_idx = ctx_word_to_char_idx
        #
        self.relations=relations,
        self.evidences=evidences,
        self.evidence_ids=evidence_ids,
        self.q_ner_labels=q_ner_labels,
        self.ctx_ner_labels=ctx_ner_labels,
        #
        self.edges = edges
        self.orig_answer_text = orig_answer_text
        self.answer_in_ques_entity_ids = answer_in_ques_entity_ids
        self.answer_in_ctx_entity_ids = answer_in_ctx_entity_ids
        self.answer_candidates_in_ctx_entity_ids= answer_candidates_in_ctx_entity_ids
        self.start_position = start_position
        self.end_position = end_position


def search_span_index_for_a_span(span_start, span_end, spans):
    for span_index, (_span_start, _span_end) in enumerate(spans):
        if _span_start <= span_start <= span_end <= _span_end:
            return span_index
    return -1


def search_span_index_for_an_index(index, spans):
    for span_index, (_span_start, _span_end) in enumerate(spans):
        if _span_start <= index <= _span_end:
            return span_index
    return -1


def prepare_training_data(data_file, tokenizer, label_encoders):
    for obj in utils.deserialize_objects(data_file):
        for example in tqdm(obj):
            all_tokens = example.question_tokens + example.doc_tokens

            doc = {
                "id": example.qas_id,
                "num_tokens": len(all_tokens),
                "num_question_tokens": len(example.question_tokens),
                "question_type": label_encoders["questions"].setdefault(
                    example.qas_type, len(label_encoders["questions"])
                ),
                "answer_label": label_encoders["answers"].setdefault(
                    example.ans_type[-1], len(label_encoders["answers"])
                ),
                "supporting_fact_spans": [],
                "supporting_fact_labels": [],
                "entity_spans": [],
                "entity_supporting_fact_indices": [],
                "entity_types": [],
                "relation_pairs": [],
                "relation_labels": [],
                "subword_entity_indices": [],
            }

            doc["transformer_features"] = tokenizer(
                example.question_tokens,
                example.doc_tokens,
                is_split_into_words=True,
                return_token_type_ids=True,
                return_special_tokens_mask=True,
            )

            word_ids = doc["transformer_features"].word_ids()
            token_type_ids = doc["transformer_features"].pop("token_type_ids")
            special_tokens_mask = doc["transformer_features"].pop("special_tokens_mask")

            doc["words_mask"] = [
                i == 0 and j
                for i, j in zip(
                    special_tokens_mask,
                    [True] + [i != j for i, j in zip(word_ids[:-1], word_ids[1:])],
                )
            ]

            assert len(word_ids) <= tokenizer.model_max_length
            assert sum(doc["words_mask"]) == doc["num_tokens"]

            word_indices = [-1] * len(doc["words_mask"])

            none_num = 0

            for subword_index, word_id in enumerate(word_ids):
                if word_id is None:
                    none_num += 1
                else:
                    if none_num == 2:
                        word_id += doc["num_question_tokens"]

                    word_indices[subword_index] = word_id

            assert none_num == 3

            cls_index = doc["transformer_features"].input_ids.index(
                tokenizer.cls_token_id
            )

            assert cls_index == 0

            context_tokens_mask = [
                ~i & j for i, j in zip(special_tokens_mask, token_type_ids)
            ]

            context_tokens_mask[cls_index] = 1

            doc["context_tokens_mask"] = context_tokens_mask

            if example.ans_type[-1] < 3:
                doc["answer_start"] = cls_index
                doc["answer_end"] = cls_index
            else:
                doc["answer_start"] = (
                    doc["transformer_features"]
                    .word_to_tokens(example.start_position[0], sequence_index=1)
                    .start
                )
                doc["answer_end"] = (
                    doc["transformer_features"]
                    .word_to_tokens(example.end_position[0], sequence_index=1)
                    .end
                    - 1
                )

                assert doc["answer_start"] <= doc["answer_end"]

                assert example.doc_tokens[example.start_position[0]].startswith(
                    doc["transformer_features"].tokens()[doc["answer_start"]][1:]
                ) and example.doc_tokens[example.end_position[0]].endswith(
                    doc["transformer_features"].tokens()[doc["answer_end"]][1:]
                )

            gold_supporting_facts = set(example.sup_fact_id)

            for supporting_fact_index, (
                supporting_fact_start,
                supporting_fact_end,
            ) in enumerate(example.sent_start_end_position):
                assert supporting_fact_start <= supporting_fact_end

                new_supporting_fact_start = (
                    supporting_fact_start + doc["num_question_tokens"]
                )
                new_supporting_fact_end = (
                    supporting_fact_end + doc["num_question_tokens"]
                )

                assert (
                    example.doc_tokens[supporting_fact_start : supporting_fact_end + 1]
                    == all_tokens[
                        new_supporting_fact_start : new_supporting_fact_end + 1
                    ]
                )

                doc["supporting_fact_spans"].append(
                    (new_supporting_fact_start, new_supporting_fact_end)
                )
                doc["supporting_fact_labels"].append(
                    label_encoders["supporting_facts"].setdefault(
                        supporting_fact_index in gold_supporting_facts,
                        len(label_encoders["supporting_facts"]),
                    )
                )

            for (entity_start, entity_end), entity_type in zip(
                example.ctx_entity_start_end_position, example.ctx_ner_labels[-1]
            ):
                assert entity_start <= entity_end

                new_entity_start = entity_start + doc["num_question_tokens"]
                new_entity_end = entity_end + doc["num_question_tokens"]

                assert (
                    example.doc_tokens[entity_start : entity_end + 1]
                    == all_tokens[new_entity_start : new_entity_end + 1]
                )

                doc["entity_spans"].append((new_entity_start, new_entity_end))
                doc["entity_supporting_fact_indices"].append(
                    search_span_index_for_a_span(
                        new_entity_start, new_entity_end, doc["supporting_fact_spans"]
                    )
                )
                assert doc["entity_supporting_fact_indices"][-1] >= 0

                doc["entity_types"].append(
                    label_encoders["entities"].setdefault(
                        entity_type, len(label_encoders["entities"])
                    )
                )

            doc["num_entities"] = len(doc["entity_spans"])

            for word_index in word_indices:
                doc["subword_entity_indices"].append(
                    search_span_index_for_an_index(word_index, doc["entity_spans"])
                )

            gold_relations = {
                (subject_id, object_id): label_encoders["relations"].setdefault(
                    relation_type, len(label_encoders["relations"])
                )
                for subject_id, relation_type, object_id in example.evidence_ids[-1]
            }

            for left_entity_index, right_entity_index in permutations(
                range(doc["num_entities"]), 2
            ):
                doc["relation_pairs"].append((left_entity_index, right_entity_index))
                doc["relation_labels"].append(
                    gold_relations.get(
                        (left_entity_index, right_entity_index),
                        label_encoders["relations"][NEGATIVE_CLASS],
                    )
                )
            # count how many candicates cover the gold evidence
            # remove all examples that missing candidicates cannot cover all evidence
            #
            yield doc


if __name__ == "__main__":
    # data_dir = "data"
    data_dir = "data"
    # 
    pretrained_model_dir = "./bigbird-roberta-base"

    logger.info("Preparing training data...")

    tokenizer = BigBirdTokenizerFast.from_pretrained(pretrained_model_dir)

    label_encoders = defaultdict(dict)

    label_encoders["relations"][NEGATIVE_CLASS] = 0

    for data_file in ("dev.gz", "train.gz", "test.gz"): # 
        data_file = os.path.join(data_dir, data_file)

        if os.path.isfile(data_file):
            logger.info("Processing: {}", data_file)

            utils.serialize_objects(
                prepare_training_data(data_file, tokenizer, label_encoders),
                data_file.replace(".gz", ".pkl"),
            )

    for label_encoder_name, label_encoder in list(label_encoders.items()):
        label_encoders[f"inv_{label_encoder_name}"] = {
            v: k for k, v in label_encoder.items()
        }

    utils.write_json(
        label_encoders, os.path.join(data_dir, "label_encoders.json"), indent=2
    )

    logger.info("Done!")