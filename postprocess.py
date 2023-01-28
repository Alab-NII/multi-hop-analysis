# -*- coding: utf-8 -*-
from glob import glob
import sys
import torch
from tqdm import tqdm

import utils


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
                 ans_type,
                 relations,
                 evidences,
                 evidence_ids,
                 q_ner_labels,
                 ctx_ner_labels,
                 #
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
        self.ans_type = ans_type,
        #
        self.relations=relations,
        self.evidences=evidences,
        self.evidence_ids=evidence_ids,
        self.q_ner_labels=q_ner_labels,
        self.ctx_ner_labels=ctx_ner_labels,
        #
        self.orig_answer_text = orig_answer_text
        self.answer_in_ques_entity_ids = answer_in_ques_entity_ids
        self.answer_in_ctx_entity_ids = answer_in_ctx_entity_ids
        self.answer_candidates_in_ctx_entity_ids= answer_candidates_in_ctx_entity_ids
        self.start_position = start_position
        self.end_position = end_position


def extract_answers(
    start_logits, end_logits, features, doc_tokens, n_best_size=20, max_answer_length=30):
    answer_cands = {((start_logits[0] + end_logits[0]).item(), "")}

    start_indexes = start_logits.numpy().argsort()[-1 : -n_best_size - 1 : -1].tolist()
    end_indexes = end_logits.numpy().argsort()[-1 : -n_best_size - 1 : -1].tolist()

    for start_index in start_indexes:
        for end_index in end_indexes:
            if (
                features.token_to_sequence(start_index) == 1
                and features.token_to_sequence(end_index) == 1
            ):
                score = (start_logits[start_index] + end_logits[end_index]).item()

                word_start_index = features.token_to_word(start_index)
                word_end_index = features.token_to_word(end_index)

                if (
                    word_end_index < word_start_index
                    or word_end_index - word_start_index + 1 > max_answer_length
                ):
                    continue

                words = doc_tokens[word_start_index : word_end_index + 1]

                answer_cands.add((score, " ".join(words)))

    # answer_cands = sorted(answer_cands, key=lambda x: x[0], reverse=True)[:n_best_size]
    answer_cands = sorted(answer_cands, key=lambda x: (-x[0], x[-1]))[:n_best_size]

    return answer_cands


if __name__ == "__main__":

    flag_task_evi = True
    prediction_file = sys.argv[1] 


    if "1task" in prediction_file or "2task_ans_sf_1_4" in prediction_file:
        flag_task_evi = False


    processed_data_file = sys.argv[2] 
    original_data_file = sys.argv[3] 

    label_encoders = utils.read_json("data/label_encoders.json") 

    # Load all data
    predictions = list(tqdm(utils.deserialize_objects(prediction_file)))
    processed_docs = list(
        tqdm(enumerate(utils.deserialize_objects(processed_data_file)))
    )
    original_docs = {
        doc.qas_id: doc
        for obj in utils.deserialize_objects(original_data_file)
        for doc in tqdm(obj)
    }

    output_file = prediction_file + ".processed"

    outputs = {
        "answer": {},
        "sp": {},
        "evidence": {},
    }

    for prediction in tqdm(predictions):
        num_supporting_fact_cands = []
        num_relation_cands = []

        for sample_index in prediction["sample_indices"]:
            assert sample_index == processed_docs[sample_index][0]

            sample = processed_docs[sample_index][-1]

            num_supporting_fact_cands.append(len(sample["supporting_fact_spans"]))
            num_relation_cands.append(len(sample["relation_pairs"]))

        assert sum(num_supporting_fact_cands) == prediction[
            "supporting_fact_preds"
        ].size(0)

        assert sum(num_relation_cands) == prediction["relation_preds"].size(0)

        all_supporting_fact_preds = prediction["supporting_fact_preds"].split(
            num_supporting_fact_cands
        )
        all_relation_preds = prediction["relation_preds"].split(num_relation_cands)

        for idx, sample_index in enumerate(prediction["sample_indices"]):
            sample = processed_docs[sample_index][1]

            original_doc = original_docs[sample["id"]]

            answer_classification_pred = label_encoders["inv_answers"][
                str(prediction["answer_classification_preds"][idx].item())
            ]

            answer_cands = extract_answers(
                prediction["answer_extraction_start_logits"][idx],
                prediction["answer_extraction_end_logits"][idx],
                sample["transformer_features"],
                original_doc.doc_tokens,
            )

            answer_span = {
                0: answer_cands[0][-1],
                1: "yes",
                2: "no",
                3: answer_cands[0][-1],
            }[answer_classification_pred]

            outputs["answer"][sample["id"]] = answer_span

            supporting_fact_preds = all_supporting_fact_preds[idx]

            assert len(original_doc.sent_names) == supporting_fact_preds.size(0)

            outputs["sp"][sample["id"]] = []
            # sp_results = []

            for supporting_fact_pred_idx in supporting_fact_preds.nonzero(
                as_tuple=True
            )[0]:
                supporting_fact_title, supporting_fact_index = original_doc.sent_names[
                    supporting_fact_pred_idx
                ]

                # 
                if supporting_fact_index > 0:
                    outputs["sp"][sample["id"]].append(
                        [supporting_fact_title, supporting_fact_index - 1]
                    )

            assert len(sample["entity_spans"]) == len(original_doc.ctx_entities_text)

            relation_preds = all_relation_preds[idx]

            outputs["evidence"][sample["id"]] = []

            if flag_task_evi == True:
                evidence_results = []

                for relation_pred_index in relation_preds.nonzero(as_tuple=True)[0]:
                    relation_pred_label = label_encoders["inv_relations"][
                        str(relation_preds[relation_pred_index].item())
                    ]
                    left_entity_index, right_entity_index = sample["relation_pairs"][
                        relation_pred_index
                    ]

                    evidence = [
                        original_doc.ctx_entities_text[left_entity_index],
                        relation_pred_label,
                        original_doc.ctx_entities_text[right_entity_index],
                    ]

                    evidence_results.append(evidence)

                evidence_results_set = set(tuple(i) for i in evidence_results)  
                evidence_results_final = [list(i) for i in evidence_results_set]
                outputs["evidence"][sample["id"]] = evidence_results_final

    utils.write_json(outputs, output_file)
