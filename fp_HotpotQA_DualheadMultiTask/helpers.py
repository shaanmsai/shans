import numpy as np
import collections
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction, ElectraConfig, ElectraForSequenceClassification
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.metrics import f1_score


QA_MAX_ANSWER_LENGTH = 30


# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def prepare_dataset_nli(examples, tokenizer, max_seq_length=None):
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['premise'],
        examples['hypothesis'],
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


# function to compute F1 for Supporting Fact Prediction
def compute_sf_f1(sf_preds: np.ndarray, sf_labels: np.ndarray):
    # Flatten arrays
    sf_preds_flat = sf_preds.flatten()
    sf_labels_flat = sf_labels.flatten()

    # Handle the edge case where F1 might be undefined (e.g., no positive labels/predictions)
    if sf_labels_flat.sum() == 0 and sf_preds_flat.sum() == 0:
        return {'supporting_fact_f1': 1.0}
    
    return {
        'supporting_fact_f1': f1_score(sf_labels_flat, sf_preds_flat)
    }


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Includes logic for generating supporting_fact_labels (Answer Span Proxy).
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length
    
    #  full context 
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    # New Label for Supporting Facts
    tokenized_examples["supporting_fact_labels"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        # -------------------------------------------------------------
        # Supporting Fact Labeling Logic (Answer Span Proxy)
        # -------------------------------------------------------------
        sf_labels = [0] * len(input_ids)
        has_answer = len(answers["answer_start"]) > 0

        if has_answer:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # Label all tokens within the ground-truth answer span as supporting facts (label=1)
            for token_idx in range(len(input_ids)):
                offset = offsets[token_idx]
                sequence_id = sequence_ids[token_idx]
                
                # Only label tokens belonging to the context (sequence_id == 1)
                if sequence_id == 1 and offset is not None:
                    token_start_char, token_end_char = offset
                    
                    # Check if the token overlaps with the answer span (proxy for supporting facts)
                    if token_start_char >= start_char and token_end_char <= end_char:
                        sf_labels[token_idx] = 1
        
        # -------------------------------------------------------------
        # QA Start/End Position Labeling Logic (Original)
        # -------------------------------------------------------------
        
        if not has_answer:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            # Must append sf_labels here too
            tokenized_examples["supporting_fact_labels"].append(sf_labels) 
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                # Must append sf_labels here too
                tokenized_examples["supporting_fact_labels"].append(sf_labels)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(
                    token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
                # Must append sf_labels here too
                tokenized_examples["supporting_fact_labels"].append(sf_labels)

    return tokenized_examples


# This function preprocesses the validation dataset, now including supporting_fact_labels.
def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []
    # ADD NEW COLUMN FOR SUPPORTING FACTS
    tokenized_examples["supporting_fact_labels"] = [] 
    answers = examples["answers"] 
    

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1
        sample_index = sample_mapping[i]
        
        # -------------------------------------------------------------
        # Supporting Fact Labeling Logic (Mirroring Training Proxy)
        # -------------------------------------------------------------
        input_ids = tokenized_examples["input_ids"][i]
        sf_labels = [0] * len(input_ids)
        
        # We need to check if the original example has an answer to create the proxy
        if len(answers[sample_index]["answer_start"]) > 0:
            start_char = answers[sample_index]["answer_start"][0]
            end_char = start_char + len(answers[sample_index]["text"][0])
            offsets = tokenized_examples["offset_mapping"][i]
            
            for token_idx in range(len(input_ids)):
                offset = offsets[token_idx]
                sequence_id = sequence_ids[token_idx]
                
                if sequence_id == 1 and offset is not None:
                    token_start_char, token_end_char = offset
                    
                    if token_start_char >= start_char and token_end_char <= end_char:
                        sf_labels[token_idx] = 1

        # 3. Append the final labels
        tokenized_examples["supporting_fact_labels"].append(sf_labels)
        # -------------------------------------------------------------
        
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                            -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                          -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                                     end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions


# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch 
        from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
        
        # Safely extract label tensors if present
        start_positions = inputs.get("start_positions")
        end_positions = inputs.get("end_positions")
        sf_labels = inputs.get("supporting_fact_labels", None)
        
        # Pop labels so they're not passed twice to model
        if start_positions is not None and end_positions is not None:
            inputs.pop("start_positions")
            inputs.pop("end_positions")
            if sf_labels is not None and "supporting_fact_labels" in inputs:
                inputs.pop("supporting_fact_labels")
        
        # Forward pass: model returns start_logits, end_logits, sf_logits
        outputs = model(**inputs)
        start_logits, end_logits, sf_logits = outputs

        # -------------------------------------------------------------
        # 🔹 Training Mode — compute full joint loss
        # -------------------------------------------------------------
        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            qa_loss = (start_loss + end_loss) / 2
            
            # Supporting Fact Binary Cross-Entropy Loss
            sf_loss = torch.tensor(0.0).to(qa_loss.device)
            if sf_labels is not None:
                bce_fct = BCEWithLogitsLoss()
                attention_mask = inputs['attention_mask'].float()
                sf_labels = sf_labels.float()
                
                sf_logits_masked = sf_logits.view(-1).masked_select(attention_mask.view(-1).bool())
                sf_labels_masked = sf_labels.view(-1).masked_select(attention_mask.view(-1).bool())
                
                if sf_labels_masked.numel() > 0:
                    sf_loss = bce_fct(sf_logits_masked, sf_labels_masked)
            
            # Combine losses
            lambda_sf = 0.5
            total_loss = qa_loss + lambda_sf * sf_loss
            return (total_loss, outputs) if return_outputs else total_loss

        # -------------------------------------------------------------
        # 🔹 Evaluation Mode — return dummy loss to avoid NoneType error
        # -------------------------------------------------------------
        dummy_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        return (dummy_loss, outputs)


    

    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits, end_logits, sf_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # Forces labels to be included for custom metric calculation, 
                # but relies on compute_loss to return outputs correctly.
                prediction_loss_only=False, 
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            
            # Extract all three prediction elements
            # output.predictions should be the raw 3-tuple of logits from compute_loss
            
            #all_start_logits, all_end_logits, all_sf_logits = output.predictions
            preds = output.predictions

            if isinstance(preds, tuple):
                if len(preds) == 3:
                    all_start_logits, all_end_logits, all_sf_logits = preds
                elif len(preds) == 2:
                    all_start_logits, all_end_logits = preds
                    # create dummy sf_logits of same shape as end_logits
                    import numpy as np
                    all_sf_logits = np.zeros_like(all_end_logits)
                else:
                    raise ValueError(f"Unexpected number of prediction tensors: {len(preds)}")
            else:
                raise ValueError("Expected tuple predictions but got non-tuple output")

            # --- QA Metrics Calculation (Existing Logic) ---
            # 1. Post-process the QA predictions (only the first two elements)
            qa_predictions = (all_start_logits, all_end_logits)
            
            eval_preds = postprocess_qa_predictions(eval_examples,
                                                    eval_dataset,
                                                    qa_predictions)
            formatted_predictions = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples]

            # 2. Compute QA metrics (EM/F1)
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions,
                               label_ids=references)
            )

            # --- Supporting Fact Metrics Calculation (New Logic) ---
            
            # 1. Convert SF logits to binary predictions (0 or 1)
            sf_preds = (all_sf_logits > 0.0).astype(np.int32) 
            
            # 2. Get the ground truth SF labels from the feature dataset
            sf_labels = np.array(eval_dataset["supporting_fact_labels"])
            
            # 3. Compute the SF F1 metric
            sf_metrics = compute_sf_f1(sf_preds, sf_labels)
            
            # 4. Merge all metrics
            metrics.update(sf_metrics)


            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state,
                                                         self.control, metrics)
        return metrics