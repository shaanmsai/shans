import torch
import numpy as np
import json
import argparse
from transformers import AutoTokenizer, ElectraConfig, ElectraForQuestionAnswering, ElectraPreTrainedModel
from torch.nn import Linear
from typing import Tuple, Dict, Any

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
MODEL_PATH = "/content/drive/MyDrive/MSAI/NLP/Assignments/FinalProject2/outputs/hotpot_electra_dual_head"
QA_MAX_ANSWER_LENGTH = 30


# -------------------------------------------------------------
# DUAL-HEAD MODEL (same as in training)
# -------------------------------------------------------------
class ElectraForDualHeadQA(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        base_electra = ElectraForQuestionAnswering.from_pretrained(config.name_or_path, config=config)
        self.electra = base_electra.electra
        self.num_labels = config.num_labels

        self.qa_outputs = Linear(config.hidden_size, config.num_labels)
        self.sf_outputs = Linear(config.hidden_size, 1)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        sf_logits = self.sf_outputs(sequence_output).squeeze(-1).contiguous()

        # Return as tuple (but HF may wrap/extend this with extras)
        return (start_logits, end_logits, sf_logits)


# -------------------------------------------------------------
# PREPROCESS FOR INFERENCE
# -------------------------------------------------------------
def prepare_inference_dataset(question: str, context: str, tokenizer):
    tokenized_examples = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=128,
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # map features to a single dummy example id
    tokenized_examples["example_id"] = []
    for i in range(len(tokenized_examples["input_ids"])):
        tokenized_examples["example_id"].append("dummy_id")
        sequence_ids = tokenized_examples.sequence_ids(i)
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    raw_example = {"id": "dummy_id", "context": context, "question": question, "answers": {"text": [], "answer_start": []}}
    return raw_example, tokenized_examples


# -------------------------------------------------------------
# ROBUST PREDICTIONS UNWRAPPER
# -------------------------------------------------------------
def unwrap_start_end_preds(preds: Any):
    """
    Accept many shapes/types and return (all_start_logits, all_end_logits)
    where each is a numpy array of shape (num_features, seq_len).
    """
    # If it's a HF ModelOutput (dict-like)
    if hasattr(preds, "get") and callable(getattr(preds, "get")):
        # ModelOutput behaves like dict
        start = preds.get("start_logits", None)
        end = preds.get("end_logits", None)
        if start is None or end is None:
            # Sometimes HF names differ; try positional extraction below
            pass
        else:
            return np.asarray(start), np.asarray(end)

    # If it's a tuple/list-like
    if isinstance(preds, (tuple, list)):
        if len(preds) >= 2:
            # Take first two elements
            start = np.asarray(preds[0])
            end = np.asarray(preds[1])
            return start, end
        else:
            raise ValueError(f"Predictions tuple/list has length {len(preds)} — expected >=2")

    # If it's a numpy array
    if isinstance(preds, np.ndarray):
        if preds.ndim >= 3:
            # assume axis 0 indexes different heads: [head, features, seq_len]
            start = preds[0]
            end = preds[1]
            return np.asarray(start), np.asarray(end)
        elif preds.ndim == 2:
            # ambiguous: maybe concatenated logits? Not expected.
            raise ValueError("Predictions is a 2D numpy array — cannot reliably split into start/end logits.")
        else:
            raise ValueError(f"Unsupported numpy prediction shape: {preds.shape}")

    # Torch tensor
    if isinstance(preds, torch.Tensor):
        arr = preds.cpu().numpy()
        return unwrap_start_end_preds(arr)

    # Fallback: try to coerce to tuple
    try:
        seq = tuple(preds)
        if len(seq) >= 2:
            return np.asarray(seq[0]), np.asarray(seq[1])
    except Exception:
        pass

    raise TypeError(f"Unrecognized prediction type: {type(preds)}")


# -------------------------------------------------------------
# POSTPROCESS FOR ANSWER EXTRACTION (robust)
# -------------------------------------------------------------
def postprocess_inference_predictions(raw_example: dict, features: Dict, predictions: Any, n_best_size: int = 20):
    # Use the robust unwrapper to get start/end arrays
    try:
        all_start_logits, all_end_logits = unwrap_start_end_preds(predictions)
    except Exception as e:
        # Provide informative debug info
        raise RuntimeError(f"Failed to unwrap predictions for postprocessing: {e}. "
                           f"Prediction type: {type(predictions)}. "
                           f"Features input_ids len: {len(features.get('input_ids', []))}")

    prelim_predictions = []

    num_features = len(features["input_ids"])
    for feature_index in range(num_features):
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]
        offset_mapping = features["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
        end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                    or end_index < start_index
                    or end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH
                ):
                    continue

                prelim_predictions.append({
                    "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                    "score": start_logits[start_index] + end_logits[end_index]
                })

    predictions_sorted = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

    if not predictions_sorted:
        return "No answer found"

    best_pred = predictions_sorted[0]
    start_offset, end_offset = best_pred["offsets"]
    predicted_answer = raw_example["context"][start_offset:end_offset]
    return predicted_answer


# -------------------------------------------------------------
# RUN INFERENCE (main)
# -------------------------------------------------------------
def run_model_inference(question: str, context: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model + tokenizer
    try:
        config = ElectraConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        config.num_labels = 2
        config.name_or_path = "google/electra-small-discriminator"

        model = ElectraForDualHeadQA.from_pretrained(MODEL_PATH, config=config, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
        model.to(device)
        model.eval()
    except Exception as e:
        return {"error": f"Failed to load model/tokenizer from {MODEL_PATH}: {e}"}

    raw_example, features = prepare_inference_dataset(question, context, tokenizer)

    try:
        with torch.no_grad():
            # build tensors (features is HF tokenized output)
            input_ids = torch.tensor(features["input_ids"]).to(device)           # shape: (num_features, seq_len)
            attention_mask = torch.tensor(features["attention_mask"]).to(device)
            token_type_ids = torch.tensor(features["token_type_ids"]).to(device)

            # The model expects batch dimension: shape (batch=num_features, seq_len)
            # Our model forward handles batch processing. Pass them directly.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            # outputs could be: tuple (start,end,sf, ...), ModelOutput, or longer tuple
            # We only need start/end (and optionally sf)
            # Normalize to numpy arrays for postprocessing
            # Best approach: try to extract start/end directly, else fallback to tuple slicing
            start_logits = end_logits = sf_logits = None

            # If it's a tuple/list-like
            if isinstance(outputs, (tuple, list)):
                if len(outputs) >= 3:
                    start_logits_t = outputs[0]
                    end_logits_t = outputs[1]
                    sf_logits_t = outputs[2]
                elif len(outputs) == 2:
                    start_logits_t = outputs[0]
                    end_logits_t = outputs[1]
                    sf_logits_t = torch.zeros_like(end_logits_t)
                else:
                    # unexpected - try to coerce
                    seq = tuple(outputs)
                    if len(seq) >= 2:
                        start_logits_t, end_logits_t = seq[0], seq[1]
                        sf_logits_t = torch.zeros_like(end_logits_t)
                    else:
                        raise RuntimeError(f"Model returned unexpected tuple length: {len(outputs)}")
            elif hasattr(outputs, "get") and callable(getattr(outputs, "get")):
                # ModelOutput / dict-like
                start_logits_t = outputs.get("start_logits", None)
                end_logits_t = outputs.get("end_logits", None)
                sf_logits_t = outputs.get("sf_logits", None)
                # If ModelOutput returned only start/end inside 'logits' or similar, try positional fallback
                if start_logits_t is None or end_logits_t is None:
                    # fallback: try to coerce to tuple
                    try:
                        seq = tuple(outputs)
                        if len(seq) >= 2:
                            start_logits_t, end_logits_t = seq[0], seq[1]
                            sf_logits_t = seq[2] if len(seq) >= 3 else torch.zeros_like(end_logits_t)
                    except Exception:
                        raise RuntimeError("ModelOutput missing start_logits/end_logits and cannot fallback to tuple.")
            elif isinstance(outputs, torch.Tensor):
                raise RuntimeError(f"Model returned a single tensor of shape {outputs.shape} — cannot split to start/end.")
            else:
                # final fallback attempt
                try:
                    seq = tuple(outputs)
                    start_logits_t, end_logits_t = seq[0], seq[1]
                    sf_logits_t = seq[2] if len(seq) >= 3 else torch.zeros_like(end_logits_t)
                except Exception as e:
                    raise RuntimeError(f"Cannot interpret model outputs of type {type(outputs)}: {e}")

            # Convert torch -> numpy arrays with shape (num_features, seq_len)
            start_logits_arr = start_logits_t.detach().cpu().numpy() if isinstance(start_logits_t, torch.Tensor) else np.asarray(start_logits_t)
            end_logits_arr = end_logits_t.detach().cpu().numpy() if isinstance(end_logits_t, torch.Tensor) else np.asarray(end_logits_t)
            sf_logits_arr = sf_logits_t.detach().cpu().numpy() if isinstance(sf_logits_t, torch.Tensor) else np.asarray(sf_logits_t)

            # Post-process QA answer using robust postprocessor
            predicted_answer = postprocess_inference_predictions(raw_example, features, (start_logits_arr, end_logits_arr))

            # SF tokens (binary)
            sf_preds_binary = (sf_logits_arr > 0.0).astype(np.int32)
            # If shape matches (num_features, seq_len)
            if sf_preds_binary.ndim == 2:
                sf_tokens_first_chunk = sf_preds_binary[0].tolist()
            elif sf_preds_binary.ndim == 1:
                sf_tokens_first_chunk = sf_preds_binary.tolist()
            else:
                # flatten and take first seq slice
                sf_tokens_first_chunk = sf_preds_binary.reshape(sf_preds_binary.shape[0], -1)[0].tolist()

            # Optional: map predicted SF token indices to words for the first chunk
            sf_words = []
            ids_first_feature = features["input_ids"][0]
            for i, tok_id in enumerate(ids_first_feature):
                if i < len(sf_tokens_first_chunk) and sf_tokens_first_chunk[i] == 1:
                    sf_words.append(tokenizer.convert_ids_to_tokens(int(tok_id)))

            return {
                "answer": predicted_answer,
                "sf_tokens": sf_tokens_first_chunk,
                "sf_words": sf_words,
                "context": context
            }

    except Exception as e:
        # Include extra debugging info for easier diagnosis
        return {"error": f"Inference failed during prediction: {e}"}


# -------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dual-Head QA Inference.")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the model.")
    parser.add_argument("--context", type=str, required=True, help="Context passage to search for the answer.")
    args = parser.parse_args()

    print(f"--- Running Inference on Trained Model at: {MODEL_PATH} ---")
    results = run_model_inference(args.question, args.context)

    if "error" in results:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"\nQuestion: {args.question}")
        print(f"Context: {args.context[:200]}...")
        print("-" * 60)
        print(f"🧩 Predicted Answer: {results['answer']}")
        print(f"✨ Supporting Fact Tokens (first chunk): {sum(results['sf_tokens'])} positive tokens")
        print(f"🔍 Supporting Fact Words (sample): {results['sf_words'][:50]}")
