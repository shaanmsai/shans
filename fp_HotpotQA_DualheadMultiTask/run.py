from transformers import ElectraForQuestionAnswering, ElectraPreTrainedModel
import torch
import torch.nn as nn
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    #Shankar Part 2 Fix Modifying run.py: Model and Metrics Setup
    ElectraForTokenClassification, # We'll use a token classification head for sentence prediction
    # END ADD
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
import evaluate
from helpers import (
    prepare_dataset_nli,
    prepare_train_dataset_qa,
    prepare_validation_dataset_qa,
    QuestionAnsweringTrainer,
    compute_accuracy,
)
import os
import json

NUM_PREPROCESSING_WORKERS = 2

# model with a second head for Supporting Fact (SF) prediction
class ElectraForDualHeadQA(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Base ELECTRA Encoder
        self.electra = ElectraForQuestionAnswering.from_pretrained(config.name_or_path, config=config).electra
        self.num_labels = config.num_labels  # QA labels (start/end)

        # Head 1: QA Head (Start/End Logits) - Copied from ElectraForQuestionAnswering
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Head 2: Supporting Fact Prediction Head (Binary Classification for each token)
        self.sf_outputs = nn.Linear(config.hidden_size, 1)

        self.post_init()

    # The forward pass now returns both QA logits and SF logits
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                start_positions=None, end_positions=None, supporting_fact_labels=None, **kwargs):
        
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        # QA Logits (Start/End)
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # SF Logits (Token Classification)
        sf_logits = self.sf_outputs(sequence_output).squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            return (start_logits, end_logits, sf_logits)

        # Return predictions during evaluation
        return (start_logits, end_logits, sf_logits)
        
# -------------------------------------------------------------
# Utility: adapt HotpotQA to SQuAD-style format
# -------------------------------------------------------------
def adapt_hotpotqa_format(dataset):
    """Convert HotpotQA examples to the expected SQuAD-style format used in helpers.py"""
    def _transform(example):
        # Merge all paragraphs into one context string
        if isinstance(example["context"], list):
            merged_context = " ".join([p[1] for p in example["context"]])
        else:
            merged_context = str(example["context"])

        # Convert answer field to SQuAD-style 'answers' dict
        ans_text = example.get("answer", "")
        start_char = merged_context.find(ans_text)
        if start_char == -1:
            start_char = 0
        example["context"] = merged_context
        example["answers"] = {"text": [ans_text], "answer_start": [start_char]}
        return example

    print("🔁 Adapting HotpotQA dataset to SQuAD-style format...")
    dataset = dataset.map(_transform)
    return dataset


# -------------------------------------------------------------
# 🚀 Main
# -------------------------------------------------------------
def main():
    argp = HfArgumentParser(TrainingArguments)
    argp.add_argument("--model", type=str, default="google/electra-small-discriminator",
                      help="Base model to fine-tune (HuggingFace model ID or local checkpoint)")
    argp.add_argument("--task", type=str, choices=["nli", "qa"], required=True,
                      help="Specify task type: 'nli' or 'qa'")
    argp.add_argument("--dataset", type=str, default=None,
                      help="Dataset name, e.g., 'hotpot_qa:distractor' or 'squad'")
    argp.add_argument("--max_length", type=int, default=128,
                      help="Maximum sequence length for tokenization")
    argp.add_argument("--max_train_samples", type=int, default=None,
                      help="Limit number of training examples")
    argp.add_argument("--max_eval_samples", type=int, default=None,
                      help="Limit number of eval examples")

    training_args, args = argp.parse_args_into_dataclasses()

    # -------------------------------------------------------------
    # Dataset loading
    # -------------------------------------------------------------
    if args.dataset and (args.dataset.endswith(".json") or args.dataset.endswith(".jsonl")):
        dataset = datasets.load_dataset("json", data_files=args.dataset)
        eval_split = "train"
    else:
        default_datasets = {"qa": ("squad",), "nli": ("snli",)}
        dataset_id = tuple(args.dataset.split(":")) if args.dataset else default_datasets[args.task]
        eval_split = "validation_matched" if dataset_id == ("glue", "mnli") else "validation"
        dataset = datasets.load_dataset(*dataset_id)

    # Adapt HotpotQA dataset if selected
    if args.dataset and args.dataset.startswith("hotpot_qa"):
        dataset = adapt_hotpotqa_format(dataset)

    # -------------------------------------------------------------
    # Model and tokenizer setup
    # -------------------------------------------------------------
    task_kwargs = {"num_labels": 3} if args.task == "nli" else {}
    #model_classes = {"qa": AutoModelForQuestionAnswering, "nli": AutoModelForSequenceClassification}
    model_classes = {"qa": ElectraForDualHeadQA, "nli": AutoModelForSequenceClassification} # Change AutoModelForQuestionAnswering to ElectraForDualHeadQA
    model = model_classes[args.task].from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    
    # -------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------
    if args.task == "qa":
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    else:
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)

    if args.task == "nli" and "label" in dataset["train"].column_names:
        dataset = dataset.filter(lambda ex: ex["label"] != -1)

    # Prepare training and eval features
    train_dataset = eval_dataset = None
    train_feat = eval_feat = None
    if training_args.do_train:
        train_dataset = dataset["train"]
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_feat = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names,
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_feat = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=False
        )

    # -------------------------------------------------------------
    # Metrics & Trainer setup
    # -------------------------------------------------------------
    trainer_class = Trainer
    eval_kwargs = {}
    compute_metrics = None

    if args.task == "qa":
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs["eval_examples"] = eval_dataset
        metric = evaluate.load("squad")
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    else:
        compute_metrics = compute_accuracy

    eval_predictions = None

    def compute_metrics_and_store(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_feat,
        eval_dataset=eval_feat,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store,
    )

    # -------------------------------------------------------------
    # Train and/or Evaluate
    # -------------------------------------------------------------
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)
        print("📊 Evaluation Results:", results)

        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "eval_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
