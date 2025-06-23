import random
import numpy as np
import torch
from collections import defaultdict
from datasets import load_dataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import json

from transformers import (
    BertTokenizer, TrainingArguments, Trainer, BertForSequenceClassification,
    DataCollatorWithPadding, EarlyStoppingCallback
)
import torch.nn.functional as F
import wandb
import os

# ========================================
# 1) Basic Setup
# ========================================
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "bert-base-german-cased"
tokenizer = BertTokenizer.from_pretrained(model_id)

GROUP_NAME = "BERT_Regression_CV_LearningCurve_TopicWise"
os.makedirs("./ModelSave_LearningCurve", exist_ok=True)

# Load dataset
dataset = load_dataset("json", data_files="./output_data_v2.jsonl", split="train")

# Preprocessing
def preprocess_function(examples):
    scores = [int(float(o.split(":")[1].strip())) for o in examples["output"]]
    topics = []
    for inp in examples["input"]:
        if "Topic:" in inp:
            topics.append(inp.split("\n")[0].replace("Topic:", "").strip())
        else:
            topics.append("Unknown")
    enc = tokenizer(examples["input"], truncation=True, max_length=512, padding="max_length")
    enc["labels"] = scores
    enc["Topic"] = topics
    return enc

dataset = dataset.map(preprocess_function, batched=True)

print(f"Total dataset size: {len(dataset)}")
print("Columns:", dataset.column_names)

# Potentially remove extraneous columns if they exist
# e.g., dataset = dataset.remove_columns(["input","output"]) if needed

# Convert dataset to list of indices for fraction selection
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)

# ========================================
# 2) Custom Trainer for MSE Loss
# ========================================
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze(-1)
        #penalty_factor = 0.05  # tweak via hyperparameter search
        #distance_penalty = (predictions - labels).abs() * penalty_factor
        #loss = F.mse_loss(predictions, labels) + distance_penalty.mean()
        loss = F.mse_loss(predictions, labels)
        return (loss, outputs) if return_outputs else loss

# ========================================
# 3) A function to compute topic-wise strict & flexible accuracy
# ========================================
def compute_topic_accuracies(logits, labels, topics):
    """
    Returns a dict: { topic: { 'strict': X, 'flexible': Y }, ... }
    across the entire given split (no folds). We'll do it for each fold's val.
    """
    # Round predictions for strict
    preds_rounded = np.rint(logits).clip(0, 6)
    labels_int = labels.astype(int)

    # We'll accumulate counts
    # topic_correct_strict[topic] = number of correct predictions
    # topic_correct_flexible[topic] = number of within ±1 predictions
    # topic_counts[topic] = total samples in that topic
    topic_correct_strict = defaultdict(int)
    topic_correct_flexible = defaultdict(int)
    topic_counts = defaultdict(int)

    for i, topic in enumerate(topics):
        topic_counts[topic] += 1

        if preds_rounded[i] == labels_int[i]:
            topic_correct_strict[topic] += 1

        if abs(preds_rounded[i] - labels_int[i]) <= 1:
            topic_correct_flexible[topic] += 1

    # Now compute accuracies
    topic_accuracies = {}
    for t in topic_counts:
        strict_acc = topic_correct_strict[t] / topic_counts[t]
        flex_acc   = topic_correct_flexible[t] / topic_counts[t]
        topic_accuracies[t] = {
            "strict": strict_acc,
            "flexible": flex_acc
        }

    return topic_accuracies

# ========================================
# 4) The cross-validation function returning per-topic average accuracies
# ========================================
def run_kfold_regression_with_topic_metrics(
    subset_dataset,
    fraction_str,
    k_folds=5,
    epochs=10
):
    """
    Returns average per-topic strict & flexible accuracy across folds.
    We'll store results in a dict: { topic: {'strict': mean_strict, 'flexible': mean_flexible}, ... }
    """
    # Prepare accumulators
    fold_topic_accuracies_list = []  # each fold: {topic: {'strict': X, 'flexible': Y}}

    indices = list(range(len(subset_dataset)))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed_value)

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        run_name = f"Fraction_{fraction_str}_Fold_{fold+1}"
        wandb.init(
            project="MADRS-BERT-Regression_Learning_TopicWise",
            name=run_name,
            group=GROUP_NAME,
            config={
                "fraction": fraction_str,
                "fold": fold+1,
                "epochs": epochs
            },
            reinit=True,
        )

        train_data = subset_dataset.select(train_idx)
        val_data   = subset_dataset.select(val_idx)

        model = BertForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1,
            problem_type="regression"
        ).to(device)

        training_args = TrainingArguments(
            output_dir=f"./ModelSave_Regression/fold_{fold+1}",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,  
            metric_for_best_model="eval_loss", 
            greater_is_better=False,
            save_total_limit=1,
            logging_steps=10,
            lr_scheduler_type="linear",
            warmup_steps=500,
            max_grad_norm=0.5,
            report_to=["wandb"],
            run_name=run_name
        )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        trainer = RegressionTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train
        trainer.train()

        # Predict
        prediction_output = trainer.predict(val_data)
        logits = prediction_output.predictions.squeeze(-1)
        labels = prediction_output.label_ids
        topics = val_data["Topic"]

        # Compute topic-wise accuracies
        topic_accs = compute_topic_accuracies(logits, labels, np.array(topics))
        fold_topic_accuracies_list.append(topic_accs)

        wandb.finish()

    # Now average across folds
    # We'll get a final dict: { topic: {'strict': mean_strict, 'flexible': mean_flexible} }
    all_topics = set()
    for fold_dict in fold_topic_accuracies_list:
        all_topics.update(fold_dict.keys())

    final_topic_averages = {}
    for t in all_topics:
        strict_vals = []
        flex_vals   = []
        for fold_dict in fold_topic_accuracies_list:
            if t in fold_dict:
                strict_vals.append(fold_dict[t]["strict"])
                flex_vals.append(fold_dict[t]["flexible"])
        if len(strict_vals) > 0:
            mean_strict = np.mean(strict_vals)
            mean_flex   = np.mean(flex_vals)
            final_topic_averages[t] = {
                "strict": mean_strict,
                "flexible": mean_flex
            }
    return final_topic_averages

# =========================================================
# 5)  Nested-subset learning curve  (per-topic accuracies)
# =========================================================
fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,0.8]   # always ≤0.8 for a 5-fold split
results_by_fraction = []          # list of dicts for later plotting

# We'll define the 9 main topics (or however many you want to track)
topic_translations = {
    "Traurigkeit": "1) Reported sadness",
    "Anspannung": "2) Inner tension",
    "Schlaf": "3) Sleep disturbances",
    "Appetit": "4) Loss of appetite",
    "Konzentration": "5) Difficulties concentrating",
    "Antriebslosigkeit": "6) Lassitude",
    "Gefühlslosigkeit": "7) Emotional numbness",
    "Gedanken": "8) Pessimistic thoughts",
    "Suizid": "9) Suicidal ideations"
}
topic_order = list(topic_translations.keys())

kf_outer = KFold(n_splits=5, shuffle=True, random_state=seed_value)

for outer_fold, (train_full_idx, val_idx) in enumerate(kf_outer.split(dataset)):
    print(f"\n=== Outer CV fold {outer_fold+1}/5 "
          f"(train = {len(train_full_idx)}, val = {len(val_idx)}) ===")

    # fixed validation set for this fold
    val_dataset = dataset.select(val_idx)
    val_topics  = np.array(val_dataset["Topic"])

    # --- shuffle once inside the training pool so that
    #     nested prefixes give us the required fractions ----------
    rng = np.random.default_rng(seed_value)          # reproducible
    rng.shuffle(train_full_idx)
    train_full_dataset = dataset.select(train_full_idx)

    # cache predictions for this fold so we don’t recompute val encodings
    val_encodings = None      # will be set first time we evaluate

    for frac in fractions:
        sub_size = int(len(dataset) * frac)
        sub_indices = train_full_idx[:sub_size]
        train_subset = dataset.select(sub_indices)

        frac_label = f"{int(frac*100)}%"
        run_name   = f"Fold{outer_fold+1}_{frac_label}"

        # ---------------- W&B run ----------------
        wandb.init(
            project="MADRS-BERT-Regression_Learning_TopicWise",
            name   = run_name,
            group  = GROUP_NAME,
            config = {"outer_fold": outer_fold+1,
                      "fraction"  : frac,
                      "epochs"    : 10},
            reinit=True,
        )

        # fresh model
        model = BertForSequenceClassification.from_pretrained(
            model_id, num_labels=1, problem_type="regression").to(device)

        trainer = RegressionTrainer(
            model           = model,
            args            = TrainingArguments(
                output_dir              = f"./tmp/f{outer_fold+1}_{frac_label}",
                num_train_epochs        = 10,
                per_device_train_batch_size = 4,
                per_device_eval_batch_size  = 4,
                evaluation_strategy      = "epoch",
                save_strategy            = "no",
                learning_rate            = 2e-5,
                weight_decay             = 0.01,
                logging_steps            = 20,
                report_to               = ["wandb"],
            ),
            train_dataset   = train_subset,
            eval_dataset    = val_dataset,
            data_collator   = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            tokenizer       = tokenizer,
        )

        trainer.train()

        # ---------- evaluate on the *fixed* val set ----------
        pred = trainer.predict(val_dataset)
        logits = pred.predictions.squeeze(-1)
        labels = pred.label_ids

        topic_accs = compute_topic_accuracies(logits, labels, val_topics)

        # store
        for t, acc in topic_accs.items():
            results_by_fraction.append({
                "outer_fold" : outer_fold+1,
                "fraction"   : frac,
                "topic"      : t,
                "strict_acc" : acc["strict"],
                "flexible_acc": acc["flexible"]
            })

        wandb.finish()


# =========================================================
# 6)  Save Results for Plot
# =========================================================
import pandas as pd
df = pd.DataFrame(results_by_fraction)

# -------- plotting code unchanged except use 'agg' ----------
from scipy import stats
agg = (df
       .groupby(["fraction", "topic"])
       .agg(strict_acc   = ("strict_acc",   "mean"),
            flexible_acc = ("flexible_acc", "mean"),
            strict_sem   = ("strict_acc",   lambda x: stats.sem(x, ddof=0)),
            flexible_sem = ("flexible_acc", lambda x: stats.sem(x, ddof=0)))
       .reset_index())

# Save raw and aggregated results
df.to_csv("./ModelSave_LearningCurve/results_by_fraction.csv", index=False)
agg.to_csv("./ModelSave_LearningCurve/aggregated_results.csv", index=False)

