'''
Author: Samantha Weber (samantha.weber@bli.uzh.ch)
Project: MULTICAST - University of Zurich & University Hospital of Psychiatry Zurich

Description:
This script evaluates a **BERT-based model** for MADRS (Montgomery-Åsberg Depression Rating Scale) 
score classification based on transcribed clinical interviews. The workflow includes **data preprocessing, 
model evaluation, and visualization of confusion matrices** for both strict and flexible scoring.

Key functionalities:
1️. **Load and preprocess data**:
   - Reads a dataset from a `.jsonl` file containing transcribed MADRS interviews.
   - Tokenizes the text using `BERT-base-german-cased` tokenizer.
   - Extracts **MADRS scores** and **topic labels** for each transcription.

2️. **Model Evaluation**:
   - Uses a pre-trained `BERTForSequenceClassification` model to **predict MADRS scores**.
   - Implements strict and flexible scoring:
     - **Strict evaluation**: Exact match required between predicted and true scores.
     - **Flexible evaluation**: Predictions within **±1 score** of the true score are considered correct.
   - Computes key performance metrics including:
     - **Accuracy**
     - **Mean Absolute Error (MAE)**

3️. **Confusion Matrix Visualization**:
   - Generates confusion matrices for **strict** and **flexible** score evaluations.
   - Creates **3×3 grid plots** for each MADRS topic.
   - Saves high-resolution `.tiff` images for reporting.

4️. **Per-Topic Evaluation**:
   - Evaluates model performance **separately for each MADRS topic**.
   - Saves per-topic evaluation metrics in a `.txt` file.
   - Generates and saves **per-topic confusion matrices**.

5️. **Export Evaluation Results**:
   - Saves overall and per-topic evaluation metrics to **text files**.
   - Stores **strict and flexible confusion matrices** in a `.json` file.

Output files:
- **`evaluation_results.txt`** → Overall model performance.
- **`per_topic_evaluation.txt`** → Performance metrics for each topic.
- **`Confusion_Matrices_Strict_3x3.tiff`** → Strict evaluation confusion matrices.
- **`Confusion_Matrices_Flexible_3x3.tiff`** → Flexible evaluation confusion matrices.
- **`confusion_matrices.json`** → JSON file storing all confusion matrices.

This script provides a structured **benchmarking pipeline** for evaluating **BERT-based models** in clinical 
MADRS assessment, supporting model development and refinement.
'''

# Import libraries and modules
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import wandb
from collections import defaultdict
from datasets import load_dataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from transformers import (
    BertTokenizer, TrainingArguments, Trainer, BertForSequenceClassification,
    DataCollatorWithPadding
)
from sklearn.metrics import mean_absolute_error
import json

wandb.login(key="YOUR_KEY_FOR_WANDB")

###############################################
# 1) Basic Setup
###############################################
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "bert-base-german-cased"
tokenizer = BertTokenizer.from_pretrained(model_id)

# Load dataset
dataset = load_dataset(
    "json",
    data_files="./output_data_v2.jsonl",
    split="train"
)

def preprocess_function(examples):
    # Convert output scores to integer 0..6
    scores = [int(float(o.split(":")[1].strip())) for o in examples["output"]]
    # Extract or sanitize the topic
    topics = []
    for inp in examples["input"]:
        if "Topic:" in inp:
            topics.append(inp.split("\n")[0].replace("Topic:", "").strip())
        else:
            topics.append("Unknown")
    
    # Tokenize
    enc = tokenizer(
        examples["input"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    
    enc["labels"] = scores
    enc["Topic"] = topics
    return enc

dataset = dataset.map(preprocess_function, batched=True)
print("Full dataset size:", len(dataset))
print("Columns:", dataset.column_names)

# --------------------------
# 3) Zero-Shot Regression Trainer
# --------------------------
class RegressionTrainer(Trainer):
    """
    MSE-based zero-shot: no training, just evaluation.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        # We won't train, but if you call trainer.train(), it would do MSE
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = F.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def regression_metrics(eval_pred):
    """
    Evaluate MSE, MAE, strict & tolerant accuracy for zero-shot predictions
    """
    logits, labels = eval_pred
    if logits.ndim == 2:
        logits = logits.squeeze(-1)
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # MSE, MAE
    mse = np.mean((logits - labels)**2)
    mae = mean_absolute_error(labels, logits)

    # Strict & Tolerant accuracy
    preds_rounded = np.rint(logits).clip(0, 6)
    labels_int = labels.astype(int)

    strict_acc = (preds_rounded == labels_int).sum() / len(labels_int)
    diffs = np.abs(preds_rounded - labels_int)
    tolerant_acc = (diffs <= 1).sum() / len(labels_int)

    return {
        "mse": mse,
        "mae": mae,
        "strict_accuracy": strict_acc,
        "tolerant_accuracy": tolerant_acc
    }

###############################################
# 1b) Compute Baseline MSE and MAE Per Topic
###############################################
# 1) Gather labels by topic
topic_scores = defaultdict(list)

for i in range(len(dataset)):
    topic = dataset["Topic"][i]
    label = dataset["labels"][i]
    topic_scores[topic].append(label)

# 2) Compute mean for each topic
topic_mean_scores = {
    topic: np.mean(scores) 
    for topic, scores in topic_scores.items()
}

# 3) Per-topic MSE, MAE, and overall arrays of predictions & labels
baseline_metrics = {}
all_true = []
all_pred = []

for topic, mean_pred in topic_mean_scores.items():
    true_scores = np.array(topic_scores[topic])
    # Each data point in this topic is assigned the same predicted mean
    pred_scores = np.full_like(true_scores, mean_pred, dtype=float)
    
    # Per-topic MSE, MAE
    mse = mean_squared_error(true_scores, pred_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    
    baseline_metrics[topic] = {
        "mean_prediction": float(mean_pred),
        "mse": float(mse),
        "mae": float(mae)
    }
    
    # Accumulate for overall error calculations
    all_true.extend(true_scores)
    all_pred.extend(pred_scores)

all_true = np.array(all_true, dtype=float)
all_pred = np.array(all_pred, dtype=float)

# 4) Compute overall MSE, MAE, and the standard deviation of errors
overall_mse = mean_squared_error(all_true, all_pred)
overall_mae = mean_absolute_error(all_true, all_pred)
# Standard deviation of the errors
residuals = all_pred - all_true
overall_std_of_errors = float(np.std(residuals))

# 5) Print baseline results per topic
print("\n=== Baseline MSE and MAE Per Topic (Mean Predictor) ===")
for topic, metrics in baseline_metrics.items():
    print(
        f"{topic}: "
        f"Mean Score={metrics['mean_prediction']:.2f}, "
        f"MSE={metrics['mse']:.4f}, "
        f"MAE={metrics['mae']:.4f}"
    )

# 6) Print overall results
print("\n=== Overall Baseline Results (All Topics Combined) ===")
print(f"Overall MSE: {overall_mse:.4f}")
print(f"Overall MAE: {overall_mae:.4f}")
print(f"Std of Errors: {overall_std_of_errors:.4f}")

# 7) Save results to a text file
output_file = "./ZeroShot_Eval/baseline_per_topic_metrics.txt"
with open(output_file, "w") as f:
    f.write("=== Baseline MSE and MAE Per Topic (Mean Predictor) ===\n\n")
    for topic, metrics in baseline_metrics.items():
        f.write(
            f"{topic}: "
            f"Mean Score={metrics['mean_prediction']:.2f}, "
            f"MSE={metrics['mse']:.4f}, "
            f"MAE={metrics['mae']:.4f}\n"
        )
    f.write("\n=== Overall Baseline Results (All Topics Combined) ===\n")
    f.write(f"Overall MSE: {overall_mse:.4f}\n")
    f.write(f"Overall MAE: {overall_mae:.4f}\n")
    f.write(f"Std of Errors: {overall_std_of_errors:.4f}\n")

print(f"\nBaseline metrics saved to {output_file}")

# --------------------------
# 4) K-Fold Zero-Shot Eval
# --------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

fold_metrics = defaultdict(list)
topic_wise_metrics = defaultdict(lambda: defaultdict(list))

all_predictions_strict = []
all_predictions_tolerant = []
all_labels = []
all_topics = []

GROUP_NAME = "ZeroShot_BERT_Regression"

indices = list(range(len(dataset)))

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n=== Zero-Shot Eval Fold {fold+1}/{n_splits} ===\n")

    run_name = f"ZeroShot_fold_{fold+1}"
    wandb.init(
        project="MADRS-BERT-ZeroShot",
        name=run_name,
        group=GROUP_NAME,
        config={"fold": fold+1},
        reinit=True,
    )

    # For zero-shot, we do not train, just evaluate on val set
    val_data = dataset.select(val_idx)

    # Load base BERT model in regression mode
    model = BertForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        problem_type="regression"
    ).to(device)

    # We won't train, so training_args has num_train_epochs=0
    training_args = TrainingArguments(
        output_dir=f"./ZeroShot_Eval/fold_{fold+1}",
        num_train_epochs=0,  # no training
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=0.0,  # not used
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to=["wandb"],
        run_name=run_name,
        remove_unused_columns=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=regression_metrics
    )

    # Zero-shot: we skip trainer.train() -> no fine-tuning
    # Evaluate directly
    eval_results = trainer.evaluate(val_data)
    print(f"Fold {fold+1} Results:", eval_results)

    # Store fold-level metrics
    for key, value in eval_results.items():
        if key.startswith("eval_"):
            fold_metrics[key].append(value)

    # Collect predictions
    prediction_output = trainer.predict(val_data)
    logits = prediction_output.predictions.squeeze(-1)
    labels = prediction_output.label_ids
    preds_strict = np.rint(logits).clip(0, 6)

    # Tolerant predictions
    preds_tolerant = preds_strict.copy()
    for i in range(len(preds_strict)):
        if abs(preds_strict[i] - labels[i]) <= 1:
            preds_tolerant[i] = labels[i]

    # Extend final arrays
    all_predictions_strict.extend(preds_strict.tolist())
    all_predictions_tolerant.extend(preds_tolerant.tolist())
    all_labels.extend(labels.tolist())
    all_topics.extend(val_data["Topic"])
    
    # Per-topic metrics
    unique_topics = set(val_data["Topic"])
    for topic in unique_topics:
        topic_mask = np.array(val_data["Topic"]) == topic
        topic_logits = logits[topic_mask]
        topic_labels = labels[topic_mask]

        topic_eval = regression_metrics((topic_logits, topic_labels))
        for k, v in topic_eval.items():
            topic_wise_metrics[topic][k].append(v)

    wandb.finish()

# ------------------------------------------------
# Aggregate Cross-Fold Results (for base model)
# ------------------------------------------------
final_metrics = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in fold_metrics.items()}

final_topic_metrics = {}
for topic, metric_dict in topic_wise_metrics.items():
    final_topic_metrics[topic] = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in metric_dict.items()}

print("\n=== Final Cross-Validation Metrics (Overall, Base Model) ===")
for metric, stats in final_metrics.items():
    print(f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

print("\n=== Final Cross-Validation Metrics (Per-Topic, Base Model) ===")
for topic, metrics_dict in final_topic_metrics.items():
    print(f"\nTopic: {topic}")
    for metric, stats in metrics_dict.items():
        print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

with open("./ZeroShot_Eval/base_per_topic_evaluation.txt", "w") as f:
    for topic, metrics in final_topic_metrics.items():
        f.write(f"Metrics for Topic: {topic}\n")
        for metric, stats in metrics.items():
            f.write(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}\n")
        f.write("\n")

# -----------------------------------
# 7) Save Predictions for Future Comparison
# -----------------------------------
prediction_data = {
    "labels": all_labels,
    "strict_predictions": all_predictions_strict,
    "flexible_predictions": all_predictions_tolerant,
    "topics": all_topics
}

# Save to JSON
with open("./ZeroShot_Eval/base_model_predictions.json", "w") as f:
    json.dump(prediction_data, f, indent=4)

print("Base model predictions saved to './ZeroShot_Eval/base_model_predictions.json'")

# -----------------------------------
# PLOT Confusion Matrices for Base Model (Strict & Flexible)
# -----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

final_labels = np.array(all_labels)
final_predictions_strict = np.array(all_predictions_strict)
final_predictions_flexible = np.array(all_predictions_tolerant)
final_topics = all_topics

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
unique_topics = [t for t in topic_order if t in final_topics]

confusion_matrices = {}
topic_wise_labels_dict = {topic: [] for topic in unique_topics}
topic_wise_predictions_strict_dict = {topic: [] for topic in unique_topics}
topic_wise_predictions_flexible_dict = {topic: [] for topic in unique_topics}

for label, pred_strict, pred_flexible, topic in zip(final_labels, final_predictions_strict, final_predictions_flexible, final_topics):
    if topic in topic_wise_labels_dict:
        topic_wise_labels_dict[topic].append(label)
        topic_wise_predictions_strict_dict[topic].append(pred_strict)
        topic_wise_predictions_flexible_dict[topic].append(pred_flexible)

# Plot Strict Confusion Matrices
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for ax, topic in zip(axes, unique_topics):
    if len(topic_wise_labels_dict[topic]) > 0:
        cm_strict = confusion_matrix(topic_wise_labels_dict[topic], topic_wise_predictions_strict_dict[topic], labels=list(range(7)))
        confusion_matrices[topic] = {"strict": cm_strict.tolist()}
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_strict, display_labels=list(range(7)))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(topic_translations[topic], fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted MADRS Score', fontsize=10)
        ax.set_ylabel('True MADRS Score', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
for ax in axes[len(unique_topics):]:
    fig.delaxes(ax)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle(r"$\bf{Confusion\ Matrices\ for\ Base\ Model\ (Strict)}$", fontsize=16)
plt.savefig("./ZeroShot_Eval/Base_Confusion_Matrices_Strict.tiff", format="tiff", dpi=300)
plt.close()

# Plot Flexible Confusion Matrices
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for ax, topic in zip(axes, unique_topics):
    if len(topic_wise_labels_dict[topic]) > 0:
        cm_flexible = confusion_matrix(topic_wise_labels_dict[topic], topic_wise_predictions_flexible_dict[topic], labels=list(range(7)))
        confusion_matrices[topic]["flexible"] = cm_flexible.tolist()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_flexible, display_labels=list(range(7)))
        disp.plot(ax=ax, cmap="Greens", colorbar=False)
        ax.set_title(topic_translations[topic], fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted MADRS Score', fontsize=10)
        ax.set_ylabel('True MADRS Score', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)
for ax in axes[len(unique_topics):]:
    fig.delaxes(ax)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle(r"$\bf{Confusion\ Matrices\ for\ Base\ Model\ (Flexible)}$", fontsize=16)
plt.savefig("./ZeroShot_Eval/Base_Confusion_Matrices_Flexible.tiff", format="tiff", dpi=300)
plt.close()

with open("./ZeroShot_Eval/Base_confusion_matrices.json", "w") as f:
    json.dump(confusion_matrices, f, indent=4)

print("Final per-topic confusion matrices for base model saved.")