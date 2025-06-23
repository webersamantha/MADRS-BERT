'''
Author: Samantha Weber (samantha.weber@bli.uzh.ch)
Project: MULTICAST - University of Zurich & University Hospital of Psychiatry Zurich

Description:
This script fine-tunes a **BERT-based model** for MADRS (Montgomery-Åsberg Depression Rating Scale) 
score classification using **cross-validation and a custom loss function** that penalizes 
classification errors based on their distance from the true score. It also evaluates model performance 
with **strict and flexible scoring criteria**.

Key functionalities:
1️. **Load and preprocess data**:
   - Reads a dataset from a `.jsonl` file containing transcribed MADRS interviews.
   - Tokenizes the text using `BERT-base-german-cased` tokenizer.
   - Extracts **MADRS scores** and **topic labels** for each transcription.

2️. **Cross-validation fine-tuning**:
   - Implements **5-fold cross-validation** to train and validate the model.
   - MSE Loss Function is used

3️. **Model Evaluation**:
   - Evaluates model performance using:
     - **Strict evaluation**: Exact match required between predicted and true scores.
     - **Flexible evaluation**: Predictions within **±1 score** of the true score are considered correct.
   - Computes key performance metrics including:
     - **Accuracy**
     - **Mean Absolute Error (MAE)**

4️. **Confusion Matrix Visualization**:
   - Generates confusion matrices for **strict** and **flexible** score evaluations.
   - Creates **3×3 grid plots** for each MADRS topic.
   - Saves high-resolution `.tiff` images for reporting.

5️. **Per-Topic Evaluation**:
   - Evaluates model performance **separately for each MADRS topic**.
   - Saves per-topic evaluation metrics in a `.txt` file.
   - Generates and saves **per-topic confusion matrices**.

6️. **Export Evaluation Results**:
   - Saves overall and per-topic evaluation metrics to **text files**.
   - Stores **strict and flexible confusion matrices** in a `.json` file.

Output files:
- **`evaluation_results.txt`** → Overall model performance.
- **`per_topic_evaluation.txt`** → Performance metrics for each topic.
- **`Confusion_Matrices_Fine_tuned_Strict_3x3.tiff`** → Strict evaluation confusion matrices.
- **`Confusion_Matrices_Fine_tuned_Flexible_3x3.tiff`** → Flexible evaluation confusion matrices.
- **`confusion_matrices_fineTuned.json`** → JSON file storing all confusion matrices.

This script implements a **robust fine-tuning pipeline** for **MADRS assessment** using BERT, 
supporting model development and clinical depression severity analysis.
'''

# ---------------------------------
# 1. Imports & Setup
# ---------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertTokenizer, TrainingArguments, Trainer, BertForSequenceClassification, 
    DataCollatorWithPadding, EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import os
import numpy as np
import random
import wandb  # NEW: make sure you `pip install wandb`
import json
wandb.login(key="YOUR_KEY_FOR_WANDB")


# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make a directory to save fold outputs
os.makedirs("./ModelSave_Regression", exist_ok=True)

# ---------------------------------
# 2. Custom Trainer for MSE Loss
# ---------------------------------
class RegressionTrainer(Trainer):
    """
    Overrides the default compute_loss to use mean squared error
    on a single continuous output (num_labels=1).
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze(-1)
        #penalty_factor = 0.05  # tweak via hyperparameter search
        #distance_penalty = (predictions - labels).abs() * penalty_factor
        #loss = F.mse_loss(predictions, labels) + distance_penalty.mean()
        loss = F.mse_loss(predictions, labels)
        return (loss, outputs) if return_outputs else loss
    
class MAERegressionTrainer(Trainer):
    """
    Overrides the default compute_loss to use mean absolute error
    on a single continuous output (num_labels=1).
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        # (batch_size, 1) → (batch_size,)
        predictions = outputs.logits.squeeze(-1)
        
        # MAE instead of MSE
        loss = F.l1_loss(predictions, labels)
        
        return (loss, outputs) if return_outputs else loss

# ---------------------------------
# 3. Define Regression Metrics
# ---------------------------------
def regression_metrics(eval_pred):
    """
    Computes:
      - MSE (on continuous predictions)
      - MAE (on continuous predictions)
      - 'strict' accuracy = % predicted EXACT integer label
      - 'tolerant' accuracy = % within ±1 of the true integer label
    """
    logits, labels = eval_pred
    # logits will be (batch_size, 1); squeeze to (batch_size,)
    if logits.ndim == 2:
        logits = logits.squeeze(-1)
    
    # Convert to numpy arrays if still tensors
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Continuous predictions from the model
    continuous_preds = logits
    # Round to nearest integer for "strict" or "tolerant" checks
    preds_rounded = np.rint(continuous_preds).clip(0, 6)
    
    # Ensure labels are integer (they should be 0..6). 
    # If they were floats, we can still treat them as integers here.
    labels_int = labels.astype(int)
    
    # Strict accuracy: exact match
    strict_correct = (preds_rounded == labels_int).sum()
    strict_accuracy = strict_correct / len(labels_int)
    
    # Tolerant accuracy: within ±1
    diffs = np.abs(preds_rounded - labels_int)
    tolerant_correct = (diffs <= 1).sum()
    tolerant_accuracy = tolerant_correct / len(labels_int)
    
    # MSE (on the raw continuous predictions, not rounded)
    mse = np.mean((continuous_preds - labels)**2)
    # MAE 
    mae = mean_absolute_error(continuous_preds, labels)
    
    return {
        "mse": mse,
        "mae": mae,
        "strict_accuracy": strict_accuracy,
        "tolerant_accuracy": tolerant_accuracy
    }

# ---------------------------------
# 4. Load & Preprocess Dataset
# ---------------------------------
model_id = "./models/bert-base-german-cased"
tokenizer = BertTokenizer.from_pretrained(model_id)

# Load dataset from JSONL file
dataset = load_dataset("json", data_files="./output_data_v2.jsonl", split="train")

def preprocess_function(examples):
    # Scores: integer 0..6
    scores = [int(float(output.split(":")[1].strip())) for output in examples['output']]
    # Extract topic from the first line of `input`
    topics = [
        input_text.split("\n")[0].replace("Topic: ", "").strip() 
        if "Topic:" in input_text 
        else "Unknown" 
        for input_text in examples['input']
    ]
    
    encoded = tokenizer(
        examples['input'], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    
    # Return a dictionary with labels & topic as well
    encoded["labels"] = scores
    encoded["Topic"] = topics
    return encoded

tokenized_dataset = dataset.map(preprocess_function, batched=True)
print(f"Total samples in the dataset: {len(tokenized_dataset)}")
print("Columns:", tokenized_dataset.column_names)

# For regression, we don't need class weights, so we skip that part

# -----------------------------
# 5. Setup Weights & Biases (W&B)
# -----------------------------
# We'll group all folds under one "W&B Group" so you can see them in the same place.
group_name = "BERT_Regression_CV"  # customize as you like

# ---------------------------------
# 6. Cross-Validation Setup
# ---------------------------------
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed_value)

fold_metrics = defaultdict(list)          # store overall fold metrics
topic_wise_metrics = defaultdict(lambda: defaultdict(list))  # store per-topic

all_predictions_strict = []
all_predictions_tolerant = []
all_labels = []
all_topics = []

# Convert to list of indices
indices = list(range(len(tokenized_dataset)))

# We want to store results across folds
for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n=== Training fold {fold+1}/{n_splits} ===\n")
    
    # 6.1 W&B INIT FOR THIS FOLD
    # This creates a new run on your wandb project for each fold
    run_name = f"fold_{fold+1}"
    wandb.init(
        project="MADRS-BERT-Regression",  # CHANGE THIS to your actual W&B project
        name=run_name,
        group=group_name,
        config={
            "fold": fold+1
            # You can log other hyperparams here as well
        }
    )
    # Split dataset
    train_dataset = tokenized_dataset.select(train_idx)
    val_dataset   = tokenized_dataset.select(val_idx)

    # Create fresh BERT model in regression mode
    model = BertForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=1,               # single continuous output
        problem_type="regression"   # sets internal config for regression
    ).to(device)
    
    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"./ModelSave_Regression/fold_{fold+1}",
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5, #before e1-5
        weight_decay=0.01,#0.01 before
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=10,
        lr_scheduler_type="linear",
        warmup_steps=500,
        max_grad_norm=0.5, #before 0.5
        report_to=["wandb"],  
        run_name=run_name     
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    
    # Create our custom Trainer for MSE loss
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=regression_metrics,  # our regression metrics
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Fold {fold+1} Results:", eval_results)

    # Store metrics
    for key, value in eval_results.items():
        # e.g., 'eval_loss', 'eval_mse', 'eval_mae', ...
        fold_metrics[key].append(value)

    # ---------------------------------
    # 7a. Collect predictions for further analysis
    # ---------------------------------
    raw_preds = trainer.predict(val_dataset)
    # raw_preds.predictions shape: (num_samples, 1)
    logits = raw_preds.predictions.squeeze(-1)  # shape: (num_samples,)
    labels = raw_preds.label_ids               # shape: (num_samples,)
    
    # Round for "strict" predictions
    preds_strict = np.rint(logits).clip(0, 6)
    # Tolerant: we mark it as correct if abs diff <=1
    preds_tolerant = preds_strict.copy()
    for i in range(len(preds_strict)):
        if abs(preds_strict[i] - labels[i]) <= 1:
            preds_tolerant[i] = labels[i]    
            
    all_predictions_strict.extend(preds_strict.tolist())
    all_predictions_tolerant.extend(preds_tolerant.tolist())
    all_labels.extend(labels.tolist())
    all_topics.extend(val_dataset["Topic"])
    
    # ---------------------------------
    # 7b. Compute per-topic metrics
    # ---------------------------------
    unique_topics = set(val_dataset["Topic"])
    for topic in unique_topics:
        # Mask for this topic
        topic_mask = np.array(val_dataset["Topic"]) == topic
        topic_logits = logits[topic_mask]
        topic_labels = labels[topic_mask]

        # Evaluate metrics on these topic-specific slices
        # We'll just re-use the same `regression_metrics` function
        # by building a tuple that looks like (logits, labels).
        topic_eval = regression_metrics((topic_logits, topic_labels))
        
        # Store
        for k, v in topic_eval.items():
            topic_wise_metrics[topic][k].append(v)

    wandb.finish() 
    
        
# ------------------------------------------------
# 8. Aggregate Cross-Fold Results
# ------------------------------------------------
final_metrics = {}
for key, values in fold_metrics.items():
    # Usually "eval_loss", "eval_mse", etc. start with "eval_"
    # We just do a mean & std
    mean_val = np.mean(values)
    std_val  = np.std(values)
    final_metrics[key] = {"mean": mean_val, "std": std_val}

# Per-topic metrics
final_topic_metrics = {}
for topic, metric_dict in topic_wise_metrics.items():
    final_topic_metrics[topic] = {}
    for k, vals in metric_dict.items():
        mean_val = np.mean(vals)
        std_val  = np.std(vals)
        final_topic_metrics[topic][k] = {"mean": mean_val, "std": std_val}

# Print or save final metrics
print("\n=== Final Cross-Validation Metrics (Overall) ===")
for metric, stats in final_metrics.items():
    print(f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

print("\n=== Final Cross-Validation Metrics (Per-Topic) ===")
for topic, metrics_dict in final_topic_metrics.items():
    print(f"\nTopic: {topic}")
    for metric, stats in metrics_dict.items():
        print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
# Save per-topic aggregated metrics
with open("./ModelSave_Regression/per_topic_evaluation.txt", "w") as f:
    for topic, metrics in final_topic_metrics.items():
        f.write(f"Metrics for Topic: {topic}\n")
        for metric, stats in metrics.items():
            f.write(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}\n")
        f.write("\n")
        
        
# -----------------------------------
# 8) Save Predictions for Future Comparison
# -----------------------------------
prediction_data = {
    "labels": all_labels,
    "strict_predictions": all_predictions_strict,
    "flexible_predictions": all_predictions_tolerant,
    "topics": all_topics
}

# Save to JSON
with open("./ModelSave_Regression/regression_model_predictions.json", "w") as f:
    json.dump(prediction_data, f, indent=4)

print("Base model predictions saved to './ModelSave_Regression/regression_model_predictions.json'")

# -----------------------------------
# 9. PLOT CONFUSION MATRICES FOR ALL TOPICS
# -----------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json

## Collect final predictions and labels across folds
final_labels = np.array(all_labels)
final_predictions_strict = np.array(all_predictions_strict)
final_predictions_flexible = np.array(all_predictions_tolerant)
final_topics = all_topics

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
fig.suptitle(r"$\bf{Confusion\ Matrices\ for}\ \bf{\it{MADRS\text{-}BERT}}$", fontsize=16)
plt.savefig("./ModelSave_Regression/Confusion_Matrices_Fine_tuned_Strict_3x3.tiff", format="tiff", dpi=600)
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
fig.suptitle(r"$\bf{Confusion\ Matrices\ for}\ \bf{\it{MADRS\text{-}BERT\text{-}flexible}}$", fontsize=16)
plt.savefig("./ModelSave_Regression/Confusion_Matrices_Fine_tuned_Flexible_3x3.tiff", format="tiff", dpi=600)
plt.close()
