'''
Author: Samantha Weber (samantha.weber@bli.uzh.ch)
Project: MULTICAST - University of Zurich & University Hospital of Psychiatry Zurich

This script performs **McNemar’s test** to statistically compare the performance of different **BERT-based models** 
for MADRS (Montgomery-Åsberg Depression Rating Scale) classification. The goal is to determine whether fine-tuning 
has led to a **significant improvement** over baseline performance.

Key functionalities:
1️. **Load and Preprocess Confusion Matrices**:
   - Reads **strict and flexible** confusion matrices from JSON files.
   - Aligns topics between models for direct comparison.
   - Computes **correct predictions** and **total errors** for each model.

2️. **Perform McNemar's Test for Paired Model Comparisons**:
   - Compares model performance topic-wise using a **contingency table**:
     - **Both models correct**
     - **Base model correct, fine-tuned model incorrect**
     - **Fine-tuned model correct, base model incorrect**
     - **Both models incorrect**
   - Runs **McNemar’s test** on each topic.
   - Adjusts p-values using **Bonferroni correction** for multiple comparisons.

3️. **Compute Overall Error Reduction**:
   - Calculates total **errors reduced** by fine-tuning.
   - Expresses error reduction as a **percentage improvement** over the baseline model.

4️. **Generate Contingency Table Heatmaps**:
   - Visualizes **3×3 grid plots** of contingency tables for each MADRS topic.
   - Annotates contingency tables with **p-values** to indicate statistical significance.

5️. **Save Results and Figures**:
   - Writes McNemar’s test results (p-values, corrected p-values) into a **text file**.
   - Saves contingency table heatmaps as **PNG images**.

Comparison Scenarios:
- **MADRS-BERT-flexible vs. BERT-base-flexible**
- **MADRS-BERT vs. BERT-base**
- **MADRS-BERT-flexible vs. MADRS-BERT**
- **BERT-base-flexible vs. BERT-base**

Output files:
- **`mcnemar_test_results/*.txt`** → Text files with McNemar’s test results for each comparison.
- **`mcnemar_test_results/*.png`** → Heatmap visualizations of contingency tables.

This script provides a **statistical and visual performance comparison** between different **MADRS-BERT** 
and **BERT-base** models, helping to assess whether fine-tuning leads to **significant improvements** in 
classification accuracy.
'''


import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths to the exported confusion matrices
base_model_path = "./Results/BaseModel/Base_confusion_matrices.json"
fine_tuned_model_path = "./Results/RegressionModel/confusion_matrices_fineTuned.json"

# Define the output folder
output_folder = "mcnemar_test_results"
os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

# Load confusion matrices
with open(base_model_path, "r") as f:
    base_confusion_matrices = json.load(f)

with open(fine_tuned_model_path, "r") as f:
    fine_tuned_confusion_matrices = json.load(f)

# Define topic translations and order
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

# Exclude "Allgemein" early
filtered_topics = [topic for topic in topic_order if topic in base_confusion_matrices.keys()]

# Define comparisons
comparisons = {
    "MADRS-BERT-flexible vs. BERT-base-flexible": {"base_model_cm": "flexible", "fine_tuned_cm": "flexible"},
    "MADRS-BERT vs. BERT-base": {"base_model_cm": "strict", "fine_tuned_cm": "strict"},
    "MADRS-BERT-flexible vs. MADRS-BERT": {"base_model_cm": "strict", "fine_tuned_cm": "flexible"},
    "BERT-base-flexible vs. BERT-base": {"base_model_cm": "strict", "fine_tuned_cm": "flexible"}
}

# Initialize results storage
comparison_results = {}
overall_error_reduction = {}

for comparison_name, cm_types in comparisons.items():
    results = []
    base_total_errors = 0  # Accumulate total errors for the base model
    fine_tuned_total_errors = 0  # Accumulate total errors for the fine-tuned model
    total_samples_all_topics = 0  # Accumulate total samples across all topics

    for topic in filtered_topics:
        base_cm = np.array(base_confusion_matrices[topic][cm_types["base_model_cm"]])
        fine_tuned_cm = np.array(fine_tuned_confusion_matrices[topic][cm_types["fine_tuned_cm"]])

        # Correct predictions for each model
        base_correct = np.diag(base_cm).sum()
        fine_tuned_correct = np.diag(fine_tuned_cm).sum()

        # Total samples
        total_samples = base_cm.sum()
        
        # Increment totals for error calculations
        base_total_errors += total_samples - base_correct
        fine_tuned_total_errors += total_samples - fine_tuned_correct
        total_samples_all_topics += total_samples

        # Derive contingency table
        both_correct = min(base_correct, fine_tuned_correct)
        base_correct_only = base_correct - both_correct
        fine_tuned_correct_only = fine_tuned_correct - both_correct
        both_wrong = total_samples - (both_correct + base_correct_only + fine_tuned_correct_only)

        # Build contingency table
        contingency_table = np.array([[both_correct, base_correct_only],
                                    [fine_tuned_correct_only, both_wrong]])
        # Perform McNemar's test
        result = mcnemar(contingency_table, exact=True)
        results.append({
            "topic": topic_translations[topic],  # Use English translation
            "contingency_table": contingency_table.tolist(),
            "p-value": result.pvalue
        })

        # Calculate percentage reduction in errors
        if base_total_errors > 0:  # Avoid division by zero
            error_reduction = ((base_total_errors - fine_tuned_total_errors) / base_total_errors) * 100
        else:
            error_reduction = 0

        overall_error_reduction[comparison_name] = {
            "base_total_errors": base_total_errors,
            "fine_tuned_total_errors": fine_tuned_total_errors,
            "error_reduction_percentage": error_reduction,
            "total_samples": total_samples_all_topics
        }

    # Store results for this comparison
    comparison_results[comparison_name] = results

# Multiple comparisons correction
from statsmodels.stats.multitest import multipletests

for comparison_name, results in comparison_results.items():
    p_values = [res["p-value"] for res in results]
    corrected_p_values = multipletests(p_values, method="bonferroni")[1]
    for res, corrected_p in zip(results, corrected_p_values):
        res["corrected_p-value"] = corrected_p


for comparison_name, results in comparison_results.items():
    # Save results to a .txt file
    txt_file_path = os.path.join(output_folder, f"{comparison_name.replace(' ', '_').replace('.', '').lower()}_results.txt")
    with open(txt_file_path, "w") as f:
        f.write(f"McNemar's Test Results for {comparison_name}\n")
        f.write("=" * (len(comparison_name) + 26) + "\n\n")
        for res in results:
            f.write(f"Topic: {res['topic']}\n")
            f.write(f"Contingency Table:\n{np.array(res['contingency_table'])}\n")
            f.write(f"p-value: {res['p-value']:.20f}\n")
            f.write(f"Corrected p-value: {res['corrected_p-value']:.20f}\n\n")

        # Include error reduction details
        error_data = overall_error_reduction[comparison_name]
        f.write(f"Overall Error Analysis for {comparison_name}:\n")
        f.write(f"Base Model Total Errors: {error_data['base_total_errors']}\n")
        f.write(f"Fine-Tuned Model Total Errors: {error_data['fine_tuned_total_errors']}\n")
        f.write(f"Error Reduction Percentage: {error_data['error_reduction_percentage']:.2f}%\n")
        f.write(f"Total Samples Across Topics: {error_data['total_samples']}\n\n")

    # Create a 3x3 grid for contingency tables
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for ax, res in zip(axes, results):
        contingency_table = np.array(res["contingency_table"])

        sns.heatmap(
            contingency_table,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            linewidths=0.5,
            xticklabels=["Model Correct", "Model Not Correct"],
            yticklabels=["Model Correct", "Model Not Correct"]
        )

        # Format P-value for display
        p_value_display = (
            f"P < 0.0001" if res['corrected_p-value'] < 0.0001 else f"P = {res['corrected_p-value']:.4f}"
        )

        ax.set_title(
            f"{res['topic']} ($\it{{{p_value_display}}}$)",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlabel(f"{comparison_name.split('vs.')[0].strip()} Outcome", fontsize=10)
        ax.set_ylabel(f"{comparison_name.split('vs.')[1].strip()} Outcome", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)

    # Remove unused subplots
    for ax in axes[len(results):]:
        fig.delaxes(ax)

    # Adjust layout and save the figure
    fig.suptitle(f"Contingency Tables: {comparison_name}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure in the output folder
    output_path = os.path.join(output_folder, f"{comparison_name.replace(' ', '_').replace('.', '').lower()}_contingency_tables.png")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

print(f"All results and contingency tables saved in the folder: {output_folder}")