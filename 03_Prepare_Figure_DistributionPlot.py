"""
Author: Samantha Weber (samantha.weber@bli.uzh.ch)

Description: This project has been generated within the MULTICAST [https://www.multicast.uzh.ch/en.html] project at University of Zurich and University Hospital of Psychiatry Zurich.
This script processes MADRS (Montgomery-Åsberg Depression Rating Scale) scores 
from patient transcript data, comparing real and synthetic subject data. The analysis 
focuses on different MADRS topics, evaluating the distribution of scores across subjects.

Key functionalities:
1. Loads MADRS data from an Excel file.
2️. Extracts and organizes scores per topic for both real and synthetic data.
3️. Aggregates scores per subject and topic to avoid duplicates.
4️. Visualizes the MADRS score distributions using stacked bar plots:
   - **X-axis:** Score values (0-6)
   - **Y-axis:** Count of occurrences
   - **Color coding:** 
     - Purple (#440154FF) for real patient transcript data.
     - Green (#21908CFF) for synthetic transcript data.
5️. Generates a **3×3 grid of subplots**, one for each MADRS topic.
6️. Saves the final figure as a high-resolution `.tiff` file for reporting.

This visualization helps compare real vs. synthetic transcript data in MADRS evaluations, 
allowing researchers to assess consistency and potential biases in synthetic data.
"""


import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Path to your dataset
excel_file_path = "./05_MADRS_FINAL.xlsx"  # PATH TO DATASET
df = pd.read_excel(excel_file_path)

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

# Initialize a dictionary to track scores per topic for real and synthetic data
score_distribution = {topic: {"real": [], "synthetic": []} for topic in topic_order}

# Populate the score distribution ensuring one score per subject and topic
for (topic, subject), group in df.groupby(["Topic", "Subject"]):
    if topic in topic_order:  # Only include topics in the specified order
        # Get the first non-NaN score for this subject and topic
        valid_scores = group['Score'].dropna()
        if not valid_scores.empty:
            score = int(valid_scores.iloc[0])  # Take the first valid score
            if subject.endswith("_Syn"):  # Check if it's synthetic, only synthetic data has the _Syn in the participant code. 
                score_distribution[topic]["synthetic"].append(score)
            else:  # Otherwise, it's real data
                score_distribution[topic]["real"].append(score)

# Create a 3x3 grid for the plots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()  # Flatten axes for easy iteration

# Plot the score distribution for each topic
for ax, topic in zip(axes, topic_order):
    data = score_distribution[topic]
    real_scores = data["real"]
    synthetic_scores = data["synthetic"]

    # Count occurrences of each score
    real_counts = Counter(real_scores)
    synthetic_counts = Counter(synthetic_scores)
    possible_scores = list(range(7))  # Scores range from 0 to 6

    real_values = [real_counts.get(score, 0) for score in possible_scores]
    synthetic_values = [synthetic_counts.get(score, 0) for score in possible_scores]

    # Create a stacked bar plot
    ax.bar(possible_scores, real_values, label="Patient transcript data", color="#440154FF", alpha=0.7)
    ax.bar(possible_scores, synthetic_values, bottom=real_values, label="Synthetic data", color="#21908CFF", alpha=0.7)

    # Plot styling
    ax.set_title(topic_translations[topic], fontsize=14, fontweight='bold')
    ax.set_xlabel('MADRS score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(possible_scores)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Remove unused subplots if fewer than 9 topics
for ax in axes[len(topic_order):]:
    fig.delaxes(ax)

# Add a single legend outside the grid
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.02))

# Adjust layout to make space for the legend
plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust bottom and top margins to fit the legend

# Save the final figure
plt.savefig("./Figure_Distribution.tiff", format='tiff', dpi=600)
