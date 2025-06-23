# COMPARE SYNTHETIC DATA WITH REAL DATA. 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load the pre-trained Sentence-BERT model for German
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load real and synthetic data from the given Excel files
real_data_path = 'PATH TO REAL DATA.xlsx'
synthetic_data_path = 'PATH TO SYNTHETIC DATA.xlsx'

real_df = pd.read_excel(real_data_path)
synthetic_df = pd.read_excel(synthetic_data_path)

# Ensure both datasets have a 'Transcription' and 'Score' column
real_df = real_df.dropna(subset=['Transcription', 'Score'])
synthetic_df = synthetic_df.dropna(subset=['Transcription', 'Score'])

# Extract sentences and scores
real_sentences = real_df['Transcription'].tolist()
synthetic_sentences = synthetic_df['Transcription'].tolist()
real_scores = real_df['Score'].tolist()
synthetic_scores = synthetic_df['Score'].tolist()

# Encode the sentences using the pre-trained model
real_embeddings = model.encode(real_sentences)
synthetic_embeddings = model.encode(synthetic_sentences)

# Calculate cosine similarity between real and synthetic sentences
similarities = cosine_similarity(real_embeddings, synthetic_embeddings)

# Group the data by scores to compare within similar severity levels
score_groups = sorted(set(real_scores))  # Get unique scores

# Create a dictionary to store average similarities per score group
similarities_per_group = {}

for score in score_groups:
    # Filter data for the specific score group
    real_group_indices = [i for i, score_real in enumerate(real_scores) if score_real == score]
    synthetic_group_indices = [i for i, score_synthetic in enumerate(synthetic_scores) if score_synthetic == score]
    
    # Extract the corresponding embeddings and calculate cosine similarity
    real_group_embeddings = real_embeddings[real_group_indices]
    synthetic_group_embeddings = synthetic_embeddings[synthetic_group_indices]
    
    if len(real_group_embeddings) > 0 and len(synthetic_group_embeddings) > 0:
        group_similarities = cosine_similarity(real_group_embeddings, synthetic_group_embeddings)
        
        # Store the average similarity for this score group
        avg_group_similarity = group_similarities.mean()
        similarities_per_group[score] = avg_group_similarity

# Output average similarity for each score group
print("Average Similarity by Score Group:")
for score, similarity in similarities_per_group.items():
    print(f"Score {score}: Average Similarity = {similarity:.4f}")

# Overall average similarity
overall_avg_similarity = similarities.mean()
print(f"\nOverall Average Similarity between all real and synthetic sentences: {overall_avg_similarity:.4f}")

# Maximum similarity between any real and synthetic sentence
max_similarity = similarities.max()
print(f"Max similarity: {max_similarity:.4f}")

# Minimum similarity between any real and synthetic sentence
min_similarity = similarities.min()
print(f"Min similarity: {min_similarity:.4f}")