"""
Author: Samantha Weber (samantha.weber@bli.uzh.ch)
Date: June 2024

Description: This project has been generated within the MULTICAST [https://www.multicast.uzh.ch/en.html] project at University of Zurich and University Hospital of Psychiatry Zurich.
1️. Merging & Cleaning Transcriptions
	•	Loads the raw transcriptions from an Excel file.
	•	Merges consecutive speeches by the same speaker, preventing unnecessary fragmentation.
	•	Replaces specific words or phrases (e.g., "Vielen Dank" → "mhm", "Amen" → "mhm").
	•	Saves the processed transcriptions into a new Excel file.

2️. LLM-Based Grammar Correction --> This was only partially done for manuscript data preparation. 
	•	Loads the processed transcriptions.
	•	Uses an AI-based text correction model (MBart50) to improve grammar and fluency.
	•	Iterates through each transcription and applies German-specific corrections.
	•	Saves the corrected transcriptions into a final Excel file.
"""

import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm  # For progress tracking

# 1. Initialize, File paths ------------------------------------------------------------------------------

file_path = './03_Text_Extracted/01_all_transcriptions_20240703.xlsx'
file_path ='./03_Text_Extracted/02_processed_transcriptions.xlsx'
save_path = './03_Text_Extracted/'
processed_file_path = save_path + '02_processed_transcriptions.xlsx'
corrected_file_path = save_path + '03_processed_LLMcorrected_transcriptions.xlsx'

# 2. Initial Processing: Merging transcriptions -----------------------------------------------------------
# Load the Excel file
df = pd.read_excel(file_path)

# Ensure the columns are named correctly
df.columns = ['Subject', 'Speaker', 'Transcription']

# Initialize an empty list to hold the processed data
processed_data = []

# Variables to keep track of the previous subject and speaker
prev_subject = None
prev_speaker = None
merged_transcription = ''

# Iterate over the DataFrame
for index, row in df.iterrows():
    current_subject = row['Subject']
    current_speaker = row['Speaker']
    current_transcription = str(row['Transcription']) if pd.notna(row['Transcription']) else ''
    
    # Replace "Vielen Dank" and "Amen" with "mhm"  ##ADD HERE MORE WORDS IF NEEDED. 
    current_transcription = current_transcription.replace("Vielen Dank", "mhm")
    current_transcription = current_transcription.replace("Amen", "mhm")

    # Check if the current row has the same subject and speaker as the previous row
    if current_subject == prev_subject and current_speaker == prev_speaker:
        # Merge the transcriptions
        if current_transcription:  # Only add if there's remaining transcription
            merged_transcription += ' ' + current_transcription
    else:
        # If the subject or speaker is different, save the previous merged transcription
        if prev_subject is not None and prev_speaker is not None:
            processed_data.append([prev_subject, prev_speaker, merged_transcription])
        
        # Reset the merged transcription with the current transcription
        merged_transcription = current_transcription
    
    # Update the previous subject and speaker
    prev_subject = current_subject
    prev_speaker = current_speaker

# Append the last merged transcription
if prev_subject is not None and prev_speaker is not None:
    processed_data.append([prev_subject, prev_speaker, merged_transcription])

# Create a new DataFrame from the processed data
processed_df = pd.DataFrame(processed_data, columns=['Subject', 'Speaker', 'Transcription'])

# Save the processed data to a new Excel file
processed_df.to_excel(processed_file_path, index=False)

print(f"Processed transcriptions have been successfully merged and saved to '{processed_file_path}'.")

# 3. LLM Correction: Correcting the merged transcriptions -------------------------------------------------

# Load the processed file for LLM correction
processed_df = pd.read_excel(processed_file_path)

# Initialize the text correction model and tokenizer
model_name = "MRNH/mbart-german-grammar-corrector"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="de_DE", tgt_lang="de_DE")

# Initialize an empty list to hold the corrected data
corrected_data = []

# Apply LLM correction to each transcription with progress tracking
for index, row in tqdm(processed_df.iterrows(), total=processed_df.shape[0], desc="Progress: LLM-based grammar correction"):
    # Ensure input_text is a string
    input_text = str(row['Transcription'])
    
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate the corrected text
    output_ids = model.generate(input_ids, forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"])
    corrected_transcription = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Append the corrected transcription
    corrected_data.append([row['Subject'], row['Speaker'], corrected_transcription])

# Create a new DataFrame from the corrected data
corrected_df = pd.DataFrame(corrected_data, columns=['Subject', 'Speaker', 'Transcription'])

# Save the corrected data to a new Excel file
corrected_df.to_excel(corrected_file_path, index=False)

print(f"LLM-corrected transcriptions have been successfully saved to '{corrected_file_path}'.")