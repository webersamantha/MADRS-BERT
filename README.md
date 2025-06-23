# MADRS-BERT

The codes described below can be executed in their numerical order to reproduce the analyses of the Paper "Weber et al., (2025) Using Fine-tuned Large Language Models for Depression Evaluation."
DOI: https://doi.org/10.21203/rs.3.rs-6555767/v1

All codes have been written by Samantha Weber (samantha.weber@bli.uzh.ch) in close collaboration with my friend ChatGPT.

## 00_Download_Convert_Video.py

This script is designed to process video recordings of clinical interviews conducted with patients using the Montgomery-Åsberg Depression Rating Scale (MADRS). The interviews, recorded in Swiss German or German, are stored in the ./Video folder. 
The script performs the following key tasks:

1. Video Processing & Audio Extraction
   - Identifies video files ending with `_MADRS.mp4` in participant folders.
   - Extracts the audio track from each video and saves it as a .wav file in the 01_Raw directory.

2. Audio Preprocessing
   - Converts extracted audio to mono-channel format for consistency.
   - Saves the filtered version in the `02_Filtered` directory.

3. Subject Tracking & Excel Logging
   - Checks if a participant has already been processed by referencing an Excel file (`01_ManualChecks.xlsx`).
   - Prevents duplicate processing by skipping previously logged subjects.
   - Updates the Excel file after successful audio extraction.

4. Error Handling & Logging
   - Implements robust error handling for video/audio processing failures.
   - Uses logging to track processing status and potential issues.

## 01_Convert_MADRAS_Diarization_BothSpeakers.py

This script handles the speaker diarization and transcription of the interview data. 

1. Extracts Audio from Videos
   - Identifies subject folders (MC_1XXXX).
   - Searches for video files ending with `_MADRS.mp4`.
   - Extracts the audio track and saves it in WAV format.
   - Converts the audio to mono-channel if needed.

2. Performs Speaker Diarization
   - Uses PyAnnote’s speaker diarization pipeline to identify different speakers (`pyannote/speaker-diarization-3.1`).
   - Segments the audio file into speaker-specific parts.

3. Transcribes Speech to Text
   - Uses Whisper (OpenAI) to transcribe each diarized audio segment.
   - Saves the transcriptions along with subject IDs and speakers in an Excel file.

4. Tracks Processed Subjects in an Excel File
   - Checks if a subject is already processed (avoiding duplicate work).
   - Saves processed subject information into an Excel log (`01_all_transcriptions.xlsx`).

## 02_Clear_Transcription.py

1. Merging & Cleaning Transcriptions
   - Loads the raw transcriptions from an Excel file.
   - Merges consecutive speeches by the same speaker, preventing unnecessary fragmentation.
   - Replaces specific words or phrases (e.g., "Vielen Dank" → "mhm", "Amen" → "mhm").
   - Saves the processed transcriptions into a new Excel file.

2. LLM-Based Grammar Correction
   - Loads the processed transcriptions.
   - Uses an AI-based text correction model (MBart50) to improve grammar and fluency.
   - Iterates through each transcription and applies German-specific corrections.
   - Saves the corrected transcriptions into a final Excel file.

## 03_Prepare_Figure_DistributionPlot.py

Distribution Plot (Figure 2)

Key functionalities:
1. Loads MADRS data from an Excel file.
2. Extracts and organizes scores per topic for both real and synthetic data.
3. Aggregates scores per subject and topic to avoid duplicates.
4. Visualizes the MADRS score distributions using stacked bar plots:
   - **X-axis:** Score values (0-6)
   - **Y-axis:** Count of occurrences
   - **Color coding:** 
     - Purple (#440154FF) for real patient transcript data.
     - Green (#21908CFF) for synthetic transcript data.
5. Generates a **3×3 grid of subplots**, one for each MADRS topic.
6. Saves the final figure as a high-resolution `.tiff` file for reporting.

## 04_Prepare_FineTuneData_MADRS.py

This script prepares MADRS (Montgomery-Åsberg Depression Rating Scale) clinical interview data 
for fine-tuning a model to predict MADRS scores based on transcribed dialogues. The process 
involves data structuring, JSONL generation for model training, and score distribution visualization.

Key functionalities:
1. **Load and preprocess data**:
   - Reads transcriptions and scores from an Excel file.
   - Extracts topics and corresponding dialogue from patient interviews.
   - Associates each topic with its corresponding MADRS description and scoring criteria.

2. **Generate JSONL file for fine-tuning**:
   - Formats the dialogue and scoring guidelines into structured JSONL entries.
   - Constructs an "instruction-response" format for model training.
   - Writes the processed data to a `.jsonl` file for later use in fine-tuning.

3. **Score extraction and tracking**:
   - Extracts the first available score for each topic per subject.
   - Stores scores in a dictionary to track their distribution across topics.

4. **Visualization of MADRS score distributions**:
   - Uses `matplotlib` to generate bar plots for score distributions.
   - Saves the visualizations as `.tiff` images with a high resolution for analysis.

Output files:
   - **JSONL file (`output_data_v2.jsonl`)**: Contains structured training data for fine-tuning.
   - **Score distribution plots (`[topic]_score_distribution.tiff`)**: Visual representations of MADRS score distributions.

This script ensures a structured and standardized data pipeline for training language models 
on MADRS clinical assessments, improving interpretability and automation in depression severity analysis.

## 05_BaseModel_BERT_Evaluation_FinalFlex.py

This script evaluates a **BERT-based model** for MADRS (Montgomery-Åsberg Depression Rating Scale) 
score classification based on transcribed clinical interviews. The workflow includes **data preprocessing, 
model evaluation, and visualization of confusion matrices** for both strict and flexible scoring.

Key functionalities:
1. **Load and preprocess data**:
   - Reads a dataset from a `.jsonl` file containing transcribed MADRS interviews.
   - Tokenizes the text using `BERT-base-german-cased` tokenizer.
   - Extracts **MADRS scores** and **topic labels** for each transcription.

2. **Model Evaluation**:
   - Uses a pre-trained `BERTForSequenceClassification` model to **predict MADRS scores**.
   - Implements strict and flexible scoring:
     - **Strict evaluation**: Exact match required between predicted and true scores.
     - **Flexible evaluation**: Predictions within **±1 score** of the true score are considered correct.
   - Computes key performance metrics including:
     - **Accuracy**
     - **Mean Absolute Error (MAE)**

3. **Confusion Matrix Visualization**:
   - Generates confusion matrices for **strict** and **flexible** score evaluations.
   - Creates **3×3 grid plots** for each MADRS topic.
   - Saves high-resolution `.tiff` images for reporting.

4. **Per-Topic Evaluation**:
   - Evaluates model performance **separately for each MADRS topic**.
   - Saves per-topic evaluation metrics in a `.txt` file.
   - Generates and saves **per-topic confusion matrices**.

5. **Export Evaluation Results**:
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

## 05_FineTune_BERT_Final_Flex.py

This script fine-tunes a **BERT-based model** for MADRS (Montgomery-Åsberg Depression Rating Scale) 
score classification using **cross-validation and a MSE loss function**. It also evaluates model performance 
with **strict and flexible scoring criteria**.

Key functionalities:
1. **Load and preprocess data**:
   - Reads a dataset from a `.jsonl` file containing transcribed MADRS interviews.
   - Tokenizes the text using `BERT-base-german-cased` tokenizer.
   - Extracts **MADRS scores** and **topic labels** for each transcription.

2. **Cross-validation fine-tuning**:
   - Implements **5-fold cross-validation** to train and validate the model.
   - Uses a **MSE Loss Function**.

3. **Model Evaluation**:
   - Evaluates model performance using:
     - **Strict evaluation**: Exact match required between predicted and true scores.
     - **Flexible evaluation**: Predictions within **±1 score** of the true score are considered correct.
   - Computes key performance metrics including:
     - **Accuracy**
     - **Mean Absolute Error (MAE)**

4. **Confusion Matrix Visualization**:
   - Generates confusion matrices for **strict** and **flexible** score evaluations.
   - Creates **3×3 grid plots** for each MADRS topic.
   - Saves high-resolution `.tiff` images for reporting.

5. **Per-Topic Evaluation**:
   - Evaluates model performance **separately for each MADRS topic**.
   - Saves per-topic evaluation metrics in a `.txt` file.
   - Generates and saves **per-topic confusion matrices**.

6. **Export Evaluation Results**:
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

## 06_LearningCurve.py

This script performs repeates the 5-fold cross validation, and slowly increases training set used (ranging from 5% to 80% of entire dataset). See publication for more details. 

Data is plotted using 07_Plot_LearningCurve.py


## 07_StatisticalComparisonFineTuneBaseModel.py

This script performs **McNemar’s test** to statistically compare the performance of different **BERT-based models** 
for MADRS (Montgomery-Åsberg Depression Rating Scale) classification. The goal is to determine whether fine-tuning 
has led to a **significant improvement** over baseline performance.

Key functionalities:
1. **Load and Preprocess Confusion Matrices**:
   - Reads **strict and flexible** confusion matrices from JSON files.
   - Aligns topics between models for direct comparison.
   - Computes **correct predictions** and **total errors** for each model.

2. **Perform McNemar's Test for Paired Model Comparisons**:
   - Compares model performance topic-wise using a **contingency table**:
     - **Both models correct**
     - **Base model correct, fine-tuned model incorrect**
     - **Fine-tuned model correct, base model incorrect**
     - **Both models incorrect**
   - Runs **McNemar’s test** on each topic.
   - Adjusts p-values using **Bonferroni correction** for multiple comparisons.

3. **Compute Overall Error Reduction**:
   - Calculates total **errors reduced** by fine-tuning.
   - Expresses error reduction as a **percentage improvement** over the baseline model.

4. **Generate Contingency Table Heatmaps**:
   - Visualizes **3×3 grid plots** of contingency tables for each MADRS topic.
   - Annotates contingency tables with **p-values** to indicate statistical significance.

5. **Save Results and Figures**:
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

## 07_StatisticalComparisonFineTuneBaseModel.py

We applied a pre-trained Sentence-BERT model (https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) to embed the transcriptions of real patient interviews and synthetic data. These embeddings were compared using cosine similarity to assess how closely the synthetic sentences align with the real ones.
