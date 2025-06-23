"""
Author: Samantha Weber (samantha.weber@bli.uzh.ch)

Description: This project has been generated within the MULTICAST [https://www.multicast.uzh.ch/en.html] project at University of Zurich and University Hospital of Psychiatry Zurich.

This script prepares MADRS (Montgomery-Åsberg Depression Rating Scale) clinical interview data 
for fine-tuning a model to predict MADRS scores based on transcribed dialogues. The process 
involves data structuring, JSONL generation for model training, and score distribution visualization.

Key functionalities:
1️. **Load and preprocess data**:
   - Reads transcriptions and scores from an Excel file.
   - Extracts topics and corresponding dialogue from patient interviews.
   - Associates each topic with its corresponding MADRS description and scoring criteria.

2️. **Generate JSONL file for fine-tuning**:
   - Formats the dialogue, topic descriptions, and scoring guidelines into structured JSONL entries.
   - Constructs an "instruction-response" format for model training.
   - Writes the processed data to a `.jsonl` file for later use in fine-tuning.

3️. **Score extraction and tracking**:
   - Extracts the first available score for each topic per subject.
   - Stores scores in a dictionary to track their distribution across topics.

4️. **Visualization of MADRS score distributions**:
   - Uses `matplotlib` to generate bar plots for score distributions.
   - Saves the visualizations as `.tiff` images with a high resolution for analysis.

Output files:
- **JSONL file (`output_data_realdata.jsonl`)**: Contains structured training data for fine-tuning.
- **Score distribution plots (`[topic]_score_distribution.tiff`)**: Visual representations of MADRS score distributions.

This script ensures a structured and standardized data pipeline for training language models 
on MADRS clinical assessments, improving interpretability and automation in depression severity analysis.

"""

import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter
import datetime

structured_interview_info = {
    "topics": {
        "Allgemein": "Dieses Item beinhaltet die sich in Sprache, Mimik und Haltung ausdrückende Mutlosigkeit, Niedergeschlagenheit und Verzweiflung. Bewerten Sie nach Schweregrad und der Unfähigkeit zur Aufheiterung.",
        "Traurigkeit": "Beinhaltet die Angaben des Patienten über eine gedrückte Stimmung, einschließlich Entmutigung, Niedergeschlagenheit, dem Gefühl der Hilflosigkeit und Hoffnungslosigkeit.",
        "Anspannung": "Beinhaltet sowohl ein schwer definierbares Gefühl von Missbehagen als auch Gereiztheit, Unruhe, innere Erregung bis hin zu Angst und Panik.",
        "Schlaf": "Beinhaltet die subjektive Erfahrung verminderter Schlafdauer oder Schlaftiefe, verglichen mit dem vorher normalen Schlafverhalten.",
        "Appetit": "Beinhaltet das Gefühl, im Vergleich zum Normalzustand weniger Appetit zu haben. Bewerten Sie nach Stärke des Appetitverlusts bzw. wie sehr man sich zum Essen zwingen muss.",
        "Konzentration": "Beinhaltet Schwierigkeiten, sich zu konzentrieren, angefangen vom einfachen Sammeln der eigenen Gedanken bis zum völligen Verlust der Konzentrationsfähigkeit.",
        "Antriebslosigkeit": "Beinhaltet Schwierigkeiten, einen Anfang zu finden bzw. Schwerfälligkeit, alltägliche Tätigkeiten anzufangen und durchzuführen.",
        "Gefühlslosigkeit": "Beinhaltet das subjektive Empfinden des verminderten Interesses für die Umgebung oder Aktivitäten, die vorher Freude bereiteten.",
        "Gedanken": "Beinhaltet Schuldgefühle, Minderwertigkeitsgefühle, Selbstvorwürfe, Versündigungsideen, Reuegefühle und Untergangsideen.",
        "Suizid": "Beinhaltet das Gefühl, das Leben sei nicht mehr lebenswert, der natürliche Tod sei eine Erlösung, Selbstmordgedanken und Vorbereitung zum Selbstmord."
    },
    "scoring": {
        "Allgemein": {
            0: "Keine Traurigkeit.",
            1: "",
            2: "Sieht niedergeschlagen aus, ist aber ohne Schwierigkeiten aufzuheitern.",
            3: "",
            4: "Wirkt die meiste Zeit über traurig und unglücklich.",
            5: "",
            6: "Sieht die ganze Zeit über traurig und unglücklich aus. Extreme Niedergeschlagenheit."
        },
        "Traurigkeit": {
            0: "Den Umständen entsprechende gelegentliche Traurigkeit.",
            1: "",
            2: "Traurig oder mutlos, jedoch ohne Schwierigkeiten aufzuheitern.",
            3: "",
            4: "Ständige Gefühle von Traurigkeit und Trübsinn. Die Stimmung ist jedoch immer noch durch äußere Umstände beeinflussbar.",
            5: "",
            6: "Andauernde oder unveränderliche Traurigkeit, Mutlosigkeit oder Hoffnungslosigkeit."
        },
        "Anspannung": {
            0: "Gelassen. Nur vorübergehende innere Anspannung.",
            1: "",
            2: "Gelegentlich Gefühl von Missbehagen und Gereiztheit.",
            3: "",
            4: "Anhaltendes Gefühl innerer Anspannung oder Erregung. Kurzzeitige Panikanfälle, die der Patient nur mit Mühe beherrscht.",
            5: "",
            6: "Nicht beherrschbare Angst oder Erregung. Überwältigende Panik."
        },
        "Schlaf": {
            0: "Schläft wie gewöhnlich.",
            1: "",
            2: "Leichte Schwierigkeiten einzuschlafen. Oberflächlicher, unruhiger Schlaf. Geringfügig verkürzte Schlafdauer.",
            3: "",
            4: "Schlaf mindestens 2 Stunden verkürzt oder unterbrochen.",
            5: "",
            6: "Weniger als 2-3 Stunden Schlaf."
        },
        "Appetit": {
            0: "Normaler oder verstärkter Appetit.",
            1: "",
            2: "Geringfügige Appetitminderung.",
            3: "",
            4: "Kein Appetit. Essen schmeckt nicht.",
            5: "",
            6: "Nur mit Überredung zum Essen zu bewegen."
        },
        "Konzentration": {
            0: "Keine Konzentrationsschwierigkeiten.",
            1: "",
            2: "Gelegentliche Schwierigkeiten, die eigenen Gedanken zu sammeln.",
            3: "",
            4: "Schwierigkeiten, sich zu konzentrieren und einen Gedanken festzuhalten. Die Fähigkeit, zu lesen oder ein Gespräch zu führen wird dadurch eingeschränkt.",
            5: "",
            6: "Nicht in der Lage, ohne Schwierigkeiten zu lesen oder ein Gespräch zu führen."
        },
        "Antriebslosigkeit": {
            0: "Nahezu keine Schwierigkeiten, einen Anfang zu finden. Keine Trägheit.",
            1: "",
            2: "Schwierigkeiten, eine Tätigkeit anzufangen.",
            3: "",
            4: "Schwierigkeiten, einfache Routinetätigkeiten anzufangen, Ausführung nur mit Mühe.",
            5: "",
            6: "Völlige Antriebslosigkeit. Unfähig, ohne Hilfe etwas zu tun."
        },
        "Gefühlslosigkeit": {
            0: "Normales Interesse für die Umgebung oder für andere Menschen.",
            1: "",
            2: "Weniger Spaß an früheren Interessen.",
            3: "",
            4: "Verlust des Interesses für die Umgebung. Verlust der Gefühle für Freunde und Bekannte.",
            5: "",
            6: "Die Erfahrung der Gefühllosigkeit. Unfähig, Ärger, Trauer oder Freude zu empfinden."
        },
        "Gedanken": {
            0: "Keine pessimistischen Gedanken.",
            1: "",
            2: "Zeitweise Gedanken, „versagt zu haben“, Selbstvorwürfe und Selbsterniedrigungen.",
            3: "",
            4: "Beständige Selbstanklagen. Eindeutige, aber logisch noch haltbare Schuld- und Versündigungsideen. Zunehmend pessimistisch in Bezug auf die Zukunft.",
            5: "",
            6: "Untergangswahn, Gefühl von Reue oder nicht wiedergutzumachenden Sünden. Selbstanklagen, die zwar absurd, jedoch unerschütterlich sind."
        },
        "Suizid": {
            0: "Freude am Leben oder die Ansicht, dass man im Leben die Dinge nehmen muss, wie sie kommen.",
            1: "",
            2: "Lebensmüde. Nur zeitweise Selbstmordgedanken.",
            3: "",
            4: "Lieber tot. Selbstmordgedanken sind häufig. Selbstmord wird als möglicher Ausweg angesehen, jedoch keine genauen Pläne oder Absichten.",
            5: "",
            6: "Deutliche Selbstmordpläne, wenn sich eine Gelegenheit bietet. Aktive Vorbereitung zum Selbstmord."
        }
    }
}
# Read the Excel file
excel_file_path = "./05_MADRS_FINAL.xlsx"

df = pd.read_excel(excel_file_path)

# Define the output path for the single JSONL file
output_path = './output_data_v2_realdata.jsonl'

# Initialize a dictionary to track scores per topic
score_distribution = {topic: [] for topic in structured_interview_info['topics']}

# Open a file to write the JSONL data
with open(output_path, 'w', encoding='utf-8') as jsonl_file:
    # Process the data by subject and topic
    for subject_id, group in df.groupby('Subject'):
        print(f"Processing Subject: {subject_id}")
        
        for topic, topic_group in group.groupby('Topic'):
            print(f"  Topic: {topic}")
            
            # Initialize the input text with topic, description, and scoring guidelines
            input_text = f"Dialogue:\n"
            score = None
            
            # Track if we have already recorded a score for this topic per subject
            topic_score_recorded = False
            
            # Construct the dialogue
            for _, row in topic_group.iterrows():
                speaker = "Interviewer" if row['Speaker'] == 'SPEAKER_00' else "Patient"
                transcription = row['Transcription']
                
                # Append each turn in the dialogue to the input text
                input_text += f"{speaker}: {transcription}\n"
                
                # Capture the first valid score for the topic per subject and stop recording further
                if not topic_score_recorded and row['Speaker'] == 'SPEAKER_01' and pd.notna(row['Score']):
                    score = row['Score']
                    score_distribution[topic].append(int(score))  # Add the score for this topic per subject
                    topic_score_recorded = True  # Mark that we've recorded the score for this subject and topic
            
            # If a score was found, create a JSON entry
            if score is not None:
                output_text = f"Score: {score}"
                combined_text = (
                    f"Topic: {topic} \n"
                    f"{input_text.strip()}\n\n"
                )
                
                json_object = {
                    "input": combined_text,
                    #"topic": topic,
                    "output": output_text,
                }
                
                # Write the JSON object as a new line in the JSONL file
                jsonl_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
                print(f"Added entry for topic '{topic}' with score: {score}")


# Step 2: Plot the score distribution for each topic
for topic, scores in score_distribution.items():
    if scores:
        # Count occurrences of each score
        score_counts = Counter(scores)
        possible_scores = list(range(7))  # Scores range from 0 to 6
        counts = [score_counts.get(score, 0) for score in possible_scores]

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(possible_scores, counts, color='blue', alpha=0.7)
        
        plt.xlabel('Score', fontsize=18)
        plt.ylabel('Count', fontsize = 18)
        plt.title(f'Score Distribution for Topic: {topic}', fontsize = 20)
        plt.xticks(possible_scores, fontsize = 16)  # Ensure x-axis labels are 0 to 6
        plt.yticks(fontsize = 16)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Save the plot with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"./_notpublic/01_Figures/{topic}_score_distribution.tiff", format='tiff', dpi=600)

        plt.close()

        print(f"Score distribution plot for topic '{topic}' saved.")
        print(f"Data has been successfully saved to {output_path}")
