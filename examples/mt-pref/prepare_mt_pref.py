import csv
import sys
import random
import pandas as pd
from datasets import Dataset

# Define the system message (first message in the messages list)
system_message = (
    "You are an assistant in a conversation with a user. "
    "The user asks a question, and you solve it. "
    "You should first think about the reasoning process in your mind and then provide the user with the answer. "
    "The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

def main(csv_path, output_path):
    # Dictionary to hold segment information keyed by segment_id
    segments = {}

    df = pd.read_csv(csv_path)

    # Read the CSV file using the built-in csv module
    for _, row in df.iterrows():
        seg_id = row['segment_id']
        lp = row['lp']
        full_id = f"{lp}_{seg_id}"
        model = row['model']
        if full_id not in segments:
            segments[full_id] = {
                'segment_id': seg_id,
                'lp': row['lp'],       # language pair
                'source': row['source'],
                'translations': {},    # system: translation
                'scores': {}           # system: metric score (xcomet_xl_xxl)
            }
        segments[full_id]['translations'][model] = row['mt']
        segments[full_id]['scores'][model] = float(row['xcomet_xl_xxl'])

    # Build the dataset rows manually
    dataset_rows = []
    letters = ['A', 'B', 'C', 'D']
    for full_id, data in segments.items():
        # Compute ranking across all systems (sorted by score descending)
        sorted_systems = sorted(
            [(m, score) for m, score in data['scores'].items() if score is not None],
            key=lambda x: x[1],
            reverse=True
        )
        ranking = " > ".join([m for m, _ in sorted_systems])

        # Prepare the base row with segment info and all translations/scores
        row_dict = {
            'segment_id': data['segment_id'],
            'lp': data['lp'],
            'source': data['source'],
            'ranking': ranking
        }
        for m, translation in data['translations'].items():
            row_dict[m] = translation
        for m, score in data['scores'].items():
            row_dict[f"{m}_score"] = score

        # --- New feature: Build the messages field and related prompt information ---
        # Select four systems (or all if fewer than 4)
        systems = list(data['translations'].keys())
        if len(systems) < 4:
            print(f"Skipping segment {full_id} due to insufficient systems")
            continue
        selected_systems = random.sample(systems, 4)

        best_system = max(selected_systems, key=lambda s: data['scores'][s])

        # Shuffle the selected systems to randomize letter assignment
        random.shuffle(selected_systems)
        letter_mapping = {letter: system for letter, system in zip(letters, selected_systems)}

        choices = []
        for letter in letter_mapping:
            system_name = letter_mapping[letter]
            translation_text = data['translations'][system_name]
            choices.append(f"{letter}. {translation_text}")

        prompt_message = f"{data['source']}\nWhich of the following {data['lp']} translations is the best?\n\n{'\n'.join(choices)}\n\nAfter thinking about the reasoning process, respond with the letter (A, B, C, D) corresponding to the correct choice."

        # Identify the correct answer letter (translation with highest score among the four)
        correct_letter = None
        for letter, system in letter_mapping.items():
            if system == best_system:
                correct_letter = letter
                break

        # Build the messages field: list of two dictionaries
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_message}
        ]

        # Add new fields to the row: messages, correct_answer, and letter-to-system mapping (for bookkeeping)
        row_dict["messages"] = messages
        row_dict["correct_answer"] = correct_letter
        row_dict["letter_to_system_mapping"] = letter_mapping

        dataset_rows.append(row_dict)

    # Convert the list of row dictionaries into a column-oriented dictionary
    if dataset_rows:
        columns = {key: [] for key in dataset_rows[0].keys()}
        for row in dataset_rows:
            for key in columns:
                columns[key].append(row.get(key))
    else:
        columns = {}

    # Create and save a Hugging Face Dataset
    dataset = Dataset.from_dict(columns)
    print(dataset)
    print(dataset[0])
    dataset.save_to_disk(output_path)
    print(f"Processed dataset saved to the '{output_path}' directory.")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_csv_path> <output dataset path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
