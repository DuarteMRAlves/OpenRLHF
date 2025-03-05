import csv
import sys
from datasets import Dataset

def main(csv_path):
    # Dictionary to hold segment information keyed by segment_id
    segments = {}

    # Open the CSV file and iterate manually over each row
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            seg_id = row['segment_id']
            model = row['model']
            # If this is the first time we see this segment_id, initialize its record.
            if seg_id not in segments:
                segments[seg_id] = {
                    'segment_id': seg_id,
                    'lp': row['lp'],       # language pair
                    'source': row['source'],
                    'translations': {},    # will store system: translation
                    'scores': {}           # will store system: metric score (xcomet_xl_xxl)
                }
            # Store the translation and score for this system
            segments[seg_id]['translations'][model] = row['mt']
            try:
                # Convert the metric score to a float; adjust if your CSV has different formatting
                segments[seg_id]['scores'][model] = float(row['xcomet_xl_xxl'])
            except ValueError:
                segments[seg_id]['scores'][model] = None

    # Create a list of rows that will form our final dataset
    dataset_rows = []
    for seg_id, data in segments.items():
        # Compute ranking: sort systems by score (highest first) and join model names with " > "
        # Only consider systems with a valid (non-None) score
        sorted_systems = sorted(
            [(m, score) for m, score in data['scores'].items() if score is not None],
            key=lambda x: x[1],
            reverse=True
        )
        ranking = " > ".join([m for m, _ in sorted_systems])

        # Flatten the data for this segment
        row_dict = {
            'segment_id': data['segment_id'],
            'lp': data['lp'],
            'source': data['source'],
            'ranking': ranking
        }

        # Add one column per system for its translation
        for m, translation in data['translations'].items():
            row_dict[m] = translation

        # Add one column per system for its score (column name: <system>_score)
        for m, score in data['scores'].items():
            row_dict[f"{m}_score"] = score

        dataset_rows.append(row_dict)

    # Convert the list of dictionaries into a dictionary of lists (columns) for the Dataset
    if dataset_rows:
        columns = {key: [] for key in dataset_rows[0].keys()}
        for row in dataset_rows:
            for key in columns:
                columns[key].append(row.get(key))
    else:
        columns = {}

    # Create a Hugging Face Dataset and save it to disk
    dataset = Dataset.from_dict(columns)
    #dataset.save_to_disk("processed_dataset")
    print(dataset)
    print(dataset[0])
    print("Processed dataset saved to the 'processed_dataset' directory.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_csv_path>")
        sys.exit(1)
    main(sys.argv[1])
