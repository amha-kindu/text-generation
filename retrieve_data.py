from preprocessor import AmharicPreprocessor
from datasets import load_dataset
import json
import os

# Define base directory and paths for each split
base_dir = "pretraining-corpus"
os.makedirs(base_dir, exist_ok=True)

# Dictionary to map splits to their file paths
split_paths = {
    "train": os.path.join(base_dir, "train.jsonl"),
    # "test": os.path.join(base_dir, "test2.jsonl"),
    # "validation": os.path.join(base_dir, "validation.jsonl")
}

# Target size for the subset (5GB = 5,242,880,000 bytes)
target_size_bytes = 5_242_880_000  # 5GB in bytes
min_char_length = 128  # Minimum character length for filtering

# Initialize preprocessor (replace with your actual initialization)
preprocessor = AmharicPreprocessor()

# Load and save each split
for split_name, file_path in split_paths.items():
    # Load the dataset in streaming mode for the current split
    ds = load_dataset("yordanoswuletaw/amharic-pretraining-corpus", split=split_name, streaming=True)
    
    print(f"Processing {split_name} split...")
    total_bytes_written = 0
    example_count = 0
    filtered_count = 0

    # Open file in append mode and write each string as a JSON line
    with open(file_path, "w", encoding="utf-8") as f:
        for example in ds:
            # Assuming the dataset has a 'text' field (adjust key if different)
            text = example.get("text", "")
            
            # Apply preprocessing
            text = preprocessor.execute(text)
            
            # Filter based on character length
            if text and len(text) >= min_char_length:
                # Write the preprocessed string as a JSON line
                json.dump(text, f, ensure_ascii=False)
                f.write("\n")  # Add newline for JSONL format
                filtered_count += 1
                
                # Update total bytes written (size of the JSON string)
                example_size = len(json.dumps(text, ensure_ascii=False).encode("utf-8")) + 1  # +1 for newline
                total_bytes_written += example_size
                
                # Stop when we reach ~5GB
                # if total_bytes_written >= target_size_bytes:
                #     break

            example_count += 1

            # Print feedback every 1,000,000 samples retrieved
            if example_count % 1_000_000 == 0:
                size_mb = total_bytes_written / (1024 * 1024)
                print(f"  Processed {example_count:,} samples, Filtered {filtered_count:,} saved, ~{size_mb:.2f} MB")

    # Final report for this split
    final_size_mb = total_bytes_written / (1024 * 1024)
    print(f"Saved {filtered_count:,} filtered strings (out of {example_count:,} total) of {split_name} split to {file_path}")
    print(f"Total size: {final_size_mb:.2f} MB")

print("Processing complete.")