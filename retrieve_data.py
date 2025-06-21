import os
import re
import json
import argparse
from datasets import load_dataset

def has_mostly_amharic(text: str, threshold: float = 0.90) -> bool:
    if not text:
        return False
    amharic_count = sum(1 for char in text if '\u1200' <= char <= '\u137F' or char.isdigit() or char.isspace())
    return amharic_count / len(text) >= threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieve data from huggingface datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to retrieve")
    parser.add_argument("--type", type=str, default="pretraining", help="Whether the dataset is for pretraining or finetuning", choices=["pretraining", "finetuning"])
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to retrieve(usually one for each language)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="Split of the dataset to retrieve(train, test, val)")
    parser.add_argument("--min-length", type=int, default=128, help="Minimum character length for filtering")
    parser.add_argument("--keys", type=str, default="text", help="Comma-separated list of keys to retrieve")
    args = parser.parse_args()

    dir = os.path.join("data", args.type, f"{args.dataset.replace('/', '-')}")
    os.makedirs(dir, exist_ok=True)
    file_path = os.path.join(dir, f"{args.split}.jsonl")

    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    print(f"Processing {args.split} split of {args.dataset} dataset...")
    
    example_count = 0
    filtered_count = 0
    total_bytes_written = 0
    
    with open(file_path, "w", encoding="utf-8") as f:
        for example in ds:
            text = ""
            for key in example:
                if key in args.keys.split(","):
                    if text:
                        text += "\n"
                    if example.get(key):
                        text += example.get(key)
            text = text.strip()
            if not has_mostly_amharic(text):
                continue
            
            if text and len(text) >= args.min_length:
                json.dump(text, f, ensure_ascii=False)
                f.write("\n")
                filtered_count += 1
            example_size = len(json.dumps(text, ensure_ascii=False).encode("utf-8")) + 1
            total_bytes_written += example_size

            example_count += 1
            if example_count % 1_000_000 == 0:
                size_mb = total_bytes_written / (1024 * 1024)
                print(f"  Processed {example_count:,} samples, Filtered {filtered_count:,} saved, ~{size_mb:.2f} MB")
                
    with open(os.path.join(dir, f"{args.split}.args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    print(f"Saved {filtered_count:,} filtered strings (out of {example_count:,} total) of {args.split} split of {args.dataset} dataset to {file_path}")
    print(f"Total size: {total_bytes_written / (1024 * 1024):.2f} MB")

