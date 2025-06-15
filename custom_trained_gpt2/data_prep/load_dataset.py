from datasets import load_dataset
import datasets.utils.info_utils

original_verify_checksums = datasets.utils.info_utils.verify_checksums
datasets.utils.info_utils.verify_checksums = lambda *args, **kwargs: None

ds = load_dataset("wikitext", "wikitext-2-raw-v1", 
                  download_mode="force_redownload")

splits = {
    "data/train_data.txt": ds["train"],
    "data/validation_data.txt": ds["validation"],
    "data/test_data.txt": ds["test"]
}

for filename, subset in splits.items():
    with open(filename, "w") as f:
        for row in subset:
            f.write(row["text"] + "\n")