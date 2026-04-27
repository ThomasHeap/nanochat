"""
Protein sequence dataset utilities.
Handles the DeepFoldProtein/uniref50_processed dataset format (.jsonl.zst files).
"""

import io
import os
import json
import random
import zstandard as zstd
from huggingface_hub import hf_hub_download

REPO_ID = "DeepFoldProtein/uniref50_processed"
PROTEIN_DATA_DIR = "/data-fsx/tomheap-sandbox/protein_data"
PROTEIN_HF_CACHE = "/data-fsx/tomheap-sandbox/protein_hf_cache"
VALID_SPLITS = {"train", "valid", "test"}

os.makedirs(PROTEIN_DATA_DIR, exist_ok=True)
os.makedirs(PROTEIN_HF_CACHE, exist_ok=True)

_cached_file_lists = {}


def list_protein_files(split="train"):
    """Return sorted list of HF repo paths for a given split. Result is cached in-process."""
    assert split in VALID_SPLITS, f"Invalid split '{split}'. Expected one of {VALID_SPLITS}"

    if split in _cached_file_lists:
        return _cached_file_lists[split]

    from huggingface_hub import list_repo_files
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    split_files = sorted(f for f in all_files if f.startswith(f"{split}/") and f.endswith(".jsonl.zst"))
    _cached_file_lists[split] = split_files
    return split_files


def protein_sequences_iter(split="train", start=0, step=1, epoch=0, seed=42):
    """
    Yield (sequence, file_idx) pairs from the dataset.
    DDP sharding is at file level: rank r handles files r, r+step, r+2*step, ...
    Files are shuffled each epoch for training (deterministic given seed+epoch).
    """
    assert split in VALID_SPLITS, f"Invalid split '{split}'. Expected one of {VALID_SPLITS}"

    files = list(list_protein_files(split))  # copy so shuffle is local
    if split == "train":
        random.Random(seed + epoch).shuffle(files)

    for file_idx in range(start, len(files), step):
        remote_path = files[file_idx]
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            repo_type="dataset",
            cache_dir=PROTEIN_HF_CACHE,
        )
        dctx = zstd.ZstdDecompressor()
        with open(local_path, "rb") as f:
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    sequence = data.get("sequence", "")
                    if sequence:
                        yield sequence, file_idx


def protein_sequences_packed_iter(split="train", start=0, step=1, max_length=1024, epoch=0, seed=42):
    """
    Yield (packed_doc, file_idx) pairs where packed_doc is a list of pre-tokenized
    token-id lists. The dataloader adds BOS/EOS around each list.
    Sequences longer than max_length (including BOS/EOS) are skipped.
    """
    from nanochat.tokenizer import ProteinTokenizer
    tokenizer = ProteinTokenizer()

    current_doc = []
    current_length = 0
    current_file_idx = start

    for sequence, file_idx in protein_sequences_iter(split, start, step, epoch, seed):
        current_file_idx = file_idx
        seq_tokens = tokenizer.encode(sequence)
        seq_length = len(seq_tokens) + 2  # +2 for BOS and EOS

        if seq_length > max_length:
            continue

        if current_length + seq_length > max_length and current_doc:
            yield current_doc, current_file_idx
            current_doc = []
            current_length = 0

        current_doc.append(seq_tokens)
        current_length += seq_length

        if current_length >= max_length * 0.8:
            yield current_doc, current_file_idx
            current_doc = []
            current_length = 0

    if current_doc:
        yield current_doc, current_file_idx


if __name__ == "__main__":
    print("Testing protein dataset loader...")
    for split in ["train", "valid", "test"]:
        files = list_protein_files(split)
        print(f"{split}: {len(files)} files")

    print("\nFirst 5 sequences from valid split:")
    for i, (seq, file_idx) in enumerate(protein_sequences_iter("valid")):
        if i >= 5:
            break
        print(f"  [{file_idx}] len={len(seq)}: {seq[:60]}...")
