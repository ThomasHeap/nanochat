"""
Protein dataset utilities for loading and processing protein sequences.
This module handles the DeepFoldProtein/uniref50_processed dataset format.
"""

import os
import json
import zstandard as zstd
from huggingface_hub import hf_hub_download

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset configuration

REPO_ID = "DeepFoldProtein/uniref50_processed"
base_dir = get_base_dir()
PROTEIN_DATA_DIR = os.path.join(base_dir, "protein_data")
os.makedirs(PROTEIN_DATA_DIR, exist_ok=True)

# Cache for dataset file listings
_cached_file_lists = {}

def list_protein_files(split="train"):
    """
    List all protein dataset files for a given split.
    Files are cached locally after first download.
    
    Args:
        split: One of "train", "valid", or "test"
    
    Returns:
        List of local file paths
    """
    assert split in ["train", "valid", "test"], f"Invalid split: {split}"
    
    # Use cached list if available
    cache_key = split
    if cache_key in _cached_file_lists:
        return _cached_file_lists[cache_key]
    
    # Get list of files from HuggingFace repo
    from huggingface_hub import list_repo_files
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    
    # Filter for this split
    split_prefix = f"{split}/"
    split_files = [f for f in all_files if f.startswith(split_prefix) and f.endswith('.jsonl.zst')]
    split_files = sorted(split_files)
    
    # Download files on demand (lazy downloading happens in iterator)
    _cached_file_lists[cache_key] = split_files
    return split_files


def protein_sequences_iter(split="train", start=0, step=1, max_length=None):
    """
    Iterate through protein sequences from the dataset.
    
    Args:
        split: One of "train", "validation", or "test"
        start: Starting file index (useful for DDP)
        step: Step size for file iteration (useful for DDP)
        max_length: Optional maximum sequence length filter
    
    Yields:
        Protein sequence strings
    """
    files = list_protein_files(split)
    
    # Use cache directory inside nanochat project
    cache_dir = os.path.join(get_base_dir(), "protein_hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    for file_idx in range(start, len(files), step):
        remote_path = files[file_idx]
        
        # Download file if not cached (to nanochat directory)
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            repo_type="dataset",
            cache_dir=cache_dir
        )
        
        # Read and decompress
        dctx = zstd.ZstdDecompressor()
        with open(local_path, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                text_stream = reader.read()
                lines = text_stream.decode('utf-8').strip().split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    sequence = data['sequence']
                    
                    # Apply length filter if specified
                    if max_length is not None and len(sequence) > max_length:
                        continue
                    
                    yield sequence


def protein_sequences_iter_batched(split="train", start=0, step=1, batch_size=1024, max_length=None):
    """
    Iterate through protein sequences in batches.
    
    Args:
        split: One of "train", "validation", or "test"
        start: Starting file index (useful for DDP)
        step: Step size for file iteration (useful for DDP)
        batch_size: Number of sequences per batch
        max_length: Optional maximum sequence length filter
    
    Yields:
        Lists of protein sequence strings
    """
    batch = []
    for sequence in protein_sequences_iter(split, start, step, max_length):
        batch.append(sequence)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield remaining sequences
    if batch:
        yield batch


if __name__ == "__main__":
    # Quick test
    print("Testing protein dataset loader...")
    print("\nTrain files:")
    train_files = list_protein_files("train")
    print(f"  Found {len(train_files)} files")
    
    print("\nValidation files:")
    val_files = list_protein_files("validation")
    print(f"  Found {len(val_files)} files")
    
    print("\nTest files:")
    test_files = list_protein_files("test")
    print(f"  Found {len(test_files)} files")
    
    print("\nLoading first 10 sequences from test set:")
    for i, seq in enumerate(protein_sequences_iter("test")):
        if i >= 10:
            break
        print(f"  {i+1}. Length {len(seq)}: {seq[:60]}...")
