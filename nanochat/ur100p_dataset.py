"""
UR100P dataset utilities for protein language modeling.
This module handles the chandar-lab/UR100P dataset format with proper text-style packing.
"""

import os
import json

# Set HuggingFace cache directories BEFORE importing datasets
os.environ["HF_HOME"] = "/data-fsx/tomheap-sandbox/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/data-fsx/tomheap-sandbox/huggingface_cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data-fsx/tomheap-sandbox/huggingface_cache/hub"

from datasets import load_dataset
from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset configuration

REPO_ID = "chandar-lab/UR100P"
# Use shared filesystem for k8s cluster
UR100P_DATA_DIR = "/data-fsx/tomheap-sandbox/ur100p_cache"
os.makedirs(UR100P_DATA_DIR, exist_ok=True)

# Ensure HuggingFace cache directories exist
for cache_dir in [os.environ["HF_HOME"], os.environ["HF_DATASETS_CACHE"], os.environ["HUGGINGFACE_HUB_CACHE"]]:
    os.makedirs(cache_dir, exist_ok=True)


def load_ur100p_dataset(split="train", streaming=False, val_split_ratio=0.5, shuffle=True, seed=42):
    """
    Load the UR100P dataset from HuggingFace.
    
    Since UR100P only has train/test splits, we split the test set into validation/test.
    
    Args:
        split: One of "train", "validation", or "test"
        streaming: If True, use streaming mode to avoid downloading everything
        val_split_ratio: Fraction of original test set to use as validation (default 0.5)
        shuffle: Whether to shuffle the dataset (only works with non-streaming mode)
        seed: Random seed for shuffling (default 42 for reproducibility)
    
    Returns:
        Dataset object
    """
    assert split in ["train", "validation", "test"], f"Invalid split: {split}"
    
    print(f"Loading {REPO_ID} dataset, split: {split}, streaming: {streaming}")
    
    # UR100P only has train/test, so we need to handle validation specially
    if split == "train":
        # Load train split directly
        dataset = load_dataset(
            REPO_ID, 
            split="train",
            cache_dir=UR100P_DATA_DIR,
            streaming=streaming
        )
    elif split in ["validation", "test"]:
        # Load the original test split and partition it
        if streaming:
            # For streaming, we can't easily split, so we'll use a different approach
            print(f"Note: For streaming mode, validation and test will overlap")
            print(f"Validation uses first {int(val_split_ratio*100)}% of test samples")
            dataset = load_dataset(
                REPO_ID, 
                split="test",
                cache_dir=UR100P_DATA_DIR,
                streaming=streaming
            )
            # For streaming, we'll handle the split logic in the iterator
        else:
            # Load full test set to split it
            full_test = load_dataset(
                REPO_ID, 
                split="test",
                cache_dir=UR100P_DATA_DIR,
                streaming=False
            )
            
            # Split the test set
            total_test_size = len(full_test)
            val_size = int(total_test_size * val_split_ratio)
            
            if split == "validation":
                # Take first portion as validation
                dataset = full_test.select(range(val_size))
                print(f"Created validation split: {len(dataset)} sequences from test set")
            else:  # split == "test"
                # Take remaining portion as test
                dataset = full_test.select(range(val_size, total_test_size))
                print(f"Created test split: {len(dataset)} sequences from test set")
    
    # Apply shuffling for non-streaming datasets
    if not streaming and shuffle and hasattr(dataset, 'shuffle'):
        print(f"Shuffling {split} dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)
    elif shuffle and streaming:
        print(f"Note: Shuffling not available for streaming datasets")
    
    if not streaming and split != "validation" and split != "test":
        print(f"Loaded {len(dataset)} sequences from {split} split")
    elif streaming:
        print(f"Loaded streaming dataset from {split} split")
    
    return dataset


def ur100p_sequences_iter(split="train", start=0, step=1, val_split_ratio=0.5, shuffle=True, seed=42):
    """
    Iterate through protein sequences from the UR100P dataset.
    
    Args:
        split: One of "train", "validation", or "test"
        start: Starting index (useful for DDP)
        step: Step size for iteration (useful for DDP)
        val_split_ratio: Fraction of original test set to use as validation
        shuffle: Whether to shuffle the dataset (only works with non-streaming mode)
        seed: Random seed for shuffling
    
    Yields:
        Protein sequence strings
    """
    dataset = load_ur100p_dataset(split, streaming=True, val_split_ratio=val_split_ratio, shuffle=shuffle, seed=seed)
    
    # For streaming validation/test splits, we need to handle the partitioning ourselves
    if split in ["validation", "test"] and hasattr(dataset, '__iter__'):  # streaming dataset
        count = 0
        for item in dataset:
            # For streaming, approximate split by counting items
            # This is a simple approach - in practice, we'd want deterministic splitting
            if split == "validation":
                # Only take items that fall in the first val_split_ratio portion
                # We'll use a simple modulo approach for demo purposes
                if (count % 100) < (val_split_ratio * 100):
                    yield_item = True
                else:
                    yield_item = False
            else:  # split == "test"
                # Only take items that fall in the remaining portion
                if (count % 100) >= (val_split_ratio * 100):
                    yield_item = True
                else:
                    yield_item = False
            
            if yield_item:
                # UR100P has 'sequence' field
                if 'sequence' in item:
                    sequence = item['sequence']
                elif 'text' in item:  # Fallback in case field name differs
                    sequence = item['text']
                else:
                    # Print available fields for debugging
                    print(f"Available fields in dataset item: {list(item.keys())}")
                    raise KeyError("No 'sequence' or 'text' field found in dataset item")
                
                # Yield clean protein sequence
                if sequence and len(sequence) > 0:
                    # Remove any whitespace and ensure only valid amino acids
                    clean_sequence = ''.join(c for c in sequence.upper() if c.isalpha())
                    if len(clean_sequence) > 0:
                        yield clean_sequence
            
            count += 1
    else:
        # For non-streaming or train split, iterate normally with DDP support
        if hasattr(dataset, '__len__'):  # non-streaming dataset
            for idx in range(start, len(dataset), step):
                item = dataset[idx]
                
                # UR100P has 'sequence' field
                if 'sequence' in item:
                    sequence = item['sequence']
                elif 'text' in item:  # Fallback in case field name differs
                    sequence = item['text']
                else:
                    # Print available fields for debugging
                    print(f"Available fields in dataset item: {list(item.keys())}")
                    raise KeyError("No 'sequence' or 'text' field found in dataset item")
                
                # Yield clean protein sequence
                if sequence and len(sequence) > 0:
                    # Remove any whitespace and ensure only valid amino acids
                    clean_sequence = ''.join(c for c in sequence.upper() if c.isalpha())
                    if len(clean_sequence) > 0:
                        yield clean_sequence
        else:  # streaming train split
            for item in dataset:
                # UR100P has 'sequence' field
                if 'sequence' in item:
                    sequence = item['sequence']
                elif 'text' in item:  # Fallback in case field name differs
                    sequence = item['text']
                else:
                    # Print available fields for debugging
                    print(f"Available fields in dataset item: {list(item.keys())}")
                    raise KeyError("No 'sequence' or 'text' field found in dataset item")
                
                # Yield clean protein sequence
                if sequence and len(sequence) > 0:
                    # Remove any whitespace and ensure only valid amino acids
                    clean_sequence = ''.join(c for c in sequence.upper() if c.isalpha())
                    if len(clean_sequence) > 0:
                        yield clean_sequence


def ur100p_sequences_packed_iter(split="train", start=0, step=1, max_length=1024, val_split_ratio=0.5, shuffle=True, seed=42):
    """
    Iterate through protein sequences with text-style packing.
    Each yielded item is a "document" that can contain multiple sequences
    packed together with proper <bos>/<eos> delimiters.
    
    Args:
        split: One of "train", "validation", or "test"
        start: Starting index (useful for DDP)
        step: Step size for iteration (useful for DDP)
        max_length: Maximum length for packed sequences
        val_split_ratio: Fraction of original test set to use as validation
        shuffle: Whether to shuffle the dataset (only works with non-streaming mode)
        seed: Random seed for shuffling
    
    Yields:
        Lists of protein sequences to be packed together
    """
    from nanochat.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()
    
    current_doc = []
    current_length = 0
    
    for sequence in ur100p_sequences_iter(split, start, step, val_split_ratio, shuffle, seed):
        # Estimate token length: <bos> + sequence + <eos>
        seq_tokens = tokenizer.encode(sequence)
        seq_length = len(seq_tokens) + 2  # +2 for <bos> and <eos>
        
        # If adding this sequence would exceed max_length, yield current doc
        if current_length + seq_length > max_length and current_doc:
            yield current_doc
            current_doc = []
            current_length = 0
        
        # Add sequence to current document
        current_doc.append(sequence)
        current_length += seq_length
        
        # If current doc is getting large, yield it
        if current_length >= max_length * 0.8:  # 80% threshold to avoid tiny sequences
            yield current_doc
            current_doc = []
            current_length = 0
    
    # Yield final document if not empty
    if current_doc:
        yield current_doc


if __name__ == "__main__":
    # Quick test with streaming mode to avoid downloading everything
    print("Testing UR100P dataset loader...")
    
    try:
        print("\nLoading train split in streaming mode...")
        train_dataset = load_ur100p_dataset("train", streaming=True)
        
        print("\nTesting first few sequences:")
        count = 0
        for item in train_dataset:
            if count >= 5:
                break
            
            # Check available fields
            if 'sequence' in item:
                sequence = item['sequence']
                clean_sequence = ''.join(c for c in sequence.upper() if c.isalpha())
                print(f"  {count+1}. Length {len(clean_sequence)}: {clean_sequence[:60]}...")
            else:
                print(f"  {count+1}. Available fields: {list(item.keys())}")
            count += 1
        
        print("\n✅ Dataset loading test successful!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("This might be normal if the dataset requires authentication or has a different structure.")
        import traceback
        traceback.print_exc()
