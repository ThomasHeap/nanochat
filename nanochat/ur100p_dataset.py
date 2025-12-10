"""
UR100P dataset utilities for protein language modeling.
This module handles the chandar-lab/UR100P dataset format with proper text-style packing.
"""

import os
import json
from datasets import load_dataset
from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Dataset configuration

REPO_ID = "chandar-lab/UR100P"
base_dir = get_base_dir()
UR100P_DATA_DIR = os.path.join(base_dir, "ur100p_cache")
os.makedirs(UR100P_DATA_DIR, exist_ok=True)


def load_ur100p_dataset(split="train", streaming=False):
    """
    Load the UR100P dataset from HuggingFace.
    
    Args:
        split: One of "train", "validation", or "test"
        streaming: If True, use streaming mode to avoid downloading everything
    
    Returns:
        Dataset object
    """
    assert split in ["train", "validation", "test"], f"Invalid split: {split}"
    
    print(f"Loading {REPO_ID} dataset, split: {split}, streaming: {streaming}")
    
    # Load dataset with caching to our directory
    dataset = load_dataset(
        REPO_ID, 
        split=split,
        cache_dir=UR100P_DATA_DIR,
        streaming=streaming
    )
    
    if not streaming:
        print(f"Loaded {len(dataset)} sequences from {split} split")
    else:
        print(f"Loaded streaming dataset from {split} split")
    return dataset


def ur100p_sequences_iter(split="train", start=0, step=1):
    """
    Iterate through protein sequences from the UR100P dataset.
    
    Args:
        split: One of "train", "validation", or "test"
        start: Starting index (useful for DDP)
        step: Step size for iteration (useful for DDP)
    
    Yields:
        Protein sequence strings
    """
    dataset = load_ur100p_dataset(split)
    
    # Iterate with DDP support
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


def ur100p_sequences_packed_iter(split="train", start=0, step=1, max_length=1024):
    """
    Iterate through protein sequences with text-style packing.
    Each yielded item is a "document" that can contain multiple sequences
    packed together with proper <bos>/<eos> delimiters.
    
    Args:
        split: One of "train", "validation", or "test"
        start: Starting index (useful for DDP)
        step: Step size for iteration (useful for DDP)
        max_length: Maximum length for packed sequences
    
    Yields:
        Lists of protein sequences to be packed together
    """
    from nanochat.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()
    
    current_doc = []
    current_length = 0
    
    for sequence in ur100p_sequences_iter(split, start, step):
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
