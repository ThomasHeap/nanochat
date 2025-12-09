"""
Repackage the DeepFoldProtein/uniref50_processed dataset into parquet shards.

This converts the zstd-compressed JSONL files into parquet format with:
- Each shard containing ~100MB of sequences (after compression)
- Parquet files written with row group size of 1024
- Only the 'sequence' field is kept (other metadata discarded)

The resulting parquet files can be used with the standard nanochat dataloader
for faster loading and better compatibility.

Usage:
    python dev/repackage_protein_data.py --split train --num-files 10
    python dev/repackage_protein_data.py --split validation --num-files -1
"""

import os
import argparse
import time
import json
import zstandard as zstd
from huggingface_hub import hf_hub_download, list_repo_files

import pyarrow as pa
import pyarrow.parquet as pq

# Configuration
REPO_ID = "DeepFoldProtein/uniref50_processed"
OUTPUT_BASE_DIR = os.path.expanduser("~/.cache/nanochat/protein_data")

def download_and_process_files(split, num_files=-1, max_seq_length=None):
    """
    Download protein dataset files and convert to parquet format.
    
    Args:
        split: One of "train", "validation", or "test"
        num_files: Number of files to process (-1 for all)
        max_seq_length: Optional maximum sequence length filter
    """
    print(f"Repackaging {split} split of protein dataset...")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    # Get list of files
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
    split_files = [f for f in all_files if f.startswith(f"{split}/") and f.endswith('.jsonl.zst')]
    split_files = sorted(split_files)
    
    if num_files > 0:
        split_files = split_files[:num_files]
    
    print(f"Processing {len(split_files)} files from {split} split")
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE_DIR, split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files and write to parquet
    sequences_per_shard = 100000  # Adjust based on desired shard size
    row_group_size = 1024
    
    shard_sequences = []
    shard_index = 0
    total_sequences = 0
    total_time = 0
    start_time = time.time()
    
    for file_idx, remote_path in enumerate(split_files):
        print(f"\n[{file_idx+1}/{len(split_files)}] Processing {remote_path}...")
        
        # Download file
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            repo_type="dataset",
        )
        
        # Read and decompress
        dctx = zstd.ZstdDecompressor()
        with open(local_path, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                text_stream = reader.read()
                lines = text_stream.decode('utf-8').strip().split('\n')
                
                sequences_in_file = 0
                for line in lines:
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    sequence = data['sequence']
                    
                    # Apply length filter if specified
                    if max_seq_length is not None and len(sequence) > max_seq_length:
                        continue
                    
                    shard_sequences.append(sequence)
                    sequences_in_file += 1
                    
                    # Write shard if we have enough sequences
                    if len(shard_sequences) >= sequences_per_shard:
                        write_shard(output_dir, shard_index, shard_sequences, row_group_size)
                        total_sequences += len(shard_sequences)
                        shard_sequences = []
                        shard_index += 1
                
                print(f"  Extracted {sequences_in_file} sequences from this file")
    
    # Write remaining sequences
    if shard_sequences:
        write_shard(output_dir, shard_index, shard_sequences, row_group_size)
        total_sequences += len(shard_sequences)
        shard_index += 1
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Repackaging complete!")
    print(f"Total sequences: {total_sequences:,}")
    print(f"Total shards: {shard_index}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


def write_shard(output_dir, shard_index, sequences, row_group_size):
    """Write a shard of sequences to parquet file."""
    shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
    
    # Create PyArrow table
    table = pa.Table.from_pydict({"text": sequences})
    
    # Write to parquet
    pq.write_table(
        table,
        shard_path,
        row_group_size=row_group_size,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )
    
    file_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
    print(f"  Wrote {shard_path} ({len(sequences):,} sequences, {file_size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repackage protein dataset into parquet format")
    parser.add_argument("--split", type=str, default="valid", 
                        choices=["train", "valid", "test"],
                        help="Which split to process (default: valid)")
    parser.add_argument("--num-files", type=int, default=-1,
                        help="Number of files to process, -1 for all (default: -1)")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Maximum sequence length to include (default: None)")
    
    args = parser.parse_args()
    
    download_and_process_files(
        split=args.split,
        num_files=args.num_files,
        max_seq_length=args.max_seq_length
    )
    
    print("\nTo use the repackaged data, you can now use the standard nanochat dataloader:")
    print("from nanochat.dataloader import tokenizing_distributed_data_loader_with_state")
    print(f"\nThe parquet files are located at: {OUTPUT_BASE_DIR}/{args.split}/")
