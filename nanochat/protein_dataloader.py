"""
Dataloader for protein language modeling.
Streams protein sequences from the dataset, tokenizes them, and yields training batches.
"""

from collections import deque
import torch

from nanochat.common import get_dist_info
from nanochat.protein_dataset import protein_sequences_iter_batched
from nanochat.tokenizer import get_tokenizer


def protein_dataloader_with_state(B, T, split, batch_size=128, max_seq_length=2048, device="cuda", resume_state_dict=None):
    """
    Stream protein sequences, tokenize, and yield training batches.
    
    This implementation supports approximate resume training via state_dict.
    
    Args:
        B: Batch size
        T: Sequence length (context length)
        split: Either "train" or "validation"
        batch_size: Number of sequences to process at once before tokenizing
        max_seq_length: Maximum protein sequence length to include (longer sequences are skipped)
        device: Device to place tensors on
        resume_state_dict: Optional state dict for resuming training
    
    Yields:
        (inputs, targets, state_dict) tuples
    """
    assert split in ["train", "valid", "validation"], "split must be 'train', 'valid', or 'validation'"
    
    # Map split names to dataset split names (support both 'valid' and 'validation')
    if split == "train":
        dataset_split = "train"
    elif split in ["valid", "validation"]:
        dataset_split = "valid"
    else:
        dataset_split = split
    
    # Get DDP info for distributed training
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    bos_token_id = tokenizer.aa_to_id.get('<cls>', 0)  # Use <cls> as BOS for proteins
    
    # Resume state
    resume_file_idx = resume_state_dict.get("file_idx", 0) if resume_state_dict else 0
    first_pass = True
    
    # Infinite iterator over protein sequence batches
    def sequence_batches():
        nonlocal first_pass
        file_idx = resume_file_idx if first_pass else 0
        
        while True:  # Multi-epoch iteration
            start_idx = file_idx if first_pass else ddp_rank
            first_pass = False
            
            for batch, file_idx in protein_sequences_iter_batched_with_index(
                dataset_split, 
                start=start_idx, 
                step=ddp_world_size,
                batch_size=batch_size,
                max_length=max_seq_length
            ):
                yield batch, file_idx
            
            file_idx = 0  # Reset for next epoch
    
    batches = sequence_batches()
    
    # Token buffer for accumulating tokens
    needed_tokens = B * T + 1  # +1 for target at last position
    token_buffer = deque()
    
    while True:
        # Accumulate enough tokens for one iteration
        while len(token_buffer) < needed_tokens:
            seq_batch, file_idx = next(batches)
            
            # Tokenize sequences
            for seq in seq_batch:
                # Prepend BOS token and encode
                tokens = [bos_token_id] + tokenizer.encode(seq)
                token_buffer.extend(tokens)
        
        # Extract tokens for this batch
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        # Create tensor with memory pinning for CUDA
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        
        # Create inputs/targets
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        
        # Reshape and move to device
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        
        state_dict = {"file_idx": file_idx}
        yield inputs, targets, state_dict


def protein_sequences_iter_batched_with_index(split, start=0, step=1, batch_size=1024, max_length=None):
    """
    Helper that yields (batch, file_idx) tuples.
    """
    from nanochat.protein_dataset import list_protein_files
    from nanochat.common import get_base_dir
    from huggingface_hub import hf_hub_download
    import zstandard as zstd
    import json
    import os
    from nanochat.protein_dataset import REPO_ID
    
    files = list_protein_files(split)
    
    # Use cache directory inside nanochat project
    cache_dir = os.path.join(get_base_dir(), "protein_hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    for file_idx in range(start, len(files), step):
        remote_path = files[file_idx]
        
        # Check if file is already cached in our nanochat directory
        repo_cache_path = os.path.join(cache_dir, "datasets--DeepFoldProtein--uniref50_processed", "snapshots")
        is_cached = False
        if os.path.exists(repo_cache_path):
            for snapshot_dir in os.listdir(repo_cache_path):
                potential_path = os.path.join(repo_cache_path, snapshot_dir, remote_path)
                if os.path.exists(potential_path):
                    is_cached = True
                    break
        
        if not is_cached:
            print(f"[Protein Dataloader] Downloading {remote_path}...")
        
        # Download file if not cached (to nanochat directory)
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            repo_type="dataset",
            cache_dir=cache_dir
        )
        
        if not is_cached:
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"[Protein Dataloader] Downloaded {remote_path} ({file_size_mb:.1f} MB)")
        
        # Read and decompress
        dctx = zstd.ZstdDecompressor()
        with open(local_path, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                text_stream = reader.read()
                lines = text_stream.decode('utf-8').strip().split('\n')
                
                batch = []
                for line in lines:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    sequence = data['sequence']
                    
                    # Apply length filter
                    if max_length is not None and len(sequence) > max_length:
                        continue
                    
                    batch.append(sequence)
                    
                    if len(batch) >= batch_size:
                        yield batch, file_idx
                        batch = []
                
                # Yield remaining sequences
                if batch:
                    yield batch, file_idx


def protein_dataloader(*args, **kwargs):
    """Helper function that only emits inputs/targets without state_dict"""
    for inputs, targets, state_dict in protein_dataloader_with_state(*args, **kwargs):
        yield inputs, targets


# For backward compatibility and easy switching
tokenizing_distributed_protein_dataloader = protein_dataloader
tokenizing_distributed_protein_dataloader_with_state = protein_dataloader_with_state
