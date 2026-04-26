"""
Dataloader for protein language modeling using UR100P dataset.
Streams protein sequences with proper text-style packing using <bos>/<eos> tokens.
"""

from collections import deque
import torch

from nanochat.common import get_dist_info
from nanochat.ur100p_dataset import ur100p_sequences_packed_iter
from nanochat.tokenizer import get_tokenizer


def ur100p_dataloader_with_state(B, T, split, max_seq_length=1024, device="cuda", resume_state_dict=None, val_split_ratio=0.5, shuffle=True, seed=42):
    """
    Stream protein sequences from UR100P dataset, pack with <bos>/<eos>, and yield training batches.
    
    This implementation uses proper text-style packing and supports epoch tracking.
    Since UR100P only has train/test splits, we automatically split the test set into validation/test.
    
    Args:
        B: Batch size
        T: Sequence length (context length)
        split: One of "train", "validation", or "test"
        max_seq_length: Maximum length for packed sequences
        device: Device to place tensors on
        resume_state_dict: Optional state dict for resuming training
        val_split_ratio: Fraction of original test set to use as validation (default 0.5)
        shuffle: Whether to shuffle the dataset (only works with non-streaming mode)
        seed: Random seed for shuffling
    
    Yields:
        (inputs, targets, state_dict) tuples
    """
    assert split in ["train", "validation", "test"], "split must be 'train', 'validation', or 'test'"
    
    # Get DDP info for distributed training
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    bos_token_id = tokenizer.get_bos_token_id()
    eos_token_id = tokenizer.get_eos_token_id()
    
    # Resume state
    resume_doc_idx = resume_state_dict.get("doc_idx", 0) if resume_state_dict else 0
    resume_epoch = resume_state_dict.get("epoch", 0) if resume_state_dict else 0
    resume_tokens_in_epoch = resume_state_dict.get("tokens_in_epoch", 0) if resume_state_dict else 0
    first_pass = True
    
    # Track epoch completion
    current_epoch = resume_epoch
    tokens_in_current_epoch = resume_tokens_in_epoch
    doc_idx = resume_doc_idx
    
    # Infinite iterator over packed protein sequence documents
    def packed_documents():
        nonlocal first_pass, current_epoch, doc_idx
        
        while True:  # Multi-epoch iteration
            start_idx = doc_idx if first_pass else ddp_rank
            first_pass = False
            
            epoch_start_doc_idx = doc_idx
            for doc in ur100p_sequences_packed_iter(
                split=split,
                start=start_idx, 
                step=ddp_world_size,
                max_length=max_seq_length,
                val_split_ratio=val_split_ratio,
                shuffle=shuffle,
                seed=seed
            ):
                yield doc, doc_idx, False  # False = not epoch complete
                doc_idx += ddp_world_size
            
            # Epoch completed for this rank
            current_epoch += 1
            doc_idx = 0  # Reset for next epoch
            yield [], -1, True  # True = epoch complete marker
    
    documents = packed_documents()
    
    # Token buffer for accumulating tokens
    needed_tokens = B * T + 1  # +1 for target at last position
    token_buffer = deque()
    position_buffer = deque()

    while True:
        # Accumulate enough tokens for one iteration
        while len(token_buffer) < needed_tokens:
            doc, current_doc_idx, epoch_complete = next(documents)

            # Handle epoch completion marker
            if epoch_complete:
                print(f"[UR100P Dataloader] Epoch {current_epoch-1} completed for rank {ddp_rank}")
                continue

            # Pack sequences in document with <bos>/<eos>
            for seq_tokens in doc:
                # Add <bos> + sequence + <eos>
                tokens = [bos_token_id] + seq_tokens + [eos_token_id]
                position_buffer.extend(range(len(tokens)))
                token_buffer.extend(tokens)
                tokens_in_current_epoch += len(tokens)

        # Extract tokens for this batch
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

        # Create tensor with memory pinning for CUDA
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
        position_ids_cpu = torch.tensor([position_buffer.popleft() for _ in range(needed_tokens)], dtype=torch.long, pin_memory=use_cuda_optimizations)[:-1]

        # Create inputs/targets
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]

        # Reshape and move to device
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        position_ids = position_ids_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        state_dict = {
            "doc_idx": current_doc_idx,
            "epoch": current_epoch,
            "tokens_in_epoch": tokens_in_current_epoch
        }
        yield inputs, targets, position_ids, state_dict


def ur100p_dataloader(*args, **kwargs):
    """Helper function that only emits inputs/targets without state_dict"""
    for inputs, targets, position_ids, state_dict in ur100p_dataloader_with_state(*args, **kwargs):
        yield inputs, targets, position_ids


def test_ur100p_dataloader():
    """Test the dataloader functionality"""
    print("Testing UR100P dataloader...")
    
    # Test basic functionality
    dataloader = ur100p_dataloader_with_state(
        B=2, T=128, split="train", max_seq_length=512, device="cpu"
    )
    
    print("Getting first batch...")
    try:
        inputs, targets, position_ids, state_dict = next(dataloader)
        print(f"Inputs shape: {inputs.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Position IDs shape: {position_ids.shape}")
        print(f"State dict keys: {list(state_dict.keys())}")
        print(f"Sample tokens: {inputs[0, :20].tolist()}")
        
        # Decode a sample to verify tokenization
        tokenizer = get_tokenizer()
        sample_decoded = tokenizer.decode(inputs[0, :50].tolist())
        print(f"Decoded sample: {sample_decoded}")
        
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ur100p_dataloader()
