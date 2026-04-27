"""
Dataloader for protein language modeling.
Streams protein sequences with BOS/EOS packing and position tracking.
DDP sharding is at file level (each rank processes different files).
"""

from collections import deque
import torch

from nanochat.common import get_dist_info
from nanochat.protein_dataset import protein_sequences_packed_iter, VALID_SPLITS
from nanochat.tokenizer import ProteinTokenizer


def protein_dataloader_with_state(B, T, split, max_seq_length=2048, device="cuda", resume_state_dict=None, seed=42):
    """
    Stream protein sequences, pack with BOS/EOS, and yield training batches with state.

    Yields:
        (inputs, targets, position_ids, state_dict) tuples
    """
    assert split in VALID_SPLITS, f"Invalid split '{split}'. Expected one of {VALID_SPLITS}"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    tokenizer = ProteinTokenizer()
    bos_token_id = tokenizer.get_bos_token_id()
    eos_token_id = tokenizer.get_eos_token_id()

    resume_file_idx = resume_state_dict.get("file_idx", ddp_rank) if resume_state_dict else ddp_rank
    resume_epoch = resume_state_dict.get("epoch", 0) if resume_state_dict else 0
    resume_tokens_in_epoch = resume_state_dict.get("tokens_in_epoch", 0) if resume_state_dict else 0
    first_pass = True

    current_epoch = resume_epoch
    tokens_in_current_epoch = resume_tokens_in_epoch

    def packed_documents():
        nonlocal first_pass, current_epoch
        while True:
            start_idx = resume_file_idx if first_pass else ddp_rank
            first_pass = False
            yield from protein_sequences_packed_iter(
                split,
                start=start_idx,
                step=ddp_world_size,
                max_length=max_seq_length,
                epoch=current_epoch,
                seed=seed,
            )
            # All files for this epoch exhausted
            current_epoch += 1
            print(f"[Protein Dataloader] Epoch {current_epoch - 1} completed for rank {ddp_rank}")

    documents = packed_documents()
    needed_tokens = B * T + 1
    token_buffer = deque()
    position_buffer = deque()

    while True:
        while len(token_buffer) < needed_tokens:
            doc, file_idx = next(documents)
            for seq_tokens in doc:
                tokens = [bos_token_id] + seq_tokens + [eos_token_id]
                position_buffer.extend(range(len(tokens)))
                token_buffer.extend(tokens)
                tokens_in_current_epoch += len(tokens)

        tokens_list = [token_buffer.popleft() for _ in range(needed_tokens)]
        use_cuda = device == "cuda"
        scratch = torch.tensor(tokens_list, dtype=torch.long, pin_memory=use_cuda)
        position_ids_cpu = torch.tensor(
            [position_buffer.popleft() for _ in range(needed_tokens)],
            dtype=torch.long, pin_memory=use_cuda,
        )[:-1]

        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        position_ids = position_ids_cpu.view(B, T).to(device=device, non_blocking=use_cuda)

        state_dict = {
            "file_idx": file_idx,
            "epoch": current_epoch,
            "tokens_in_epoch": tokens_in_current_epoch,
        }
        yield inputs, targets, position_ids, state_dict


def protein_dataloader(B, T, split, **kwargs):
    """Protein dataloader without state dict."""
    for inputs, targets, position_ids, _ in protein_dataloader_with_state(B, T, split, **kwargs):
        yield inputs, targets, position_ids
