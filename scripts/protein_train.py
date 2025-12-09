"""
Train protein language model. Run as:

python protein_train.py

For a quick test run:
python protein_train.py --depth=4 --max_seq_len=512 --device_batch_size=8 --total_batch_size=4096 --num_iterations=100 --eval_every=25

For distributed training:
torchrun --nproc_per_node=8 protein_train.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.protein_dataloader import protein_dataloader_with_state  # Use protein dataloader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.engine import Engine
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "protein_test" # wandb run name
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect)
# Model architecture
depth = 12 # smaller model for protein testing
max_seq_len = 1024 # max context length (proteins are shorter than text)
# Training horizon
num_iterations = 500 # number of training steps
# Optimization
device_batch_size = 16 # per-device batch size
total_batch_size = 65536 # total batch size in tokens
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
warmup_ratio = 0.05
warmdown_ratio = 0.2
final_lr_frac = 0.1
resume_from_step = -1
# Evaluation
eval_every = 50 # evaluate every N steps
eval_tokens = 524288 # tokens for validation
sample_every = 100 # sample every N steps
save_every = -1 # save checkpoints every N steps (-1 = only at end)
# Output
model_tag = "protein" # tag for checkpoint directory
# Allow CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# wandb logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-protein", name=run, config=user_config)

# Tokenizer - automatically uses ProteinTokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Protein Tokenizer Vocab size: {vocab_size}")
assert vocab_size == 25, f"Expected protein vocab size 25, got {vocab_size}"

# Model configuration
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"Model: {num_layers} layers, {model_dim} dim, {num_heads} heads")

# Batch size calculations
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")
print0(f"Total batch size: {total_batch_size:,} tokens")

# Initialize Model
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim
)

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# Resume if needed
base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, "protein_checkpoints", model_tag)
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print0(f"Parameters: {num_params:,}")

total_tokens = total_batch_size * num_iterations
print0(f"Training tokens: {total_tokens:,}")
print0(f"Tokens:Params ratio: {total_tokens / num_params:.2f}")

# Initialize Optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

# Initialize Protein DataLoaders
print0("Initializing protein dataloaders...")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = protein_dataloader_with_state(
    device_batch_size,
    max_seq_len,
    split="train",  # or "valid" for smaller test
    batch_size=128,
    max_seq_length=2048,
    device=device,
    resume_state_dict=dataloader_resume_state_dict
)
# Validation loader
def build_val_loader():
    from nanochat.protein_dataloader import protein_dataloader
    return protein_dataloader(
        device_batch_size,
        max_seq_len,
        split="valid",
        batch_size=128,
        max_seq_length=2048,
        device=device
    )

x, y, dataloader_state_dict = next(train_loader)
print0("First batch loaded successfully!")

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# Loop state
if not resuming:
    step = 0
    min_val_loss = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_loss = loop_state["min_val_loss"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Training loop
print0("\nStarting training loop...")
while True:
    last_step = step == num_iterations

    # Evaluation
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        val_loss = 0.0
        with torch.no_grad(), autocast_ctx:
            for _ in range(eval_steps):
                x_val, y_val = next(val_loader)
                loss = model(x_val, y_val)
                val_loss += loss.item()
        val_loss /= eval_steps
        print0(f"Step {step:05d} | Val loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log({"step": step, "val/loss": val_loss})
        model.train()

    # Sampling - simple protein sequence generation
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTYPYDVPDYAGG",  # Example protein sequence
            "MKTIIALSYIFCLVFA",  # Short sequence
        ]
        for prompt in prompts:
            # Encode prompt with BOS token
            tokens = [tokenizer.char_to_id['<cls>']] + tokenizer.encode(prompt)
            x_gen = torch.tensor([tokens], dtype=torch.long, device=device)
            
            # Simple greedy generation
            max_new_tokens = 30
            with torch.no_grad(), autocast_ctx:
                for _ in range(max_new_tokens):
                    # Forward pass (use only last max_seq_len tokens if sequence is too long)
                    x_input = x_gen if x_gen.size(1) <= max_seq_len else x_gen[:, -max_seq_len:]
                    logits = orig_model(x_input)
                    # Get next token (greedy sampling)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    # Append to sequence
                    x_gen = torch.cat([x_gen, next_token], dim=1)
            
            generated = tokenizer.decode(x_gen[0].tolist())
            print0(f"Input: {prompt[:30]}... → Generated: {generated[:80]}...")
        model.train()

    # Save checkpoint
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_loss": val_loss if 'val_loss' in locals() else 0,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    
    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    
    # Optimizer step
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    if step > 10:
        total_training_time += dt
    
    tok_per_sec = int(total_batch_size / dt)
    print0(f"step {step:05d}/{num_iterations:05d} | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,}")
    
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/tok_per_sec": tok_per_sec,
        })

    step += 1

print0(f"\nTraining complete!")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation loss: {min_val_loss:.4f}")
print0(f"Checkpoint saved to: {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
