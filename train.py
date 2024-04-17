import os
import time
import math
import json
import pickle
from contextlib import nullcontext
import pandas as pd
import tiktoken
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter for TensorBoard logging

from model import GPTConfig, GPT

device = "cuda"
block_size = 1024
batch_size = 8  # 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
n_layer =  24  # 12
n_head = 16  # 12
n_embd = 1024  # 768
dropout = 0.1  # 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
eval_iters = 200
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
decay_lr = True # whether to decay the learning rate
eval_interval = 100 #2000
gradient_accumulation_steps = 1  # 5 * 8 # used to simulate larger batch sizes
out_dir = "out"
log_dir = "logs"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# data = requests.get(data_url).text
# training_data = data[:int(len(data)*0.9)]
# validation_data = data[int(len(data)*0.9):]
# encoding = tiktoken.get_encoding("gpt2")
# train_indicies = np.array(encoding.encode_ordinary(training_data), dtype=np.uint16)
# validation_indicies = np.array(encoding.encode_ordinary(validation_data), dtype=np.uint16)

# dataset
df = pd.read_csv("./chat.csv")
data = "\n".join(df[df["from"] == "martin"]["messages"].to_list())
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)}\nvocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
    json.dump({
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }, f)

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
validation_data = data[int(n*0.9):]

# encode both to integers
train_indicies = encode(train_data)
validation_indicies = encode(validation_data)
print(f"train has {len(train_indicies):,} tokens")
print(f"val has {len(validation_indicies):,} tokens")

# export to bin files
train_indicies = np.array(train_indicies, dtype=np.uint16)
validation_indicies = np.array(validation_indicies, dtype=np.uint16)

# training
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


def get_batch(split):
    if split == "train":
        indicies = train_indicies
    else:
        indicies = validation_indicies
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    ix = torch.randint(len(indicies) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((indicies[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((indicies[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

vocab_size = vocab_size
# vocab_size = 50304  # 50304  # defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)
# vocab_size = np.unique(np.concatenate([train_indicies, validation_indicies])).shape[0]
# vocab_size = np.concatenate([train_indicies, validation_indicies]).max()

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    dropout=dropout,
    vocab_size=vocab_size
)

ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)
scaler = torch.cuda.amp.GradScaler(enabled=False)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type=device)
print("compile model")
model = torch.compile(model)


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
with SummaryWriter(log_dir) as writer:
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            writer.add_scalar("Loss/train", losses['train'], iter_num)
            writer.add_scalar("Loss/validation", losses['val'], iter_num)
            writer.add_scalar("Learning_rate", lr, iter_num)
            writer.add_scalar("MFU", running_mfu*100, iter_num)
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        # 'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt:.0f}s, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
# writer.close()
