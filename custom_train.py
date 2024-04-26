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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# dataset
df = pd.read_csv("./chat.csv")
# data = "\n".join(df[df["from"] == "martin"]["messages"].to_list())
data = "\n".join(df["messages"].to_list())
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)}\nvocab size: {vocab_size:,}")

# create a mapping from characters to integers
str_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_str = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [str_to_int[c] for c in s]  # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([int_to_str[i] for i in l]) # decoder: take a list of integers, output a string
with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
    json.dump({
        'vocab_size': vocab_size,
        'itos': int_to_str,
        'stoi': str_to_int,
    }, f)

# training
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.text = text
        self.block_size = block_size

    def __len__(self):
        return len(self.text) - self.block_size

    def __getitem__(self, idx):
        inputs = np.array(encode(self.text[idx:idx+self.block_size])).astype(np.int64)
        targets = np.array(encode(self.text[idx+1:idx+self.block_size+1])).astype(np.int64)
        inputs = torch.from_numpy(inputs).pin_memory().to(device, non_blocking=True)
        targets = torch.from_numpy(targets).pin_memory().to(device, non_blocking=True)
        return inputs, targets

def create_batches(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Split dataset into train and validation sets
# train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)  # todo - make this custom, based on sentences
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_dataset = TextDataset(train_data, block_size)
val_dataset = TextDataset(val_data, block_size)

# Create batches for train and validation sets
train_dataloader = create_batches(train_dataset, batch_size, shuffle=False)
val_dataloader = create_batches(val_dataset, batch_size, shuffle=False)

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


# Function to generate text samples
@torch.no_grad()
def generate_samples(model, prompt, max_length=100, temperature=1.0):
    model.eval()
    prompt_encoded = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    new_tokens = []
    for _ in range(max_length):
        logits, loss = model(prompt_encoded)
        next_token_logits = logits[0, -1, :] / temperature
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        prompt_encoded = torch.cat((prompt_encoded, next_token.unsqueeze(0)), dim=1)
        new_tokens.append(next_token.unsqueeze(0))
    return decode(torch.cat(new_tokens).squeeze().tolist())


@torch.no_grad()
def estimate_train_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for evaluation_index in range(eval_iters):
        inputs, targets = next(iter(train_dataloader))
        with ctx:
            logits, loss = model(inputs, targets)
        losses[evaluation_index] = loss.item()
    mean_loss = losses.mean()
    model.train()
    return mean_loss


@torch.no_grad()
def estimate_validation_loss():
    model.eval()
    losses = torch.zeros(len(val_dataloader))
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
        with ctx:
            logits, loss = model(inputs, targets)
        losses[batch_idx] = loss.item()
    sample_tokens = inputs[0].cpu().detach().numpy()
    input_prompt = decode(sample_tokens).replace('\n',' ')
    generated_text = generate_samples(model, input_prompt[:200])
    print(f"input: {input_prompt[:200]}\noutput: {generated_text}")
    mean_loss = losses.mean()
    model.train()
    return mean_loss

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


epochs = 20
running_mfu = -1.0
with SummaryWriter(log_dir) as writer:
    for epoch_idx in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            start_of_iter = time.time()
            # inputs = inputs.pin_memory().to(device, non_blocking=True)
            # targets = targets.pin_memory().to(device, non_blocking=True)
            iter_num = (epoch_idx + 1) * (batch_idx + 1)
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if iter_num % eval_interval == 0:
                train_loss = estimate_train_loss()
                print(f"step {iter_num}: train loss {train_loss:.4f}")
                writer.add_scalar("Loss/train", train_loss, iter_num)
                writer.add_scalar("Learning_rate", lr, iter_num)
                writer.add_scalar("MFU", running_mfu*100, iter_num)

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                with ctx:
                    logits, loss = model(inputs, targets)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
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
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            delta_time_of_iter = time.time() - start_of_iter
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, delta_time_of_iter)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {delta_time_of_iter:.0f}s, mfu {running_mfu*100:.2f}%")
        
        # eval
        start_of_val_eval = time.time()
        validation_loss = estimate_validation_loss()
        validation_delta_time = time.time() - start_of_val_eval
        print(f"epoch {epoch_idx}: validation loss {validation_loss:.4f}, time {validation_delta_time:.0f}")
        writer.add_scalar("Loss/validation", validation_loss, iter_num)
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
