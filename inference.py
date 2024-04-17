import torch
import tiktoken
import json
from model import GPT, GPTConfig

device = "cuda"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
start = "kamoo" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"

checkpoint = torch.load("./out/ckpt.pt", map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
model = torch.compile(model)

ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)

# enc = tiktoken.get_encoding("gpt2")
# encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
# decode = lambda l: enc.decode(l)
with open('out/meta.json') as f:
    meta = json.load(f)
itos = meta["itos"]
stoi = meta["stoi"]
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[str(i)] for i in l]) # decoder: take a list of integers, output a string
# def encode(text):
#     return [vocab.find(c) for c in text]
# def decode(text_indicies):
#     return "".join([vocab[text_index] for text_index in text_indicies])
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')


