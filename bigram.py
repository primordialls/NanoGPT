import torch
import torch.nn as nn
from torch import tensor as tt
from torch.nn import functional as F
from engine import BigramLanguageModel

torch.manual_seed(1337)

#hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4  
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

with open("input.txt","r",encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda x: [stoi[i] for i in x]
decode = lambda y: ''.join(itos[i] for i in y)

data = tt(encode(text),dtype = torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    
    #every once in a while evaluate loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print (f"step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")

    #sample batch of data
    xb,yb = get_batch("train")
    
    #evaluate loss
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())

context = torch.zeros((1,1),dtype=torch.long,device=device)

print(decode(m.generate(idx = context,max_new_tokens=500)[0].tolist()))