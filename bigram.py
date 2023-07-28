import torch
import torch.nn as nn
from torch import tensor as tt
from torch.nn import functional as F

#hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

#--------
torch.manual_seed(1337)

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

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = {
        'train': train_data,
        'val': val_data,
    }[split]
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train","val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    
    def forward(self, idx, targets=None):
        B,T = idx.shape



        tok_embd = self.token_embedding_table(idx) #(B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
        x = tok_embd + pos_embd #(B,T,C)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) tensor of indices in current context
        for _ in range(max_new_tokens):
            #get predictions:
            logits,loss = self(idx)
            #focus only on last time step:
            logits = logits[:,-1,:]
            #apply softmax to get probs
            probs = F.softmax(logits,dim = 1)
            #sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
            #append sampled index to running sequence
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
            

model = BigramLanguageModel()
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