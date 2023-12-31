import torch
import torch.nn as nn
from torch import tensor as tt
from torch.nn import functional as F

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

#--------

with open("mastersuitsnotitles.txt","r",encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda x: [stoi[i] for i in x]
decode = lambda y: ''.join(itos[i] for i in y)


class Head(nn.Module):
    """One head of self attention"""

    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) # buffer, not trained

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores (affinities)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B,T,C)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # (B,T,C)
        wei = F.softmax(wei,dim = -1) # (B,T,T)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ simple linear layer followed by relu """

    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd), # projection layer
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self,n_embd,n_head):
        #n_embd: embedding dimension, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x)) # add "x + ..."for residual implementation
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd,n_head=n_head),
            Block(n_embd,n_head=n_head),
            Block(n_embd,n_head=n_head),
            Block(n_embd,n_head=n_head),
            Block(n_embd,n_head=n_head),
            Block(n_embd,n_head=n_head),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_embd = self.token_embedding_table(idx) #(B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
        x = tok_embd + pos_embd #(B,T,C)
        x = self.blocks(x)
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
            #crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            #get predictions:
            logits,loss = self(idx_cond)
            #focus only on last time step:
            logits = logits[:,-1,:]
            #apply softmax to get probs
            probs = F.softmax(logits,dim = 1)
            #sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
            #append sampled index to running sequence
            idx = torch.cat((idx,idx_next),dim=1)
            print(decode([idx[-1,-1].item()]),end="")
        return idx