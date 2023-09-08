from engine import BigramLanguageModel
import torch

with open("mastersuitsnotitles.txt","r",encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}
encode = lambda x: [stoi[i] for i in x]
decode = lambda y: ''.join(itos[i] for i in y)

print(vocab_size)

lm = BigramLanguageModel(vocab_size)
lm2 = lm.to(torch.device('cpu'))
lm2.load_state_dict(torch.load("params.pt",map_location=torch.device('cpu')))
context = torch.zeros((1,1),dtype=torch.long)
lm.generate(idx = context,max_new_tokens=10000)