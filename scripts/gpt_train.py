import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

block_size = 128 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
dropout = 0.2

torch.manual_seed(1337)
start_time = time.time()
with open('input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):       
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

param_settings = {
    'param1': {
        'batch_size':32,
        'n_embd': 128,
        'n_head': 4,
        'n_layer': 4
    },
    'param2': {
        'batch_size':64,
        'n_embd': 192,
        'n_head': 6,
        'n_layer': 2
    },
    'param3': {
        'batch_size':32,
        'n_embd': 96,
        'n_head': 4,
        'n_layer': 4
    }
}

start_time = time.time()
train_loss = {}
val_loss = {}
results = []

for param_id, param in param_settings.items():
    print(f"Trial: {param_id}")
    print(f"Training with parameters: {param}")

    batch_size = param['batch_size']
    n_embd = param['n_embd']
    n_head = param['n_head']
    n_layer = param['n_layer']

    model = GPTLanguageModel()
    m = model.to(device)
    num_params = sum(p.numel() for p in m.parameters())/1e6
    print(num_params, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    training_losses = []
    validation_losses = []
    param_start_time = time.time()

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            training_losses.append((iter, losses['train'].item()))
            validation_losses.append((iter, losses['val'].item()))
            elapsed = time.time() - start_time
            eta = (elapsed / (iter + 1)) * (max_iters - iter - 1) if iter > 0 else 0

            print(f"step {iter:4d}: train {losses['train']:.4f}, val {losses['val']:.4f} | "
                  f"{timedelta(seconds=int(elapsed))} / ETA {timedelta(seconds=int(eta))}")

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    train_loss[param_id] = training_losses
    val_loss[param_id] = validation_losses
    
    final_train_loss = training_losses[-1][1] if training_losses else None
    final_val_loss = validation_losses[-1][1] if validation_losses else None
    training_time = time.time() - param_start_time
    
    results.append({
        'Parameter Setting': param_id,
        'Batch Size': param['batch_size'],
        'Embedding Dim': param['n_embd'],
        'Num Heads': param['n_head'],
        'Num Layers': param['n_layer'],
        'Num Parameters (M)': round(num_params, 2),
        'Final Train Loss': round(final_train_loss, 4) if final_train_loss else None,
        'Final Val Loss': round(final_val_loss, 4) if final_val_loss else None,
        'Training Time (s)': round(training_time, 2)
    })

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
for p in param_settings.keys():
    if p in train_loss:
        i, l = zip(*train_loss[p])
        plt.plot(i, l, label=f'{p} (Train)', marker='o', markersize=3)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for p in param_settings.keys():
    if p in val_loss:
        i, l = zip(*val_loss[p])
        plt.plot(i, l, label=f'{p} (Val)', marker='s', markersize=3)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Validation Loss Over Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

comparison_df = pd.DataFrame(results)
print("\nComparison results:\n")
print(comparison_df.to_string(index=False))
comparison_df.to_csv('comparison_results.csv', index=False)

