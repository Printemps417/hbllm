import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#hyperparameters
d_model = 512
context_length = 20
num_heads = 8
head_dim = d_model // num_heads
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dropout_rate = 0.1
num_blocks = 12

print(f"Using device: {device}")

class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.linear2 = nn.Linear(4*d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model, head_dim, bias=False)
        self.Wk = nn.Linear(d_model, head_dim, bias=False)
        self.Wv = nn.Linear(d_model, head_dim, bias=False)
        self.register_buffer("mask",torch.tril(torch.ones(context_length,context_length)).to(device))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # batch size, time steps(context length), embedding dimension
        B,T,D = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        out = (q @ k.transpose(-2,-1))/math.sqrt(head_dim)
        out = out.masked_fill(self.mask[:T,:T]==0,float('-inf'))
        out= F.softmax(out, dim=-1)
        out= self.dropout(out)
        out = out @ v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x): # x is (B,T,D)
        output = torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,D)
        output = self.wo(output)
        output = self.dropout(output)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForwardNetwork()

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class Model(nn.Module):
    def __init__(self,max_token_value=100256):
        super().__init__()
        self.vocab_linear = nn.Linear(d_model,max_token_value) #last layer to vocab size
        self.te_table = nn.Embedding(max_token_value, d_model) # token embedding table
        self.transformer_blocks =nn.Sequential(
            *([TransformerBlock() for _ in range(num_blocks)]+nn.LayerNorm(d_model))
        )

    def forward(self, x_batch, y_batch=None):
        B,T,D = x_batch.shape
        pe_table = torch.zeros((context_length,d_model),device=device)
        position = torch.arange(0,context_length,detype=torch.float,device=device).unsqueeze(1)

        div_term = torch.exp(-math.log(10000.0)*torch.arange(0,d_model,2,device=device).float()/d_model)
        pe_table[:,0::2] = torch.sin(position*div_term)
        pe_table[:,1::2] = torch.cos(position*div_term)

        output = self.te_table(x_batch)+pe_table[:T,:]
        output = self.transformer_blocks(output)
        logits = self.vocab_linear(output)
        if y_batch is not None:
            B,T,D = logits.shape
            logits = logits.view(B*T,D)
            y_batch = y_batch.view(B*T)

            loss = F.cross_entropy(logits,y_batch)
        else:
            loss = None
        return logits, loss

