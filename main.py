import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, seq_len=512, d_model=512, heads=6):
        super().__init__()
        self.d_k = d_model//heads
        self.Wq = nn.Linear(d_model, self.d_k)
        self.Wk = nn.Linear(d_model, self.d_k)
        self.Wv = nn.Linear(d_model, self.d_k)

    def forward(self, q, k, v, mask=None):
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        qk = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            qk = qk.masked_fill(mask, float('-inf'))
        qk = torch.softmax(qk, -1)
        qkv = torch.matmul(qk, v)

        return qkv

class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len=512, d_model=512, num_heads=6):
        super().__init__()

        self.heads = nn.ModuleList([Attention(seq_len, d_model, num_heads) for _ in range(num_heads)])
        self.conc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # add parallelizing logic later

        output = torch.cat([head(q, k, v, mask) for head in self.heads], -1)
        output = self.conc(output)

        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()    

        self.f1 = nn.Linear(d_model, 4*d_model)
        self.f2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        output = self.f1(x)
        output = torch.relu(output)
        output = self.f2(output)

        return output
    
class Encoder(nn.Module):
    def __init__(self, seq_len=512, d_model=512, num_heads=6):
        super().__init__()

        self.attention = MultiHeadAttention(seq_len, d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.attention(x, x, x)
        output = self.norm1(x + output)
        output_ = self.ff(output)
        output_ = self.norm2(output + output_)

        return output_
    
class Decoder(nn.Module):
    def __init__(self, seq_len=512, d_model=512, num_heads=6):
        super().__init__()

        self.attention = MultiHeadAttention(seq_len, d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.crossattention = MultiHeadAttention(seq_len, d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, k_encoder, v_encoder, mask):
        output = self.attention(x, x, x, mask)
        output = self.norm1(x+output)
        output_ = self.crossattention(output, k_encoder, v_encoder)
        output = self.norm2(output + output_)
        output_ = self.ff(output)
        output = self.norm3(output + output_)

        return output

class Transformer(nn.Module):

    def _generate_positional_encoding(self, seq_len=512, d_model=512):
        # Formula: sin for even, cos for odd -> 10000^(2i/d)

        # create constant positional embedding matrix
        dimension = torch.arange(0, d_model, 2).unsqueeze(0) # dimensions numbered as indices
        position = torch.arange(0, seq_len).unsqueeze(1) # positions numbered as indices

        # calculate the sin and cos part for the positional encoding
        div_term = torch.exp(-dimension/d_model * torch.log(torch.tensor(10000.0)))
        sin = torch.sin(position*div_term)
        cos = torch.cos(position*div_term)

        # create the positional encoding
        positional_encoding = torch.zeros(seq_len, d_model)
        positional_encoding[:, 0::2] = sin
        positional_encoding[:, 1::2] = cos

        return positional_encoding

    def __init__(self, seq_len=512, d_model=512, num_heads=6, num_embeddings=1024, num_blocks=6):
        super().__init__()
        self.register_buffer('positional_encoding', self._generate_positional_encoding(seq_len, d_model))
        self.num_blocks = num_blocks
        self.embedding = nn.Embedding(num_embeddings, d_model)
        self.encoders = nn.ModuleList([Encoder(seq_len, d_model, num_heads) for _ in range(num_blocks)])
        self.decoders = nn.ModuleList([Decoder(seq_len, d_model, num_heads) for _ in range(num_blocks)])

    def forward(self, x):
        output = self.embedding(x)
        encodings = []
        for encoder in self.encoders:
            output = encoder(output)
            encodings.append(output)

        return output
