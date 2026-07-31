import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

        def forward(self, x: torch.Tensor):
                q = x @ self.wq
                k = x @ self.wk
                v = x @ self.wv

                # attention calculation
                attention = q @ k.transpose(1, 2)
                attention /= torch.sqrt(torch.tensor(self.d_k))
                attention = torch.softmax(attention, 2)
                attention = attention @ v

                attention = attention.transpose(0, 1).contiguous().reshape(x.shape[0], self.d_model)

                return attention @ self.wl + self.wb


        def __init__(self, seq_len=512, d_model=512, heads=6):
          super().__init__()
          self.d_k = d_model//heads
          self.d_model = d_model
          self.wq = nn.Parameter(torch.rand(heads, d_model, self.d_k))
          self.wk = nn.Parameter(torch.rand(heads, d_model, self.d_k))
          self.wv = nn.Parameter(torch.rand(heads, d_model, self.d_k))
          self.wl = nn.Parameter(torch.rand(d_model, d_model))
          self.wb = nn.Parameter(torch.rand(d_model))

class Transformer(nn.Module):

        def _generate_positional_encoding(self, seq_len=512, d_model=512):
                # Formula: sin for even, cos for odd -> 10000^(2i/d)

                dimension = torch.arange(0, d_model, 2).unsqueeze(0)
                position = torch.arange(0, seq_len).unsqueeze(1)
    
                div_term = torch.exp(-dimension/d_model * torch.log(torch.tensor(10000.0)))
                sin = torch.sin(position*div_term)
                cos = torch.cos(position*div_term)
    
                positional_encoding = torch.zeros(seq_len, d_model)
                positional_encoding[:, 0::2] = sin
                positional_encoding[:, 1::2] = cos
    
                return positional_encoding
    
        def __init__(self, seq_len=512, d_model=512):
                super().__init__()
                self.register_buffer('positional_encoding', self._generate_positional_encoding(seq_len, d_model))