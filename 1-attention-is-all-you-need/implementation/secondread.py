import torch
import torch.nn as nn



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
    
    def __init__(self, seq_len=512, d_model=512, num_heads=6, num_embeddings=1024, num_blocks=6):
        super().__init__()
        self.register_buffer('positional_encoding', self._generate_positional_encoding(seq_len, d_model))





print(torch.arange(0, 512, 2).size())
print(torch.arange(0, 512, 2).unsqueeze(0).size())