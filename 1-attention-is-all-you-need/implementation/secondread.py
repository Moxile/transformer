import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

        def forward(self, x: torch.Tensor, mask: torch.Tensor=None, encoder: torch.Tensor=None):
                q = x.unsqueeze(1) @ self.wq
                if encoder is not None:
                        k = encoder.unsqueeze(1) @ self.wk
                        v = encoder.unsqueeze(1) @ self.wv
                else:
                        k = x.unsqueeze(1) @ self.wk
                        v = x.unsqueeze(1) @ self.wv

                # attention calculation
                attention = q @ k.transpose(2, 3)
                attention /= torch.sqrt(torch.tensor(self.d_k))

                # masking
                mask = mask.unsqueeze(1)
                attention = attention.masked_fill(mask == 0, -1e9)

                attention = torch.softmax(attention, 3)
                attention = attention @ v

                # concanating heads
                attention = attention.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], self.d_model)

                return attention @ self.wl + self.wb


        def __init__(self, seq_len=512, d_model=512, heads=6, batch=1):
          super().__init__()

          assert d_model % heads == 0

          self.d_k = d_model//heads
          self.d_model = d_model
          self.batch = batch

          # k, q, v
          self.wq = nn.Parameter(torch.rand(heads, d_model, self.d_k))
          self.wk = nn.Parameter(torch.rand(heads, d_model, self.d_k))
          self.wv = nn.Parameter(torch.rand(heads, d_model, self.d_k))
          # linear layer
          self.wl = nn.Parameter(torch.rand(d_model, d_model))
          self.wb = nn.Parameter(torch.rand(d_model))

class Encoder(nn.Module):

        def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
                attention = self.attention(x, mask)
                attention = self.norm1(x + self.dropout1(attention))
                ff = self.ff1(attention)
                ff = self.relu(ff)
                ff = self.ff2(ff)
                return self.norm2(attention + self.dropout2(ff))

        def __init__(self, seq_len=512, d_model=512, heads=6, batch=1, p_drop=0.1):
               super().__init__()
               self.attention = MultiHeadAttention(seq_len, d_model, heads, batch)
               self.norm1 = nn.LayerNorm(d_model)
               self.dropout1 = nn.Dropout(p_drop)
               self.ff1 = nn.Linear(d_model, 4*d_model)
               self.relu = nn.ReLU()
               self.ff2 = nn.Linear(4*d_model, d_model)
               self.norm2 = nn.LayerNorm(d_model)
               self.dropout2 = nn.Dropout(p_drop)


class Decoder(nn.Module):

        def forward(self, x: torch.Tensor, mask: torch.Tensor, encoder_mask: torch.Tensor, encoder: torch.Tensor):
               attention = self.attention(x, Transformer.merge_tri_mask(mask))
               attention = self.norm1(x + self.dropout1(attention))
               cross = self.crossattention(attention, encoder_mask, encoder)
               attention = self.norm2(attention + self.dropout2(cross))
               ff = self.ff1(attention)
               ff = self.relu(ff)
               ff = self.ff2(ff)
               return self.norm3(attention + self.dropout3(ff))

        def __init__(self, seq_len=512, d_model=512, heads=6, batch=1, p_drop=0.1):
              super().__init__()
              self.attention = MultiHeadAttention(seq_len, d_model, heads, batch)
              self.norm1 = nn.LayerNorm(d_model)
              self.dropout1 = nn.Dropout(p_drop)
              self.crossattention = MultiHeadAttention(seq_len, d_model, heads, batch)
              self.norm2 = nn.LayerNorm(d_model)
              self.dropout2 = nn.Dropout(p_drop)
              self.ff1 = nn.Linear(d_model, 4*d_model)
              self.relu = nn.ReLU()
              self.ff2 = nn.Linear(4*d_model, d_model)
              self.norm3 = nn.LayerNorm(d_model)
              self.dropout3 = nn.Dropout(p_drop)


class Transformer(nn.Module):

        @staticmethod
        def merge_tri_mask(mask: torch.Tensor, seq_len=512):
                tri = torch.tril(torch.ones(seq_len, seq_len))
                return mask * tri

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

        def forward(self, x: torch.Tensor, output: torch.Tensor):
               src_mask = (x != self.pad_token_id).unsqueeze(1)
               output_mask = (output != self.pad_token_id).unsqueeze(1)

               src = self.input_embedding(x) * torch.sqrt(torch.tensor(self.d_model))
               tgt = self.output_embedding(output) * torch.sqrt(torch.tensor(self.d_model))

               src = self.dropout_src(src + self.positional_encoding.unsqueeze(0))
               tgt = self.dropout_tgt(tgt + self.positional_encoding.unsqueeze(0))

               encoding = src
               for encoder in self.encoders:
                      encoding = encoder(encoding, src_mask)

               out = tgt
               for decoder in self.decoders:
                      out = decoder(out, output_mask, src_mask, encoding)

               out = self.ff(out)
               return self.softmax(out) 
                

               
        def __init__(self, seq_len=512, d_model=512, layers=6, heads=6, batch=1, num_embeddings_input=1000, num_embeddings_output=1000, pad_token_id = 0, p_drop=0.1):
                super().__init__()
                self.pad_token_id = pad_token_id
                self.d_model = d_model

                self.register_buffer('positional_encoding', self._generate_positional_encoding(seq_len, d_model))
                self.input_embedding = nn.Embedding(num_embeddings_input, d_model)
                self.output_embedding = nn.Embedding(num_embeddings_output, d_model)
                self.dropout_src = nn.Dropout(p_drop)
                self.dropout_tgt = nn.Dropout(p_drop)

                self.encoders = nn.ModuleList([Encoder(seq_len, d_model, heads, batch, p_drop) for _ in range(layers)])
                self.decoders = nn.ModuleList([Decoder(seq_len, d_model, heads, batch, p_drop) for _ in range(layers)])

                self.ff = nn.Linear(d_model, num_embeddings_output)

                self.ff.weight = self.output_embedding.weight
                if num_embeddings_input == num_embeddings_output:
                    self.input_embedding.weight = self.output_embedding.weight
                    
                self.softmax = nn.Softmax(2)



model = Transformer(seq_len=512, d_model=64, layers=2, heads=4, num_embeddings_input=100, num_embeddings_output=100)
x = torch.randint(0, 100, (2, 512))
output = torch.randint(0, 100, (2, 512))
result = model(x, output)
print(result.shape)
print(sum(result[0][0]))
