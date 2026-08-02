from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset



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

          def per_head():
                 return nn.Parameter(torch.stack([
                        nn.init.xavier_uniform_(torch.empty(d_model, self.d_k)) for _ in range(heads)
                 ]))

          self.wq = per_head()
          self.wk = per_head()
          self.wv = per_head()
          # linear layer
          self.wl = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_model)))
          self.wb = nn.Parameter(torch.zeros(d_model))

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
        def merge_tri_mask(mask: torch.Tensor):
                seq_len = mask.shape[-1]
                tri = torch.tril(torch.ones(seq_len, seq_len, device=mask.device))
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

               src = self.dropout_src(src + self.positional_encoding[:x.shape[1]].unsqueeze(0))
               tgt = self.dropout_tgt(tgt + self.positional_encoding[:output.shape[1]].unsqueeze(0))

               encoding = src
               for encoder in self.encoders:
                      encoding = encoder(encoding, src_mask)

               out = tgt
               for decoder in self.decoders:
                      out = decoder(out, output_mask, src_mask, encoding)

               out = self.ff(out)
               return out
                

               
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

                # embeddings get scaled by sqrt(d_model) and are tied to the output projection,
                # so the default N(0, 1) init blows the logits up to std ~14 and the loss to ~47
                nn.init.normal_(self.input_embedding.weight, std=d_model ** -0.5)
                nn.init.normal_(self.output_embedding.weight, std=d_model ** -0.5)

                self.ff.weight = self.output_embedding.weight
                if num_embeddings_input == num_embeddings_output:
                    self.input_embedding.weight = self.output_embedding.weight



class TranslationDataset(Dataset):
    # pairs: list of (src, tgt) tensors, each already padded/truncated to seq_len
    def __init__(self, pairs, bos_token_id, pad_token_id=0):
        self.pairs = pairs
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        # shift right: decoder sees <bos> + tgt[:-1], and predicts tgt itself one step ahead
        bos = torch.tensor([self.bos_token_id], dtype=tgt.dtype)
        decoder_input = torch.cat([bos, tgt[:-1]])
        loss_target = tgt

        return src, decoder_input, loss_target


def lr_lambda(step, d_model, warmup_steps=4000):
       step = max(step, 1)
       return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def get_device():
       if torch.cuda.is_available():
              return torch.device("cuda")
       if torch.backends.mps.is_available():
              return torch.device("mps")
       return torch.device("cpu")


def train(model: Transformer, epochs: int, dataloader: DataLoader, optimizer: optim.Adam, scheduler: LambdaLR, pad_token_id=0, device=None):
       device = device or get_device()
       loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)
       model.train()
       for epoch in range(epochs):
              epoch_loss = 0.0
              for step, (src_batch, decoder_input_batch, loss_target_batch) in enumerate(dataloader):
                     src_batch = src_batch.to(device)
                     decoder_input_batch = decoder_input_batch.to(device)
                     loss_target_batch = loss_target_batch.to(device)

                     optimizer.zero_grad()
                     predictions = model(src_batch, decoder_input_batch)
                     loss = loss_fn(predictions.reshape(-1, predictions.shape[-1]), loss_target_batch.reshape(-1))
                     loss.backward()
                     optimizer.step()
                     scheduler.step()

                     epoch_loss += loss.item()
                     #print(f"epoch {epoch+1}/{epochs} step {step+1}/{len(dataloader)} "
                     #      f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}")

              print(f"epoch {epoch+1}/{epochs} done, avg loss {epoch_loss/len(dataloader):.4f}")


@torch.no_grad()
def predict(model: Transformer, src: torch.Tensor, bos_token_id: int, pad_token_id: int, seq_len: int, max_len=10, device=None):
       device = device or get_device()
       model.eval()
       src = src.unsqueeze(0).to(device)  # add batch dim

       decoder_input = torch.full((1, seq_len), pad_token_id, dtype=torch.long, device=device)
       decoder_input[0, 0] = bos_token_id

       generated = []
       for t in range(min(max_len, seq_len - 1)):
              logits = model(src, decoder_input)
              next_token = logits[0, t].argmax(-1).item()
              decoder_input[0, t + 1] = next_token
              generated.append(next_token)

       return generated


d_model = 64
seq_len = 32
pad_token_id = 0
bos_token_id = 1

device = get_device()
print(f"using device: {device}")

model = Transformer(seq_len=seq_len, d_model=d_model, layers=2, heads=4, num_embeddings_input=100, num_embeddings_output=100, pad_token_id=pad_token_id)
model.to(device)

# reverse-copy toy task: tgt is src reversed, so there's an actual pattern to learn
# tokens start at 2 so the toy data never collides with the pad / bos ids
pairs = []
for _ in range(20):
       src = torch.randint(2, 100, (seq_len,))
       tgt = src.flip(0)
       pairs.append((src, tgt))
dataset = TranslationDataset(pairs, bos_token_id=bos_token_id, pad_token_id=pad_token_id)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# LambdaLR *multiplies* this base lr by lr_lambda, and lr_lambda already is the Noam
# learning rate, so the base has to be 1.0 rather than Adam's 1e-3 default
optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = LambdaLR(optimizer, lr_lambda=partial(lr_lambda, d_model=d_model, warmup_steps=400))

train(model, epochs=200, dataloader=dataloader, optimizer=optimizer, scheduler=scheduler, pad_token_id=pad_token_id, device=device)

example_src = pairs[0][0]
prediction = predict(model, example_src, bos_token_id=bos_token_id, pad_token_id=pad_token_id, seq_len=seq_len, device=device)
print("target:    ", pairs[0][1][:len(prediction)].tolist())
print("prediction:", prediction)



