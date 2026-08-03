import torch
import torch.nn as nn





def generate_rope_encoding(seq_len=8, d_k=8):
                # O_i = 10000^2i/d base frequency with 10000 changes how quick clocks begin to rotate, d is d_model and i is the index in the dimension
                # The rotation matrix becomes [cos m*O_i -sin m*O_i]
                #                             [sin m*O_i  cos m*O_i]
                # m is the absolute position in the sequence

                dimensions = torch.arange(0, d_k//2)
                positions = torch.arange(0, seq_len, dtype=torch.float32)
                O = torch.exp(-2*dimensions/d_k * torch.log(torch.tensor(10000)))

                matrix = positions.unsqueeze(-1) @ O.unsqueeze(0)

                cos = torch.cos(matrix)
                sin = torch.sin(matrix)

                #return torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
                return sin, cos


generate_rope_encoding()