# predict.py

import torch
from torch import nn

import numpy as np


class Transformer(nn.Module):
  def __init__(self, num_head, model_dim, max_len, num_layer, pre_len):
    super().__init__()
    self.num_head = num_head
    self.model_dim = model_dim
    self.max_len = max_len
    self.num_layer = num_layer
    self.pre_len = pre_len

    self.input_linear = nn.Linear(self.model_dim, self.model_dim)

    self.pos_encode = SinCosPosEncoding(self.model_dim, self.max_len)

    encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.num_head, batch_first=True)

    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layer)

    self.output_layer = nn.Sequential(nn.Linear(self.model_dim, 64), nn.ReLU(), nn.Linear(64, self.pre_len))

  def forward(self, src):
    src = self.input_linear(src)

    src = self.pos_encode(src)

    memory = self.transformer_encoder(src)

    last_step = memory[:, -1, :]

    output = self.output_layer(last_step)  # [batch, pred_length]
    return output


class SinCosPosEncoding(nn.Module):
  def __init__(self, model_dim, max_len):
    super(SinCosPosEncoding, self).__init__()

    self.model_dim = model_dim
    self.max_len = max_len

    pe = torch.zeros(self.max_len, self.model_dim)
    # position[[ 0.], [ 1.], [ 2.], [ 3.], ... [ 999.]]
    position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, self.model_dim, 2).float() * (-np.log(10000) / self.model_dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: (self.model_dim // 2)])
    # if self.model_dim % 2 == 0:
    #     pe[:, 1::2] = torch.cos(position * div_term[:, :self.model_dim // 2])
    # else:
    #     pe[:, 1::2] = torch.cos(position * div_term[:, :(self.model_dim - 1)// 2])

    self.register_buffer("pe", pe.unsqueeze(0))

  def forward(self, x) -> torch.Tensor:
    return x + self.pe[:, : x.size(1)]


# predict.py ends here
