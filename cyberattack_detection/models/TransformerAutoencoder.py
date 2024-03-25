import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.positional_encoding import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import TransformerDecoder


class TransformerAutoencoder(nn.Module):
	def __init__(self, feats):
		super().__init__()
		self.name = 'TransformerAutoencoder'
		self.lr = 0.01
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10

		self.lin = nn.Linear(1, feats)
		self.out_lin = nn.Linear(feats, 1)
		self.pos_encoder = PositionalEncoding(feats, 0.1, feats*self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		# bs x (ws x features) x features
		src = src * math.sqrt(self.n_feats)
		src = self.lin(src.unsqueeze(2))
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)

		tgt = tgt * math.sqrt(self.n_feats)
		tgt = self.lin(tgt.unsqueeze(2))
		tgt = self.pos_encoder(tgt)
		x = self.transformer_decoder(tgt, memory)
		x = self.out_lin(x)
		x = self.fcn(x)
		return x