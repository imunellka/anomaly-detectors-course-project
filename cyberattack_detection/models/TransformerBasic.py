from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import TransformerDecoder
import torch
import torch.nn as nn
import math

from utils.positional_encoding import PositionalEncoding


class TransformerBasic(nn.Module):
	def __init__(self, feats, lr, window_size):
		super(TransformerBasic, self).__init__()
		self.name = 'TransformerBasic'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = window_size
		self.scale = 16
		self.linear_layer = nn.Linear(feats, self.scale*feats)
		self.output_layer = nn.Linear(self.scale*feats, feats)
		self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)
		encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)

		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.linear_layer(src)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		#print('memory',memory)
		tgt = tgt * math.sqrt(self.n_feats)
		tgt = self.linear_layer(tgt)
		x = self.transformer_decoder(tgt, memory)
		x = self.output_layer(x)
		x = self.fcn(x)
		return x