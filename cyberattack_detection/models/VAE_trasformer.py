from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from torch.nn import TransformerDecoder
import torch
import torch.nn as nn
import math

from utils.positional_encoding import PositionalEncoding



class VAE_Transformer(nn.Module):
    def __init__(self, feats, lr, window_size, latent_dim):
        super(VAE_Transformer, self).__init__()
        self.name = 'VAE_Transformer'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = window_size
        self.scale = 16
        self.latent_dim = latent_dim

        self.linear_layer_enc = nn.Linear(feats, self.scale*feats)
        self.linear_layer_dec = nn.Linear(latent_dim, self.scale*feats)
        self.output_layer = nn.Linear(self.scale*feats, feats)
        self.pos_encoder = PositionalEncoding(self.scale*feats, 0.1, self.n_window, batch_first=True)

        encoder_layers = TransformerEncoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        decoder_layers = TransformerDecoderLayer(d_model=feats*self.scale, nhead=feats, batch_first=True, dim_feedforward=256, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)

        self.fc_mu = nn.Linear(feats*self.scale, latent_dim)
        self.fc_logvar = nn.Linear(feats*self.scale, latent_dim)
        self.fcn = nn.Sigmoid()

        self.prior = torch.distributions.MultivariateNormal(
                torch.zeros(latent_dim).to('cuda'), torch.eye(latent_dim).to('cuda')
            )
        self.criterion = nn.MSELoss(reduction=None)

    def encode(self, x):
        x = x * math.sqrt(self.n_feats)
        x = self.linear_layer_enc(x)
        x = self.pos_encoder(x)
        memory = self.transformer_encoder(x)
        #memory = memory.mean(dim=1)  # Aggregate memory
        mu = self.fc_mu(memory)
        logvar = self.fc_logvar(memory)
        return mu, logvar,memory

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z,memory):
        z = self.linear_layer_dec(z)
        #z = self.pos_encoder(z)
        x = self.transformer_decoder(z, z)
        x = self.output_layer(x)
        x = self.fcn(x)
        return x

    def forward(self, src, tgt):
        mu, logvar,memory = self.encode(src)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        recon_tgt = self.decode(z,z)
        return recon_tgt, mu, logvar

    def loss_function(self, recon_tgt, tgt, mu, logvar):
        l = nn.MSELoss(reduction = 'none')
        BCE = l(recon_tgt, tgt)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD