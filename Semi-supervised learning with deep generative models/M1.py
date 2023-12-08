import torch
import torch.nn as nn
import torch.nn.functional as F

"""q(z|x;phi)"""
class M1Encoder(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size_qz : list, 
                 latent_size):
        
        super(M1Encoder, self).__init__()
        self.layers = nn.ModuleList()
        prev_h = input_size
        for h in hidden_size_qz:
            self.layers.append(nn.Linear(prev_h, h))
            prev_h = h
        self.mu_layer = nn.Linear(prev_h, latent_size)
        self.logvar_layer = nn.Linear(prev_h, latent_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.softplus(layer(x)) # softplus 사용
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

"""p(x|z:theta)"""
class M1Decoder(nn.Module):
    def __init__(self, 
                 latent_size,
                 hidden_size_px : list,
                 output_size):
        
        super(M1Decoder, self).__init__()
        self.layers = nn.ModuleList()
        prev_h = latent_size
        for h in hidden_size_px:
            self.layers.append(nn.Linear(prev_h, h))
            prev_h = h
        self.reconstruction_layer = nn.Linear(prev_h, output_size)

    def forward(self, z):
        for layer in self.layers:
            z = F.softplus(layer(z))
        reconstruction = torch.sigmoid(self.reconstruction_layer(z)) 
        return reconstruction

class M1model(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size_qz : list, 
                 latent_size,
                 hidden_size_px : list):
        
        super(M1model, self).__init__()
        self.encoder = M1Encoder(input_size, hidden_size_qz, latent_size)
        self.decoder = M1Decoder(latent_size, hidden_size_px, input_size)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z, mu, logvar = self._reparameterize(mu, logvar) 
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def _reparameterize(self, mu, logvar):        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # eps ~ N(0,I)
        z = mu + eps * std 
        return z, mu, logvar

    def _loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.BCELoss(reduction = 'sum')(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        return loss