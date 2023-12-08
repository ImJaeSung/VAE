import torch
import torch.nn as nn

class prob_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(prob_encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh()
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.sigma = nn.Linear(hidden_size, latent_size)
    
    def forward(self, x):
        x = self.fc1(x)
        mu = self.mu(x)
        log_sigma2 = self.sigma(x) # log(sigma^2) ; sigma must be positive

        return mu, log_sigma2

class prob_decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(prob_decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc1(z)
        pred = self.fc2(z)

        return pred

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = prob_encoder(self.input_dim, self.hidden_dim, self.latent_dim)
        self.decoder = prob_decoder(self.latent_dim, self.hidden_dim, self.input_dim) # symmetric 
    
    def forward(self, x):
        '''recognition model : q(z|x)'''
        mu, log_sigma2 = self.encoder(x)
        z = self.reparameter(mu, log_sigma2)

        '''generative mode   l : p(x|z)'''
        pred = self.decoder(z)

        return mu, log_sigma2, pred
      
    def reparameter(self, mu, log_sigma2):
        std = torch.exp(0.5*log_sigma2) # sigma = 0.5*exp(log(sigma^2))    
        eps = torch.randn_like(std, dtype = torch.float32) # eps ~ N(0, I)
        z_ = mu + std*eps
        # z = torch.tensor(z, dtype = torch.float32)

        return z_