import torch
import torch.nn as nn
import torch.nn.functional as F

"""q(y|x;phi) = Cat(y|pi(x))"""
class Classifier(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size : list,
                 label_size):
        
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList()
        prev_h = input_size
        for h in hidden_size:
            self.layers.append(nn.Linear(prev_h, h))
            prev_h = h
        self.output_layer = nn.Linear(prev_h, label_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.softplus(layer(x))
        x = self.output_layer(x)
        return F.softmax(x)


"""q(z|x,y:phi)"""
class M2Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size : list,
                 latent_size):
        
        super(M2Encoder, self).__init__()
        self.layers = nn.ModuleList()
        prev_h = input_size
        for h in hidden_size:
            self.layers.append(nn.Linear(prev_h, h))
            prev_h = h
        self.mu_layer = nn.Linear(prev_h, latent_size)
        self.logvar_layer = nn.Linear(prev_h, latent_size)

    def forward(self, xy):
        for layer in self.layers:
            xy = F.softplus(layer(xy))
        mu = self.mu_layer(xy)
        logvar = F.softplus(self.logvar_layer(xy)) ####
        return mu, logvar


"""p(x|y,z;theta)"""
class M2Decoder(nn.Module):
    def __init__(self,
                 latent_size,
                 hidden_size : list,
                 output_size):
        
        super(M2Decoder, self).__init__()
        self.layers = nn.ModuleList()
        prev_h = latent_size
        for h in hidden_size:
            self.layers.append(nn.Linear(prev_h, h))
            prev_h = h
        self.reconstruction_layer = nn.Linear(prev_h, output_size)

    def forward(self, zy):
        for layer in self.layers:
            zy = F.softplus(layer(zy))
        reconstruction = torch.sigmoid(self.reconstruction_layer(zy))
        return reconstruction

class M2model(nn.Module):
    def __init__(self,
                 classifier_input_size, hidden_size_py : list, label_size,
                 input_size, hidden_size_qz : list, latent_size, hidden_size_px : list, output_size):
        
        super(M2model, self).__init__()
        
        # q(y|x;phi)
        self.classifier = Classifier(classifier_input_size,
                                     hidden_size_py,
                                     label_size)
        # q(z|x,y;phi)
        self.M2Encoder = M2Encoder(input_size + label_size,  
                                   hidden_size_qz,
                                   latent_size)
        # p(x|y,z;theta)
        self.M2Decoder = M2Decoder(latent_size + label_size, 
                                   hidden_size_px, 
                                   output_size)
        
        self.apply(self._init_weights)
    
    def forward(self, x, y):
        xy = torch.cat([x, y], dim = 1)   # dim_x(mnist = 784) + dim_y (mnist = 10)
        z_mu, z_logvar = self.M2Encoder(xy) # mu;z|x,y sigma;z|x,y

        z = self._reparameterize(z_mu, z_logvar) # z에 대해 reparmeter
        zy = torch.cat([z, y], dim = 1) # dim_z + dim_y(mnist = 10)
        
        x_recon = self.M2Decoder(zy) # p(x|y,z;theta)   
        
        return x_recon, z_mu, z_logvar
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.001)
            nn.init.constant_(module.bias, 0)
        
    def _expand_y(self, x, y_dim):
        batch_size = x.size(0)
        y = torch.zeros(batch_size * y_dim, y_dim)
        for i in range(y_dim):
            y[i * batch_size:(i + 1) * batch_size, i] = 1

        if x.is_cuda:
            y = y.cuda()

        return y.float()
    
    def _reparameterize(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + eps * std
    
    # p(y)
    def _prior(self, y): 
        prior = F.softmax(torch.ones_like(y), dim = 1) # batchx10
        prior.requires_grad = False

        Prior = y*torch.log(prior + 1e-8)
        Prior = torch.sum(Prior, dim = -1)

        return Prior   

    def _loss_function(self, x_recon, x, z_mu, z_logvar, y_hat_l, y_hat_ul, y, alpha = 0.1*3000):
        
        BCE = nn.BCELoss(reduction = 'none')(x_recon, x) # 64x784 / 640x784
        BCE = torch.sum(BCE, dim = -1) # 64x784 / 640x784 -> 64 / 640

        KLD = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        KLD = torch.sum(KLD, dim = -1) # 64x50 / 640x50 -> 64 / 640 

        Prior = self._prior(y) # 64x1 / 640
        
        L = BCE + KLD + Prior # 64x1 / 640

        if y_hat_l is not None:
            # L = torch.sum(L)
            # CE = nn.CrossEntropyLoss(reduction = 'sum')(y_hat_l, y) # E(-log q(y|x);p(x,y)) : classification loss term
            # y_hat_l = self.classifier(x)
            L = torch.sum(L)
            CE = -torch.mul(y, torch.log(y_hat_l + 1e-8)) # softmax 2번 태우기 방지 : 직접 구현
            CE = torch.sum(CE) # 64
            # CE = torch.mean(CE) # 1

            loss = L + alpha*CE  # J + alpha*E(-log q(y|x);p(x,y))

        elif y_hat_ul is not None:
            L = L.view_as(y_hat_ul.t()) # 640 -> 10x64
            L = L.t() # 10x64 -> 64x10
            L_ = torch.mul(y_hat_ul, L) # 64x 10

            H = torch.mul(y_hat_ul, torch.log(y_hat_ul + 1e-8)) # 64x10

            U = L_ + H # 64x10
            U = torch.sum(U) # 64
            # U = torch.mean(U) # 1 
            loss = U 

        return loss
#%%
