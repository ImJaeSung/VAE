import torch
from M1 import *
from M2 import *
from data_utils import *

from torch.utils.data import DataLoader
import torch.optim as optim
from itertools import cycle
import copy

def M1train(model, optimizer, train_loader):
    
    model.train()
    cost = 0.0

    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(x.size(0), -1)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = model._loss_function(x_recon, x, mu, logvar)
        loss.backward()

        cost += loss.item()
        optimizer.step()
    
    avg_loss = cost / len(train_loader.dataset)
    return avg_loss
    # print(f'[Epoch {epoch + 1}] Loss: {avg_loss:.4f}')

def M1eval(model, valid_loader):
   
    model.eval()
    valid_cost = 0.0
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(valid_loader):
            x = x.view(x.size(0), -1)
            x_recon, mu, logvar = model(x)
            valid_cost += model._loss_function(x_recon, x, mu, logvar).item()

    avg_valid_loss = valid_cost / len(valid_loader.dataset)
    return avg_valid_loss
    # print(f'[Epoch {epoch + 1}] Validation Loss: {avg_valid_loss:.4f}')
#%%
def M2train(model, 
            optimizer, 
            train_loader_l, 
            train_loader_ul):
    
    model.train()
    cost = 0.0

    for (x_l, y), (x_ul, _) in zip(cycle(train_loader_l), train_loader_ul):
        # labeled data
        x_l = x_l.view(x_l.size(0), -1)
        y_hat_l = model.classifier(x_l)
        
        y_l = F.one_hot(y, num_classes = 10).float()
        
        x_recon_l, z_mu_l, z_logvar_l = model(x_l, y_l)
        loss_l = model._loss_function(x_recon_l, x_l, z_mu_l, z_logvar_l, y_hat_l, y_hat_ul =  None, y = y_l)

        # unlabeled data
        x_ul = x_ul.view(x_ul.size(0), -1) # 64x784
        y_hat_ul = model.classifier(x_ul) # q(y|x;phi) 64x10
        
        y_ul = model._enumerate_discrete(x_ul, 10) # 640x10
        x_ul = x_ul.repeat(10, 1)  # 640x784
        
        x_recon_ul, z_mu_ul, z_logvar_ul = model(x_ul, y_ul)
        loss_ul = model._loss_function(x_recon_ul, x_ul, z_mu_ul, z_logvar_ul, y_hat_l = None, y_hat_ul = y_hat_ul, y = y_ul)

        loss = loss_l + loss_ul

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost += loss.item()

    avg_loss = cost / (2*len(train_loader_ul.dataset))
    return avg_loss
    # print(f'[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}')


def M2eval(model,
           valid_loader_l,
           valid_loader_ul):
    
    model.eval()
    valid_cost = 0.0
    with torch.no_grad():
        for (x_l, y), (x_ul, _) in zip(cycle(valid_loader_l), valid_loader_ul):
            # labeled data
            x_l = x_l.view(x_l.size(0), -1)
            y_hat_l = model.classifier(x_l)

            y_l = F.one_hot(y, num_classes = 10).float()
            x_recon_l, z_mu_l, z_logvar_l= model(x_l, y_l)
            loss_l = model._loss_function(x_recon_l, x_l, z_mu_l, z_logvar_l, y_hat_l, y_hat_ul =  None, y = y_l)

            # unlabeled data
            x_ul = x_ul.view(x_ul.size(0), -1)
            y_hat_ul = model.classifier(x_ul)

            y_ul = model._enumerate_discrete(x_ul, 10)
            x_ul = x_ul.repeat(10, 1)

            x_recon_ul, z_mu_ul, logvar_ul = model(x_ul, y_ul)
            loss_ul = model._loss_function(x_recon_ul, x_ul, z_mu_ul, z_logvar_ul, y_hat_l = None, y_hat_ul = y_hat_ul, y = y_ul)

            loss = loss_l + loss_ul

            valid_cost += loss.item()

    avg_valid_loss = valid_cost / (len(valid_loader_l.dataset) + len(valid_loader_ul.dataset))
    return avg_valid_loss
    # print(f'[Epoch {epoch + 1}] Validation Loss: {avg_valid_loss:.4f}')
# %%
