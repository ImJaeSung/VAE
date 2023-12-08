from data_utils import load_MNIST
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import copy
from tqdm import tqdm
import wandb


train, valid = load_MNIST()   
train_loader = DataLoader(train, shuffle = True, batch_size = 100)
valid_loader = DataLoader(valid, shuffle = False, batch_size = 100)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = {'input_dim' : 28*28,
          'hidden_dim' : 500,
          'latent_dim' : 200,
          'batch_size' : 100,
          'epochs' : 30,
          'lr' : 0.01,
          'best_loss' : 10**9,
          'patience_limit' : 3}
wandb.init(
    # set the wandb project where this run will be logged
    project = "VAE",
    
    # track hyperparameters and run metadata
    config = config 
)


model = VAE(config['input_dim'], config['hidden_dim'], config['latent_dim']).to(device)
optimizer = optim.Adagrad(model.parameters(), lr = config['lr'])

def VAELoss(pred, label, mu, log_sigma2):
     recon = nn.BCELoss(reduction = 'sum')(pred, label)
     kl = 0.5*torch.sum(mu**2 + torch.exp(log_sigma2) - log_sigma2 -1)

     return recon + kl

# parameter 초기값 N(0, 0.01)에서 random sampling
for param in model.parameters():
    nn.init.normal_(param, 0, 0.01)

patience_check = 0
for epoch in tqdm(range(config['epochs'])):
    model.train()
    train_loss = 0
    
    for i, (x, _) in enumerate(train_loader):
        x_train = x.view(-1, config['input_dim']).to(device)
        mu, log_sigma2, pred = model(x_train)

        t_loss = VAELoss(pred, x_train, mu, log_sigma2)

        optimizer.zero_grad()
        t_loss.backward()
        train_loss += t_loss.item()
        optimizer.step()

    print('Epoch: {} Train_Loss: {} :'.format(epoch, train_loss/len(train_loader.dataset)))

    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for batch_idx, (x, _) in enumerate(valid_loader): # VAE는 label이 필요없음
            x_valid = x.view(-1, config['input_dim']).to(device)
            mu, log_sigma2, pred = model(x_valid)
            
            v_loss = VAELoss(pred, x_valid, mu, log_sigma2)
            valid_loss += v_loss.item()

        if valid_loss > config['best_loss']:
            patience_check +=1
                
        if patience_check >= config['patience_limit']:
            break
        
        else:
            best_loss = valid_loss
            patience_check = 0
            best_model = copy.deepcopy(model)        

        print('Epoch: {} Valid_Loss: {} :'.format(epoch, valid_loss/len(valid_loader.dataset)))

        wandb.log({"train_loss": train_loss/len(train_loader.dataset), "valid_loss": valid_loss/len(valid_loader.dataset)})


grid = gridspec.GridSpec(3, 3)
plt.figure(figsize = (10, 10))
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)

for i in range(9):
    ax = plt.subplot(grid[i])
    x, y = valid[i]
    _, _, pred = best_model(x.view(-1,784))
    plt.imshow(pred.detach().numpy().reshape(28,28), cmap = 'gray_r')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('label : {}'.format(y))

wandb.log({"pred" : ax})
wandb.finish()