import torch
from M1 import *
from M2 import *
from data_utils import *

from torch.utils.data import DataLoader
import torch.optim as optim
from itertools import cycle
import copy

def M1train(model, 
            optimizer, 
            train_loader, 
            valid_loader, 
            epochs,
            patience,
            save_path_model,
            save_path_model_dict):
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
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
        print(f'[Epoch {epoch + 1}] Loss: {avg_loss:.4f}')

        model.eval()
        valid_cost = 0.0
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(valid_loader):
                x = x.view(x.size(0), -1)
                x_recon, mu, logvar = model(x)
                valid_cost += model._loss_function(x_recon, x, mu, logvar).item()

        avg_valid_loss = valid_cost / len(valid_loader.dataset)
        print(f'[Epoch {epoch + 1}] Validation Loss: {avg_valid_loss:.4f}')

        # 조기 종료 및 모델 저장
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            best_model = copy.deepcopy(model)
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping!")
            break
    
    if best_model is not None:
        torch.save(best_model, save_path_model)
        torch.save(best_model_state, save_path_model_dict)
        print(f"Best model saved with Valid Loss: {best_loss:.4f}")

model = M1model(784, [600, 600], 50, [600, 600])
trainset, validset = load_mnist_data()
train_loader = DataLoader(trainset, 64, shuffle = True)
valid_loader = DataLoader(validset, 64, shuffle = True)
optimizer = optim.Adagrad(model.parameters(), lr = 0.001)
M1train(model, optimizer, train_loader, valid_loader, 1000, 3, 'model/best_M1.pt', 'model/best_M1_state.pt')

#%%
def M2train(model, 
            optimizer, 
            train_loader_l, 
            train_loader_ul,
            valid_loader_l,
            valid_loader_ul, 
            epochs,
            patience = 3,  
            alpha = 0.1):
    
    best_loss = float('inf')
    patience_counter = 0
    model.train()
    for epoch in range(epochs):
        cost = 0.0

        for (x_l, y), (x_ul, _) in zip(cycle(train_loader_l), train_loader_ul):
            # labeled data
            x_l = x_l.view(x_l.size(0), -1)
            x_recon_l, mu_l, logvar_l, y_hat_l, _, y_l = model(x_l, y)
            loss_l = model._loss_function(x_recon_l, x_l, mu_l, logvar_l, y_hat_l = y_hat_l, y_hat_ul =  None, y = y_l)

            # unlabeled data
            x_ul = x_ul.view(x_ul.size(0), -1)
            x_recon_ul, mu_ul, logvar_ul, _, y_hat_ul, y_ul  = model(x_ul, y = None)
            x_ul = x_ul.repeat(10, 1)
            loss_ul = model._loss_function(x_recon_ul, x_ul, mu_ul, logvar_ul, y_hat_l = None, y_hat_ul = y_hat_ul, y = y_ul)

            loss = loss_l + loss_ul

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost += loss.item()

        avg_loss = cost / (len(train_loader_l.dataset) + len(train_loader_ul.dataset))
        print(f'[Epoch {epoch + 1}] Average Loss: {avg_loss:.4f}')

        model.eval()
        valid_cost = 0.0
        with torch.no_grad():
            for (x_l, y_l), (x_ul, _) in zip(cycle(valid_loader_l), valid_loader_ul):
                x_l = x_l.view(x_l.size(0), -1)
                x_recon_l, mu_l, logvar_l, y_hat_l, _, y_l = model(x_l, y)
                loss_l = model._loss_function(x_recon_l, x_l, mu_l, logvar_l, y_hat_l = y_hat_l, y_hat_ul =  None, y = y_l)

                # unlabeled data
                x_ul = x_ul.view(x_ul.size(0), -1)
                x_recon_ul, mu_ul, logvar_ul, _, y_hat_ul, y_ul  = model(x_ul, y = None)
                x_ul = x_ul.repeat(10, 1)
                loss_ul = model._loss_function(x_recon_ul, x_ul, mu_ul, logvar_ul, y_hat_l = None, y_hat_ul = y_hat_ul, y = y_ul)

                loss = loss_l + loss_ul

                valid_cost += loss.item()

        avg_valid_loss = valid_cost / (len(valid_loader_l.dataset) + len(valid_loader_ul.dataset))
        print(f'[Epoch {epoch + 1}] Validation Loss: {avg_valid_loss:.4f}')

        # 조기 종료
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping!")
            break

#%%
trainset, validset = load_mnist_data()
train_labeled_dataset, train_unlabeled_dataset = create_semisupervised_datasets(trainset)
train_loader_l = DataLoader(train_labeled_dataset, batch_size = 256, shuffle = True, drop_last = True)
train_loader_ul = DataLoader(train_unlabeled_dataset, batch_size = 256, shuffle = True,  drop_last = True)

valid_labeled_dataset, valid_unlabeled_dataset = create_semisupervised_datasets(validset)
valid_loader_l = DataLoader(valid_labeled_dataset, batch_size = 256, shuffle = True, drop_last = True)
valid_loader_ul = DataLoader(valid_unlabeled_dataset, batch_size = 256, shuffle = True,  drop_last = True)

model = M2model(784, 10, [500], [500], [500], 2)
optimizer = optim.Adagrad(model.parameters(), lr = 0.001)


M2train(model, optimizer, train_loader_l, train_loader_ul, valid_loader_l, valid_loader_ul, 10)
# %%
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np


def generate_grid(dim, grid_size, grid_range):
    """
    dim: 차원 수
    grid_size: 그리드 크기
    grid_range: 그리드의 범위, (시작, 끝)
    """
    grid = []
    for i in range(dim):
        axis_values = np.linspace(grid_range[0], grid_range[1], grid_size)
        grid.append(axis_values)

    grid_points = np.meshgrid(*grid)
    grid_points = np.column_stack([point.ravel() for point in grid_points])

    return grid_points

grid = generate_grid(2, 10, (-5,5))

latent_image = [model.M2Decoder(torch.cat([torch.FloatTensor(i), 
                              torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])])).reshape(-1,28,28) 
                              for i in grid]
latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
plt.imshow(latent_grid_img.permute(1,2,0))
plt.show()
# %%
