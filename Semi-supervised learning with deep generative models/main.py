import torch
from module import *
from data_utils import *
from M1 import *
from M2 import *
import copy
import numpy as np


#%%\
"""dataset"""

trainset, validset = load_mnist_data()
train_loader = DataLoader(trainset, 64, shuffle = True, drop_last = True)
valid_loader = DataLoader(validset, 64, shuffle = True, drop_last = True)

num_labeled = 3000
train_labeled_dataset, train_unlabeled_dataset = create_semisupervised_datasets(trainset, num_labeled = num_labeled)
train_loader_l = DataLoader(train_labeled_dataset, batch_size = 128, shuffle = True, drop_last = True)
train_loader_ul = DataLoader(train_unlabeled_dataset, batch_size = 128, shuffle = True,  drop_last = True)

valid_labeled_dataset, valid_unlabeled_dataset = create_semisupervised_datasets(validset, num_labeled = num_labeled)
valid_loader_l = DataLoader(valid_labeled_dataset, batch_size = 128, shuffle = True, drop_last = True)
valid_loader_ul = DataLoader(valid_unlabeled_dataset, batch_size = 128, shuffle = True,  drop_last = True)

#%%
"""M1 train"""
model = M1model(784, [600, 600], 50, [600, 600])
optimizer = optim.Adagrad(model.parameters(), lr = 0.001)

best_loss = float('inf')
patience = 3
patience_counter = 0
epochs = 30

for epoch in range(epochs):
    
    avg_loss = M1train(model, optimizer, train_loader)
    avg_valid_loss = M1eval(model, valid_loader)
    print(f'[Epoch {epoch + 1}] Loss: {avg_loss:.4f}')
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
        print(f"best loss : {best_loss:.4f}")
        break

torch.save(best_model, 'models/M1.pt')
torch.save(best_model_state, 'models/M1_state.pt')
# print(f"Best model saved with Valid Loss: {best_loss:.4f}")
#%%
"""M1 Classifier"""

from classifier import Classifier

modelM1 = torch.load("models/best_M1.pt")
model = Classifier(50, [500, 500], 10)
optimizer = optim.Adam(model.parameters(), lr = 0.0003, betas = (0.9, 0.999))
loss_ = nn.CrossEntropyLoss(reduction = 'sum')
num_labeled = 3000

for epoch in range(epochs):
    cost = 0.0 
    for (x_l, y), (x_ul, _) in zip(train_loader_l, train_loader_ul):
        x_l = x_l.view(x_l.size(0), -1)
        mu, logvar = modelM1.encoder(x_l)
        z, _, _ = modelM1._reparameterize(mu, logvar)

        pred = model(z)

        # y = F.one_hot(y).float()
        loss = loss_(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost += loss.item()

    avg_loss = cost / len(train_loader_l.dataset)
    print(f'[Epoch {epoch + 1}] Loss: {avg_loss:.4f}')

torch.save(model, 'models/M1_classifier.pt')
torch.save(model.state_dict(), 'models/M1_classifier_state.pt')

#%%
"""M2 train"""

alpha = 0.1*num_labeled
best_loss = float('inf')
patience = 5
patience_counter = 0

epochs = 50

model = M2model(784, [500, 500], 10, 
                784, [500, 500], 200, [500, 500], 784)
# optimizer = optim.Adagrad(model.parameters(), lr = 0.001)
optimizer = optim.Adam(model.parameters(), lr = 0.0003, betas = (0.9, 0.999))
# optimizer = optim.RMSprop(model.parameters(), lr = 0.0003, alpha = 0.1, eps = 0.001, momentum = 0.9, centered = True)
# alpha : first momentum decay = 0.1
# eps : second momentum decay = 0.001
# momentum
# centered : initialization bias correction


for epoch in range(epochs):
    
    avg_loss = M2train(model, optimizer, train_loader_l, train_loader_ul)
    avg_valid_loss = M2eval(model, valid_loader_l, valid_loader_ul)
    print(f'[Epoch {epoch + 1}] Loss: {avg_loss:.4f}')
    print(f'[Epoch {epoch + 1}] Validation Loss: {avg_valid_loss:.4f}')
    

torch.save(model, 'models/M2_latent200.pt')
torch.save(model.state_dict(), 'models/M2_latent200_state.pt')
    # 조기 종료 및 모델 저장
    # if avg_valid_loss < best_loss:
    #     best_loss = avg_valid_loss
    #     best_model = copy.deepcopy(model)
    #     best_model_state = copy.deepcopy(model.state_dict())
    #     patience_counter = 0
    # else:
    #     patience_counter += 1

    # if patience_counter >= patience:
    #     print("Early stopping!")
    #     print(f"best loss : {best_loss:.4f}")
    #     break

# torch.save(best_model, 'models/M2.pt')
# torch.save(best_model_state, 'models/M2_state.pt')
# print(f"Best model saved with Valid Loss: {best_loss:.4f}")
#%%
"""M2 accuaracy"""
model = torch.load("models/M2.pt")
print(accuracy(model, valid_loader)) # 93.1891%

#%%
"""M1, M2 stack"""

alpha = 0.1*num_labeled
best_loss = float('inf')
patience = 5
patience_counter = 0

epochs = 30

modelM1 = torch.load("models/best_M1.pt")
model = M2model(50, [500, 500], 10, 50, [500, 500], 50, [500, 500], 784)
optimizer = optim.Adam(model.parameters(), lr = 0.0003, betas = (0.9, 0.999))
# modelM1.eval()
# with torch.no_grad():


for epoch in range(epochs):

    avg_loss = M1_M2train(modelM1, model, optimizer, train_loader_l, train_loader_ul)
    avg_valid_loss = M1_M2eval(modelM1, model, valid_loader_l, valid_loader_ul)
    print(f'[Epoch {epoch + 1}] Loss: {avg_loss:.4f}')
    print(f'[Epoch {epoch + 1}] Validation Loss: {avg_valid_loss:.4f}')  

torch.save(model, 'models/M1_M2.pt')
torch.save(model.state_dict(), 'models/M1_M2_state.pt')
    
    # 조기 종료 및 모델 저장
    # if avg_valid_loss < best_loss:
    #     best_loss = avg_valid_loss
    #     best_model = copy.deepcopy(model)
    #     best_model_state = copy.deepcopy(model.state_dict())
    #     patience_counter = 0
    # else:
    #     patience_counter += 1

    # if patience_counter >= patience:
    #     print("Early stopping!")
    #     print(f"best loss : {best_loss:.4f}")
    #     break

# torch.save(best_model, 'models/M2.pt')
# torch.save(best_model_state, 'models/M2_state.pt')
# print(f"Best model saved with Valid Loss: {best_loss:.4f}")
    
# %%
"""M1+M2 accuaracy"""
model = torch.load("models/M1_M2.pt")
modelM1 = torch.load("models/best_M1.pt")

accuarrcies = []
for x, y in valid_loader:
    x = x.view(x.size(0), -1)
    mu, logvar = modelM1.encoder(x)
    z1, _, _ = modelM1._reparameterize(mu, logvar)
    
    pred = torch.argmax(model.classifier(z1), dim = -1)
    accuarrcies.append(torch.sum(pred == y)/x.size(0))

print(np.mean(accuarrcies))

# %%
