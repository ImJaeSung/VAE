import torch
from module import *
from data_utils import *
from M1 import *
from M2 import *
import copy

#%%\
trainset, validset = load_mnist_data()
train_loader = DataLoader(trainset, 64, shuffle = True)
valid_loader = DataLoader(validset, 64, shuffle = True)

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
num_labeled = 3000
alpha = 0.1*num_labeled
best_loss = float('inf')
patience = 5
patience_counter = 0

epochs = 30

trainset, validset = load_mnist_data()
train_labeled_dataset, train_unlabeled_dataset = create_semisupervised_datasets(trainset, num_labeled = num_labeled)
train_loader_l = DataLoader(train_labeled_dataset, batch_size = 128, shuffle = True, drop_last = True)
train_loader_ul = DataLoader(train_unlabeled_dataset, batch_size = 128, shuffle = True,  drop_last = True)

valid_labeled_dataset, valid_unlabeled_dataset = create_semisupervised_datasets(validset, num_labeled = num_labeled)
valid_loader_l = DataLoader(valid_labeled_dataset, batch_size = 128, shuffle = True, drop_last = True)
valid_loader_ul = DataLoader(valid_unlabeled_dataset, batch_size = 128, shuffle = True,  drop_last = True)

model = M2model(784, 10, [500, 500], [500, 500], [500, 500], 2)
# optimizer = optim.Adagrad(model.parameters(), lr = 0.001)
# optimizer = optim.Adam(model.parameters(), lr = 0.0003, betas = (0.9, 0.999))
optimizer = optim.RMSprop(model.parameters(), lr = 0.0003, alpha = 0.1, eps = 0.001, momentum = 0.9, centered = True)
# alpha : first momentum decay = 0.1
# eps : second momentum decay = 0.001
# momentum
# centered : initialization bisas correction


for epoch in range(epochs):
    
    avg_loss = M2train(model, optimizer, train_loader_l, train_loader_ul)
    avg_valid_loss = M2eval(model, valid_loader_l, valid_loader_ul)
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

torch.save(best_model, 'models/M2.pt')
torch.save(best_model_state, 'models/M2_state.pt')
# print(f"Best model saved with Valid Loss: {best_loss:.4f}")
