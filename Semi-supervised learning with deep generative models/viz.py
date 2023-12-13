import torch
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import load_mnist_data

device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
"""2D latent variable z"""

model = torch.load('models/M2.pt')

def latent_img(grid_size, grid_range = (-5, 5), latent_size = 2):
    grid = []
    for _ in range(latent_size):
        axis = np.linspace(grid_range[0], grid_range[1], grid_size) # -5부터 5까지 10개 
        grid.append(axis)

    grid_points = np.meshgrid(*grid) # 한개의 axis로 좌표공간 생성 
    grid_points = np.column_stack([point.ravel() for point in grid_points])

    return grid_points

save_path = 'imgs/'

for i in range(10):
    grids = latent_img(10)
    latent_image = [model.M2Decoder(torch.cat([torch.FloatTensor(grid),
                                                F.one_hot(torch.LongTensor([i]), num_classes = 10).reshape(-1)])).reshape(-1, 28, 28)
                                                for grid in grids] # latent_size + label_size = 12 : M2Decoder input
    
    # latent_image[1].size() # 1x28x28
    latent_grid_img = torchvision.utils.make_grid(latent_image, nrow = 10) 
    # 3x302x302 (280+4+2x9) 4:좌우끝, 2:이미지 사이 간격
    plt.imshow(latent_grid_img.permute(1, 2, 0), cmap = 'gray') # CxHxW -> HxWxC
    plt.axis('off')  # 축 정보 제거
    plt.savefig(f'{save_path}latent_image_{i}.png', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

#%%
"""MNIST analogies implement"""

_, test_dataset = load_mnist_data()

model = torch.load('models/M2_latent200.pt')
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True)
 
def analogy_img(model, data_loader):
    x, _ = next(iter(data_loader))
    x = x.view(x.size(0), -1).to(device) # 1x784

    with torch.no_grad():
        y_hat = model.classifier(x) # 10차원 확률벡터 
        y_pred = torch.argmax(y_hat, dim = 1).unsqueeze(0) # 1
        z_mu, _ = model.M2Encoder(torch.cat([x, y_hat], dim = 1)) # 1xlatent_dim

    # Generating image
    analogy_imgs = []
    for i in range(10):
        y_one_hot = torch.eye(10)[i].unsqueeze(0).to(device) # 1x10
        zy = torch.cat([z_mu, y_one_hot], dim = 1) # z_mu 글씨체 정보
        
        with torch.no_grad():
            x_recon = model.M2Decoder(zy)
        analogy_imgs.append(x_recon.cpu())

    return x, analogy_imgs

xs = []
imgs = []
for i in range(10):
    x, img = analogy_img(model, test_loader)
    xs.append(x); imgs.append(img)

fig, axes = plt.subplots(10, 11, figsize=(22, 20))
for i in range(10):
    # 원래 이미지 출력
    ax = axes[i][0]
    ax.imshow(xs[i].view(28, 28).cpu().numpy(), cmap = 'gray_r')
    # ax.set_title('Original')
    ax.axis('off')

    # 생성된 이미지 출력
    for j in range(10):
        ax = axes[i][j+1]
        ax.imshow(imgs[i][j].view(28, 28).cpu().numpy(), cmap = 'gray_r')
        # ax.set_title(f'Class {j}')
        ax.axis('off')

fig.savefig('imgs/analogy_image.png', bbox_inches = 'tight', pad_inches = 0)
