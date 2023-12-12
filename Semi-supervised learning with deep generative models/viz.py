import torch
import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# save_path_model = 'model/best_M2.pt',
# save_path_model_dict = 'model/best_M2_state.pt'
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
    grids= latent_img(10)
    latent_image = [model.M2Decoder(torch.cat([torch.FloatTensor(grid),
                                               F.one_hot(torch.LongTensor([i]), num_classes = 10).reshape(-1)])).reshape(-1, 28, 28)
                                               for grid in grids]
    latent_grid_img = torchvision.utils.make_grid(latent_image, nrow=10)
    plt.imshow(latent_grid_img.permute(1,2,0))
    plt.axis('off')  # 축 정보 제거
    plt.savefig(f'{save_path}latent_image_{i}.png', bbox_inches = 'tight', pad_inches = 0)
    plt.show()