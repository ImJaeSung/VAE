from torchvision.datasets import MNIST
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_mnist_data(binarize_y = False):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root = 'data', 
                                   train = True,
                                   download = True,
                                   transform = transform)
    
    valid_dataset = datasets.MNIST(root='./data',
                                  train=False, 
                                  transform=transform)

    # 라벨 벡터로 만들기
    # train_dataset.targets = F.one_hot(train_dataset.targets, 
    #                                     num_classes = 10).float()
    # test_dataset.targets = F.one_hot(test_dataset.targets, 
    #                                     num_classes = 10).float()

    return train_dataset, valid_dataset


def create_semisupervised_datasets(train_dataset,
                                   num_classes = 10, 
                                   num_labeled = 3000):

    num_labeled_per_class = num_labeled//num_classes # 한 클래스에 할당할 라벨링 된 데이터 개수 균등하게 맞추기
    labeled_indices = []
    unlabeled_indices = []

    for i in range(10):  # 10개 클래스에 대해 반복
        indices = np.where(train_dataset.targets.numpy() == i)[0]
        np.random.shuffle(indices) # 랜덤으로 뽑기 위해 셔플
        labeled_indices.extend(indices[:num_labeled_per_class]) # 라벨된 데이터 10개씩 저장
        unlabeled_indices.extend(indices[num_labeled_per_class:]) # 나머지는 라벨링 안된 데이터로 저장

    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)

    return labeled_dataset, unlabeled_dataset


