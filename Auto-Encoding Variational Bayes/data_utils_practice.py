import numpy as np
from torchvision import datasets
import os
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


trainset = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)

testset = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

class OpenMNIST:
    def __init__(self, path):
        self.x_train = self.loadimages(os.path.join(path, 'train-images-idx3-ubyte'))
        self.y_train = self.loadlabels(os.path.join(path, 'train-labels-idx1-ubyte'))
        self.x_test = self.loadimages(os.path.join(path, 't10k-images-idx3-ubyte'))
        self.y_test = self.loadlabels(os.path.join(path, 't10k-labels-idx1-ubyte'))

    def loadimages(self, path):
        f = open(path, 'rb')
        
        magic_number = int.from_bytes(f.read(4), 'big')
        num_imgs = int.from_bytes(f.read(4), 'big')
        h = int.from_bytes(f.read(4), 'big')
        w = int.from_bytes(f.read(4), 'big')

        arr_byte = bytearray(f.read())
        imgs = np.array(arr_byte, dtype = np.uint8).reshape(num_imgs, h, w) # 

        f.close()

        return imgs

    def loadlabels(self, path):
        f = open(path, 'rb')

        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        arr_byte = bytearray(f.read())
        labels = np.array(arr_byte, dtype = np.uint8).reshape(num_labels, 1) 

        f.close()

        return labels  

class CustomDataset(Dataset):
    def __init__(self, path):
        self.data = OpenMNIST(path).x_train/255
        self.data_size, h, w = self.data.shape
        self.sample_dim = h*w
        self.data = self.data.reshape(self.data_size, self.sample_dim)
        self.data = torch.from_numpy(self.data).float()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        
        return sample
