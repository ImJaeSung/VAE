from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def load_MNIST():
    mnist_transform = transforms.Compose([
    transforms.ToTensor()
    ])
    
    trainset = MNIST(
        root = 'data',
        train = True,
        download = True,
        transform = mnist_transform
        )
    
    validset = MNIST(
        root = 'data',
        train = False,
        download = True,
        transform = mnist_transform
        )
    
    return trainset, validset