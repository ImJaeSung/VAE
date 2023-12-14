import torch.nn as nn
import torch.nn.functional as F


"""q(y|x;phi) = Cat(y|pi(x))"""
class Classifier(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size : list,
                 label_size):
        
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList()
        prev_h = input_size
        for h in hidden_size:
            self.layers.append(nn.Linear(prev_h, h))
            prev_h = h
        self.output_layer = nn.Linear(prev_h, label_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.softplus(layer(x))
        x = self.output_layer(x)
        return F.softmax(x)