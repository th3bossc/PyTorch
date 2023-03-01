
import os
import torch
import torch.nn as nn
from torchmetrics import Accuracy

class TinyVGG(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units, adjust_mat_size):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape, out_channels = hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, padding = 1, stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = adjust_mat_size, out_features = output_shape)
        )

    def forward(self, X):
        return self.classifier(self.layers2(self.layers1(X)))

def getLossOpt(model, lr = 1e-3, choice = 'Adam', num_classes = 0):
    optimizer : torch.optim.Optimizer
    lossfunc = nn.CrossEntropyLoss()
    if (choice == 'Adam'):
        optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    elif choice == 'SGD':
        optimizer = torch.optim.SGD(params = model.parameters(), lr = lr)
    elif choice == 'RMS':
        optimizer = torch.optim.RSMProp(params = model.parameters(), lr = lr)


    accuracy = Accuracy(task = 'multiclass', num_classes = num_classes)

    return lossfunc, optimizer, accuracy
