
"""
    takes data stored in train, test folders

    returns train_dataloader, test_dataloader and class_names
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

default_transform = transforms.Compose([
    transforms.Resize(size = (64, 64)),
    transforms.ToTensor()
])

class CustomDatasetTrain(Dataset):
    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def collectDataFormatted(train_path, test_path, batch_size = BATCH_SIZE, train_transform = default_transform, test_transform = default_transform):
    train_data = datasets.ImageFolder(root = train_path, transform = train_transform, target_transform = None)
    test_data = datasets.ImageFolder(root = test_path, transform = test_transform, target_transform = None)

    train_dataloader = DataLoader(dataset = train_data, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True, pin_memory = True)
    test_dataloader = DataLoader(dataset = test_data, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True, pin_memory = True)
    
    print(train_data.classes)
    return train_dataloader, test_dataloader, train_data.classes

def collectDataUnformatted(images, labels, batch_size = BATCH_SIZE, train_transform = default_transform, test_transform = default_transform):
    train_data = CustomDatasetTrain(images, labels, train_transform)
    test_data = CustomDatasetTrain(images, labels, test_transform)
    
    train_dataloader = DataLoader(dataset = train_data, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True, pin_memory = True)
    test_dataloader = DataLoader(dataset = test_data, batch_size = batch_size, num_workers = NUM_WORKERS, shuffle = True, pin_memory = True)
    
    return train_dataloader, test_dataloader
