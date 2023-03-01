
import os
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from timeit import default_timer as timer


def calculate_time (start, end, device):
    total = end-start
    print(f"Time taken was {total:.3f}s on {device}")

def train_step(model, dataloader, lossfunc, optimizer, accuracy_fn, device):
    model.train()
    train_loss, train_acc = 0, 0
    for X_train, y_train in dataloader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_logits = model(X_train)
        y_preds = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

        loss = lossfunc(y_logits, y_train)
        acc = accuracy_fn(y_preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        train_acc += acc

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model, dataloader, lossfunc, accuracy_fn, device):
    model.eval()
    with torch.inference_mode():
        loss = 0
        acc = 0
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_logits = model(X_test)
            loss += lossfunc(y_logits, y_test)
            acc += accuracy_fn(torch.softmax(y_logits, dim = 1).argmax(dim = 1), y_test)

        loss /= len(dataloader)
        acc /= len(dataloader)
        
        return loss, acc


def train(model, epochs, train_dataloader, test_dataloader, lossfunc, optimizer, accuracy_fn, device = 'cpu'):
    print(device)
    model.to(device)
    accuracy_fn.to(device)

    train_loss, test_loss, train_acc, test_acc = [], [], [], []

    start_time = timer()
    for t in tqdm(range(epochs)):
        print(f"Epoch #{t+1}\n------------------------------------------------------------")
        loss, acc = train_step(model, train_dataloader, lossfunc, optimizer, accuracy_fn, device)
        train_loss.append(loss)
        train_acc.append(acc)
        print(f"---Train Acc : {(acc*100):.3f}% || Train Loss : {loss:.3f}---")

        loss, acc = test_step(model, test_dataloader, lossfunc, accuracy_fn, device)
        test_loss.append(loss)
        test_acc.append(acc)
        print(f"---Test Acc : {(acc*100):.3f}% || Test Loss : {loss:.3f}---")
    end_time = timer()
    calculate_time(start_time, end_time, device)
    return list(train_loss), list(train_acc), list(test_loss), list(test_acc)
