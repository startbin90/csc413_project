import torch
import numpy as np
import sys
from torch import nn
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch import nn
import matplotlib.pyplot as plt
import yaml
import os
from densenet import  *

def prepare_dataset():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = datasets.CIFAR10(root='../', train=True,
                                      download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    return train_loader

def prepare_model(all_n):
    model = densenet121(pretrained=False, num_classes=10, all_n=all_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    return model, optimizer, loss_func

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_loader = prepare_dataset()
    memory = []
    for all_n in [(1,1,1,1), (2,3,6,4), (3,6,12,8), (6,12,24,16)]:
        torch.cuda.reset_max_memory_allocated()
        model, optimizer, loss_func = prepare_model(all_n)
        model = model.to(device)
        model.train()
        for i, (img, lbl) in enumerate(train_loader):
            img = img.to(device)
            lbl = lbl.to(device)
            output = model(img)
            loss = loss_func(output, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            memory.append(mem)
            break
        torch.cuda.empty_cache()
    print(memory)

    plt.figure()
    plt.plot(['config1', 'config2', 'config3', 'config4'], memory, color='red')
    plt.title("Peak memory usage under different feature resuability of one batch of training")
    plt.xlabel('Configurations')
    plt.ylabel('Memory usage in Mbs')
    plt.savefig("memory.pdf")
    plt.close()

if __name__ == '__main__':
    main()

