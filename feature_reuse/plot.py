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

n_1 = torch.load("/h/u17/c6/01/chuchun9/csc413/project/result_n_1/list_9.bin")
n_quater = torch.load("/h/u17/c6/01/chuchun9/csc413/project/result_n_quater/list_9.bin")
n_half = torch.load("/h/u17/c6/01/chuchun9/csc413/project/result_n_half/list_9.bin")
n_full = torch.load("/h/u17/c6/01/chuchun9/csc413/project/result_n_full/list_9.bin")

x = range(1, 11)

n_1_train_loss = n_1[0]
n_1_train_acc = n_1[1]
n_1_test_loss = n_1[2]
n_1_test_acc = n_1[3]

n_quater_train_loss = n_quater[0]
n_quater_train_acc = n_quater[1]
n_quater_test_loss = n_quater[2]
n_quater_test_acc = n_quater[3]


n_half_train_loss = n_half[0]
n_half_train_acc = n_half[1]
n_half_test_loss = n_half[2]
n_half_test_acc = n_half[3]

n_full_train_loss = n_full[0]
n_full_train_acc = n_full[1]
n_full_test_loss = n_full[2]
n_full_test_acc = n_full[3]


plt.figure()
plt.plot(x, n_1_train_loss, label="n = 1")
plt.plot(x, n_quater_train_loss, label="n = k/4")
plt.plot(x, n_half_train_loss, label="n = k/2")
plt.plot(x, n_full_train_loss, label="n = k")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("train_loss.pdf")
plt.close()

plt.figure()
plt.plot(x, n_1_test_loss, label="n = 1")
plt.plot(x, n_quater_test_loss, label="n = k/4")
plt.plot(x, n_half_test_loss, label="n = k/2")
plt.plot(x, n_full_test_loss, label="n = k")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("test_loss.pdf")
plt.close()