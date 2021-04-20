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


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, hparam):
        self.lr = hparam['lr']
        self.model_name = hparam['model_name']
        self.bs = hparam['batchsize']
        self.logspace = hparam['logspace']
        self.gamma = hparam['gamma']
        self.momentum = hparam['momentum']
        self.weight_decay = hparam['weight_decay']
        self.epoch = hparam['epoch']
        self.seed = hparam['seed']
        self.opt = hparam['opt']
        self.datapath = hparam['datapath']
        self.result_path = hparam['result_path']
        self.lst = [[], [], [], [], [], [], [], []]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(self.device)
        else:
            self.device = torch.device("cpu")
        self.num_classes = len(os.listdir(os.path.join(hparam['datapath'], "train")))

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        seed_torch(int(self.seed))

        self.prepare_model()
        self.prepare_dataset()

    def prepare_model(self):

        model = densenet121(pretrained=False, num_classes=self.num_classes, all_n=(2,3,6,4))
        # model = models.densenet121(pretrained=False, num_classes=self.num_classes)
        self.img_size = 224

        self.model = model
        self.model = self.model.to(self.device)
        if self.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_func = nn.CrossEntropyLoss(reduction="mean")

    def prepare_dataset(self):
        ## Data augmentation
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # self.train_set = datasets.ImageFolder(root=self.datapath + "/train", transform=train_transform)
        self.train_set = datasets.CIFAR10(root='../', train=True,
                                        download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.bs, shuffle=True)
        # self.test_set = datasets.ImageFolder(root=self.datapath + "/val", transform=test_transform)
        self.test_set = datasets.CIFAR10(root='../', train=False,
                                       download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=int(self.bs/2), shuffle=True, )

    def train(self):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        Loss = 0
        Acc = 0
        for i, (img, lbl) in enumerate(self.train_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)

            output = self.model(img)
            loss = self.loss_func(output, lbl)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(output.size())
            prediction = torch.argmax(output, axis=1)

            accuracy = torch.sum(prediction == lbl).item() / len(prediction)
            Acc += accuracy
            Loss += loss.cpu().item()

            if i % 10 == 0:
                print("[train] batch: %d, loss: %.3f, acc: %.3f" % (i + 1, Loss / (i + 1), Acc / (i + 1)))
        self.lst[0].append(Loss / (i + 1))
        self.lst[1].append(Acc / (i + 1))

    def draw(self):
        x = np.arange(0, len(self.lst[0]), 1)
        train_l = np.array(self.lst[0])
        train_e = np.array(self.lst[1])
        test_l = np.array(self.lst[2])
        test_e = np.array(self.lst[3])
        plt.figure()
        plt.subplot(211)
        plt.plot(x, train_l, color='red')
        plt.plot(x, test_l, color='blue')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.subplot(212)
        plt.plot(x, train_e, color='red')
        plt.plot(x, test_e, color='blue')
        plt.xlabel("epoch")
        plt.ylabel("acc")

        plt.savefig(os.path.join(self.result_path, "curve.png"))
        plt.close()

    def test(self):
        self.model.eval()
        # self.model.to(self.device)

        Loss = 0
        Acc = 0
        for i, (img, lbl) in enumerate(self.test_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)

            output = self.model(img)
            loss = self.loss_func(output, lbl)
            # print(output.size())
            prediction = torch.argmax(output, axis=1)

            accuracy = torch.sum(prediction == lbl).item() / len(prediction)
            Acc += accuracy
            Loss += loss.cpu().item()

            if i % 10 == 0:
                print("[test] batch: %d, loss: %.3f, acc: %.3f" % (i + 1, Loss / (i + 1), Acc / (i + 1)))
        self.lst[2].append(Loss / (i + 1))
        self.lst[3].append(Acc / (i + 1))

    def start(self):
        if self.logspace != 0:
            logspace_lr = np.logspace(np.log10(self.lr), np.log10(self.lr) - self.logspace, self.epoch)

        for e in range(self.epoch):
            if self.logspace != 0:
                for param in self.optimizer.param_groups:
                    param['lr'] = logspace_lr[e]
            self.train()
            self.test()
            self.draw()
        torch.save(self.model.state_dict(), os.path.join(self.result_path, "model_{}.bin".format(e)))
        torch.save(self.lst, os.path.join(self.result_path, "list_{}.bin".format(e)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # print("hello")
    stream = open('./hparam.yaml', 'r')
    hparam = yaml.load(stream)
    print(hparam)

    trainer = Trainer(hparam)
    trainer.start()


if __name__ == "__main__":
    main()