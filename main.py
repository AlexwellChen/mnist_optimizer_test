import sys
sys.path.append('..')

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder

# 定义数据预处理
train_tf = tfs.Compose([
    tfs.RandomResizedCrop(224),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 使用 ImageNet 的均值和方差
])

valid_tf = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 使用 ImageFolder 定义数据集
train_set = ImageFolder('./hymenoptera_data/train/', train_tf)
valid_set = ImageFolder('./hymenoptera_data/val/', valid_tf)
# 使用 DataLoader 定义迭代器
train_data = DataLoader(train_set, 64, True, num_workers=4)
valid_data = DataLoader(valid_set, 128, False, num_workers=4)

# 使用预训练的模型
net = models.resnet50(pretrained=True)

# 将最后的全连接层改成二分类
net.fc = nn.Linear(2048, 2)

criterion = nn.CrossEntropyLoss()

from adan import Adan
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, fused=True)

from utils import train
print("Train with fused adam")
train(net, train_data, valid_data, 20, optimizer, criterion)

print("--------------------")
print("Train with fused adan")
betas = (0.98, 0.92, 0.9)
optimizer = Adan(net.parameters(), betas=betas, lr=1e-3, foreach=True, fused=True)
# 使用预训练的模型
net = models.resnet50(pretrained=True)
# 将最后的全连接层改成二分类
net.fc = nn.Linear(2048, 2)
train(net, train_data, valid_data, 20, optimizer, criterion)