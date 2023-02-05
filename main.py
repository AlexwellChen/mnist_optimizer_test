#gpu方式的mnist手写数字识别
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from adan_Opt import AdanOptimizer
import time
 
EPOCH = 1
BATCH_SIZE = 32
LR = 0.001
DOWNLOAD_MNIST = True
 
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False
)
#change in here
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7,10) #10分类的问题
 
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.out(x)
        return x
        
def main():
    cnn = CNN()
    cnn.cuda()
 
    # optimizer = optim.Adam(cnn.parameters(),lr=LR)
    # optimizer = LSAdam(cnn.parameters())
    optimizer = AdanOptimizer(cnn.parameters())
    loss_func = nn.CrossEntropyLoss()
    
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for epoch in range(EPOCH):
            start_time = time.time()
            for step,(x,y) in enumerate(train_loader):
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()
                output = cnn(b_x)
                loss = loss_func(output,b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prof.step()
                if step % 50 == 0:
                    test_output = cnn(test_x)
    
                    # !!!!!!!! Change in here !!!!!!!!! #
                    pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU
    
                    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
            end_time = time.time()
            print("One Epoch cost ", end_time - start_time, " s.")
if __name__ == '__main__':
    main()