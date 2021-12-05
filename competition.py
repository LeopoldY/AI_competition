#!/usr/bin/env python
# coding: utf-8

# # 人工智能杯程序设计竞赛
# 请在指定区域答题，运行结果请保留，存档后上交ipynb文件。非指定答题区域请勿修改。
# 1-4题每题10分，5-8题每题15分，满分100分
# 

# * 1. 已知两个list x和y，用列表推导实现x和y中前5个数的平方相加，得到新的列表，例如下面的例子应该得到列表 5，25……等

# In[ ]:


# 请在此处答题

x = [1, 3, 5, 7, 9, 11, 13, 15]
y = [2, 4, 6, 8, 10, 12, 14, 16]


# * 2. 已知一个用list表示的矩阵，求它的转置，仍以list表示。
# 要求写一个函数实现这个功能，并调用这个函数，输出结果。矩阵的行、列不能固定死。
# 

# In[ ]:


# 请在此处答题
mat = [[1,2,3],[4,5,6]]
def transpose(m):
    pass



# * 3. 从一段英文文章article.txt 里统计出现频率最高的3个单词，生成一个字典，形如{'How': 5, 'Many':4, 'road':3}，输出该字典内容
# 

# In[ ]:


# 请在此处答题


# * 4. 定义一个学生类。有以下属性：1 姓名 2 年龄  3 一个字典，记录了每门课的成绩（语文，数学，英语)。
# 有3个类方法：1 获取学生的姓名：get_name() 返回类型:str   2 获取学生的年龄：get_age() 返回类型:int
# 3 返回3门科目的最高分：get_max() 返回类型:int
# 
# 写好类以后，支持以下调用方式，并调用三个类方法输出结果：
# zm = Student('zhangming', 20, [80, 90, 85])
# 意思是 zhangming，20岁，[80，90，85] 表示语文、数学、英语三门课的成绩。
# 

# In[ ]:


# 请在此处答题
zm = Student('zhangming', 20, [80, 90, 85])


# * 5. 定义一个电影售票机类TicketingMachine，支持以下功能（每个功能由一个成员函数实现）：
# （1）addFilm 录入新上映的电影名，以及座位数，默认50座，单价，默认30元。要能支持多个电影。
# （2）book 指定电影名、张数进行订票。如果没有这个电影，或已满坐，要有提示。
# （3）refund 指定电影名、张数进行退票。
# （4）total 显示目前为止的总票房。
# 

# In[ ]:


#coding=utf-8

# 请在此处答题
class TicketingMachine:
    def __init__(self):
        pass

    def addFilm(self, name, maxSeat):
        return True

    def book(self, name, count):
        return True
    
    def refund(self, name, count):
        return True
    
    def total(self):
        return 10000

# 下面的代码请勿修改
if __name__ == '__main__':
    t = TicketingMachine()
    t.addFilm('沙丘', 20) # 增加一部影片沙丘，最大20座
    t.addFilm('悬崖之上', 10)
    if t.book('悬崖之上', 8):
        print('成功预订')
    else:
        print('预定失败')
    if t.book('悬崖之上', 6):
        print('成功预订')
    else:
        print('预定失败')
    if t.refund('悬崖之上', 9):
        print('退票失败')
    else:
        print('退票成功')


# * 6. 打开一张图片road.jpg，实现阈值分割，即大于某个阈值设置为白色，否则为黑色，效果如“道路分割参考结果”所示。提示，如果用pyplot，即plt.imshow() 方式显示灰度图，颜色可能会与真实不符，没有关系。

# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE) # 以灰度图方式打开，此行不用修改


# * 7. 以下是根据一个人的身高和体重，判断ta 的性别的训练程序。训练集和验证集为 sex_train.txt， sex_val.txt。
# 请把网络结构修改为三层神经网络结构，输入层、隐藏层、输出层，其中隐藏层有3个节点，隐藏层带激活函数。保留程序运行的训练过程结果。

# In[ ]:


# 请在此处答题

import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random

class sexnet(nn.Module):
    def __init__(self):
        super(sexnet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(2, 2),
        )

    def forward(self, x):
        out = self.dense(x)
        return out

class SexDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        data = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data.append((float(words[0]) / 2.0, float(words[1]) / 80.0, int(words[2])))
        random.shuffle(data)
        self.data = data

    def __getitem__(self, index):
        return torch.FloatTensor([self.data[index][0], self.data[index][1]]), self.data[index][2]

    def __len__(self):
        return len(self.data)

def train():
    batchsize = 10
    train_data = SexDataset(txt='sex_train.txt')
    val_data = SexDataset(txt='sex_val.txt')
    train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batchsize)

    model = sexnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_func = nn.CrossEntropyLoss()
    epochs = 100
    for epoch in range(epochs):
        # training-----------------------------------
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            print('epoch: %2d/%d batch %3d/%d  Train Loss: %.3f, Acc: %.3f'
                  % (epoch + 1, epochs, batch, math.ceil(len(train_data) / batchsize),
                     loss.item(), train_correct.item() / len(batch_x)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # 更新learning rate
        print('Train Loss: %.6f, Acc: %.3f' % (train_loss / (math.ceil(len(train_data)/batchsize)),
                                               train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Val Loss: %.6f, Acc: %.3f' % (eval_loss / (math.ceil(len(val_data)/batchsize)),
                                             eval_acc / (len(val_data))))
        # save model --------------------------------
        #if (epoch + 1) % 1 == 0:
        #   torch.save(model.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')

if __name__ == '__main__':
    train()
    print('finished')


# * 8. 把性别识别训练程序修改为支持3个特征（身高、体重、腰围），数据请参考sex_train3.txt sex_val3.txt。网络结构不作要求，但不要过于复杂。保留程序运行的训练过程结果

# In[ ]:


# 请在此处答题


