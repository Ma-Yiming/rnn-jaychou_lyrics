# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:24:43 2021

@author: MaYiming
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class charRNN(nn.Module):
    def __init__(self,num_classes, embed_dim, hidden_size, num_layers,dropout):
        super(charRNN,self).__init__()
        #设置网络参数
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #词嵌入
        self.embedding = nn.Embedding(num_classes, embed_dim)
        #三种网络
        #self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        #self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        #输出网络
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self,x , hs = None):
        batch = x.shape[0]
        #hs层的初始化
        # if hs is None:
        #     hs = (torch.zeros(self.num_layers, batch, self.hidden_size).cuda(),torch.zeros(self.num_layers, batch, self.hidden_size).cuda())
        if hs is None:
            hs = Variable(torch.zeros(self.num_layers, batch, self.hidden_size))
        if torch.cuda.is_available():
            hs = hs.cuda()
        #词嵌入
        embed = self.embedding(x)
        embed = embed.permute(1,0,2)
        #返回
        out, h0 = self.rnn(embed, hs)
        #reshape
        le, mb, hd = out.size()
        out = out.view(le*mb, hd)
        #输出
        out = self.classifier(out)
        #reshape为标准格式
        out = out.view(le, mb,-1)
        out = out.permute(1,0,2).contiguous()
        return out.view(-1,out.size(2)), h0
