# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:52:42 2021

@author: MaYiming
"""
#导入库
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from trans import CharAndInt, TextData
from charRNN import charRNN
from songci import CharAndInt_S
#参数设置
params = {}
params["DataPath"] = "jaychou_lyrics.txt"
params["MaxLen"] = 2000
params["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params["steps"] = 20
params["batch_size"] = 128
params["epochs"] = 300
params["hidden_size"] = 2048
params["num_layers"] = 2
params["dropout"] = 0.5
#定义数据
#建立歌词数字转化表
charAndint = CharAndInt_S(params)

params["num_classes"] = charAndint.vocab_size()
params["embed_dim"] = params["num_classes"]
#将文本转化为rnn需要的序列
TrainSet = TextData(params,charAndint,params["steps"])
#转化为tensor的格式
TrainSet = DataLoader(TrainSet, params["batch_size"], shuffle = True)
#定义训练模型
model = charRNN(params["num_classes"], params["embed_dim"], params["hidden_size"], 
                params["num_layers"], params["dropout"]).to(params["device"])
#损失函数
criterion = nn.CrossEntropyLoss()
#优化函数
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
#随机选择
def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c
#训练
loss = []
pre = []
for e in range(params["epochs"]):
    model = model.train()
    l = 0
    #每个数据
    for batch in TrainSet:
        x, y = batch
        #转化为long
        x = x.long().to(params["device"])
        y = y.long().to(params["device"])
        #正向传递和反向传播
        out, _ = model(x)
        los = criterion(out, y.view(-1))
        optimizer.zero_grad()
        los.backward()
        #计算loss
        l += los.item()
        #归一化，防止爆炸
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    #输出
    print('epoch: {}, loss is: {:.3f}, perplexity is: {:.3f}'.format(e+1, l, np.exp(l/len(TrainSet))))
    loss.append(l)
    pre.append(np.exp(l/len(TrainSet)))
    #训练
    with torch.no_grad():
        #设置初始
        begin = '夜来匆匆饮散 小诗'
        text_len = 30
        model = model.eval()
        samples = [charAndint.WordToInt(c) for c in begin]
        #初始化输入
        input_txt = torch.LongTensor(samples)[None].to(params["device"])
        _, init_state = model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        #循环输出
        for i in range(text_len):
            out, init_state = model(model_input, init_state)
            pred = pick_top_n(out, top_n=5)
            result.append(pred[0])
            #转化为text
            text_now=charAndint.arrTotext([pred[0]])
            #更新input
            sample_now=[charAndint.WordToInt(c) for c in text_now]
            input_now=torch.LongTensor(sample_now)[None].to(params["device"])
            model_input=input_now[:, -1][:, None]
        text = charAndint.arrTotext(result)
        text = text.replace('<UNK>', ' ')
        print('Generate text is: {}'.format(text))
#绘图
plt.figure()
plt.plot(loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.figure()
plt.plot(pre)
plt.xlabel('epoch')
plt.ylabel('perplexity')
