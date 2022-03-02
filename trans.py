# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:12:25 2021

@author: MaYiming
"""
import numpy as np
import torch

#映射关系类，可以看作一种哈希
class CharAndInt:
    def __init__(self,params):
        #以utf-8读入，txt为gbk
        with open(params["DataPath"],encoding = "utf-8") as p:
            text = p.read()
            #为了调试方便，将换行等替换为空格
            text = text.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
        #将每个字读入，构建列表
        word_list = [v for s in text for v in s]
        #不重复列表
        vocab = set(word_list)
        #下边主要用来排序
        vocab_dict = {}
        #统计出现次数
        for word in vocab:
            vocab_dict[word] = 0
        for word in word_list:
            vocab_dict[word] += 1
        vocab_sorted = []
        #append入列表
        for word in vocab_dict:
            vocab_sorted.append((word, vocab_dict[word]))
        #利用lambda按照出现次数排序
        vocab_sorted.sort(key = lambda x:x[1], reverse = True)
        #构建排好序的不重复词表
        vocab = [x[0] for x in vocab_sorted]
        #最大长度设置
        if len(vocab) > params["MaxLen"]:
            vocab = vocab[:params["MaxLen"]]
        self.vocab = vocab
        self.len = len(self.vocab)
        #两个字典，一个是词与数字的转换，一个是数字与词的转换。
        self.wordToint = {c:i for i,c in enumerate(vocab)}
        self.intToword = dict(enumerate(vocab))
    def vocab_size(self):
        return self.len + 1
    #单个文字到数字
    def WordToInt(self,word):
        if word in self.vocab:
            return self.wordToint[word]
        else:
            return len(self.vocab)
    #单个数字到文字
    def IntToWord(self,Int):
        if Int < len(self.vocab):
            return self.intToword[Int]
        elif Int == len(self.vocab):
            return '<UNK>'
        else:
            return Exception('Unknown index')
    #文章到数字
    def textToarr(self,text):
        arr = []
        for word in text:
            arr.append(self.WordToInt(word))
        return np.array(arr)
    #数字到文章
    def arrTotext(self,arr):
        text = []
        for Int in arr:
            text.append(self.IntToWord(Int))
        return "".join(text)
#通过CharAndInt类将任意文本处理为RNN可识别数据
class TextData(object):
    def __init__(self, params, trans, step):
        self.trans = trans
        self.step = step
        with open(params["DataPath"],encoding = "utf-8") as p:
            text = p.read()
            #为了调试方便，将换行等替换为空格
            text = text.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
        #将每个字读入，构建列表
        words = [v for s in text for v in s]
        seq = int(len(words)/step)
        self.seq = seq
        #去除剩余的
        words = words[:seq*step]
        arr = trans.textToarr(words)
        #变换序列
        arr = arr.reshape((seq,-1))
        self.arr = torch.from_numpy(arr)
    def __getitem__(self, index):
        #返回数据和标签
        x = self.arr[index,:]
        y = torch.zeros(x.size())
        y[:-1],y[-1] = x[1:],x[0]
        return x,y
    def __len__(self):
        return self.seq