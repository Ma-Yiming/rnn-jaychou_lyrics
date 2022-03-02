# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:27:27 2021

@author: MaYiming
"""
with open("songci.txt",encoding="utf-8")as p:
    textlines = p.readlines()
    text = ""
    for index,line in enumerate(textlines):
        if line == '\n':
            textlines[index-1] = ' '
    for line in textlines:
        line = line.strip('\n')
        line = line.strip(' ')
        text += line
