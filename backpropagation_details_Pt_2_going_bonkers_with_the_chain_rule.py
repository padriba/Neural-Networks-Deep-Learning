#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:46:46 2023

@author: bmatougui
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

x_train = torch.tensor([0,0.5,1])
y_train = torch.tensor([0,1,0])



def dssr_db1(y_train,y_pred,w3,x1_i):
    r = torch.exp(x1_i)/(1+(torch.exp(x1_i)))
    return torch.sum(-2 * w3 * r * (y_train - y_pred))

def dssr_dw1(x_train,y_train,y_pred,w3,x1_i):
    r = torch.exp(x1_i)/(1+(torch.exp(x1_i)))
    return torch.sum(-2 * w3 * r * x_train * (y_train - y_pred))

def dssr_dw2(x_train,y_train,y_pred,w4,x2_i): 
    r = torch.exp(x2_i)/(1+(torch.exp(x2_i)))
    return torch.sum(-2 * w4 * r * x_train * (y_train - y_pred))


def dssr_db2(y_train,y_pred,w4,x2_i):
    r = torch.exp(x2_i)/(1+(torch.exp(x2_i)))
    return torch.sum(-2 * w4 * r * (y_train - y_pred))

def dssr_dw3(y_train,y_pred,y1_i):
    return torch.sum(-2 * y1_i * (y_train-y_pred))


def dssr_dw4(y_train,y_pred,y2_i):
    return torch.sum(-2 * y2_i * (y_train-y_pred))


def dssr_db3(y_train,y_pred):
    return torch.sum(-2 * (y_train-y_pred))


def passforward(x_train,w1,w2,w3,w4,b1,b2,b3):
    x1_i = w1 * x_train + b1
    x2_i = w2 * x_train + b2
    
    sftplus = nn.Softplus()
    y1_i = sftplus(x1_i)
    y2_i = sftplus(x2_i)
    
    y_pred = (y1_i * w3 + y2_i * w4) + b3
    
    return y_pred,x1_i,x2_i,y1_i,y2_i


def prediction(x,w1,w2,w3,w4,b1,b2,b3):
    
    sftplus = nn.Softplus()
    
    x1_i = w1 * x + b1
    y1_i = sftplus(x1_i)
    blue_curve_y = y1_i * w3
    
    x2_i = w2 * x + b2
    y2_i = sftplus(x2_i)
    orange_curve_y = y2_i * w4
    
    green_curve_y = blue_curve_y + orange_curve_y + b3
    
    return blue_curve_y,orange_curve_y,green_curve_y
    
    
    


#intialise parameters
learning_rate = 0.1
w = torch.normal(mean =0, std = 1, size=(1,4))
# w1,w2,w3,w4 = w[0][0].item(),w[0][1].item(),w[0][2].item(),w[0][3].item()
w1,w2,w3,w4 = 2.74,-1.13,0.36,0.63
b1,b2,b3 = 0,0,0


#plot graphic settings
x = torch.from_numpy(np.linspace(0, 1, 100))
y= prediction(x,w1,w2,w3,w4,b1,b2,b3)
plt.ion()
figure, ax = plt.subplots(figsize=(20, 20))
line1, = ax.plot(x, y[2],color= "green")
line2, = ax.plot(x, y[0],color= "blue")
line3, = ax.plot(x, y[1],color= "orange")
ax.scatter(x_train, y_train, s=100)
ax.margins(y=1)

for i in range (460): 
   
    y_pred,x1_i,x2_i,y1_i,y2_i  = passforward(x_train, w1, w2, w3, w4, b1, b2, b3)
    
    print('y_train:' , y_train)
    print('y_pred:' , y_pred)
    
    #Optimisation
    #Gradient decent
    dssr_w1 = dssr_dw1(x_train, y_train, y_pred, w3, x1_i)
    dssr_w2 = dssr_dw2(x_train, y_train, y_pred, w4, x2_i)
    dssr_b1 = dssr_db1(y_train, y_pred, w3, x1_i)
    dssr_b2 = dssr_db2(y_train, y_pred, w4, x2_i)
    dssr_b3 = dssr_db3(y_train,y_pred)
    dssr_w3 = dssr_dw3(y_train,y_pred,y1_i)
    dssr_w4 = dssr_dw4(y_train,y_pred,y2_i)
    
    step_size = torch.tensor([dssr_w1, dssr_w2, dssr_b1 ,dssr_b2,dssr_b3,dssr_w3,dssr_w4 ]) * learning_rate
    
    #print('step_size:' , step_size)
    
    w1,w2,b1,b2,b3,w3,w4 = torch.tensor([w1,w2,b1,b2,b3,w3,w4]) - step_size
    w1,w2,b1,b2,b3,w3,w4 =w1.item(),w2.item(),b1.item(),b2.item(),b3.item(),w3.item(),w4.item() 
    
    
    ##prediction
    blue_curve_y,orange_curve_y,green_curve_y = prediction(x,w1,w2,w3,w4,b1,b2,b3)
    
    #plot
    line1.set_xdata(x)
    line1.set_ydata(green_curve_y)
    
    line2.set_xdata(x)
    line2.set_ydata(blue_curve_y)
    
    line3.set_xdata(x)
    line3.set_ydata(orange_curve_y)
    
    
    figure.canvas.flush_events()
    time.sleep(0.1)


plt.savefig("Pt_2.png")



    
                        
    
    
    
