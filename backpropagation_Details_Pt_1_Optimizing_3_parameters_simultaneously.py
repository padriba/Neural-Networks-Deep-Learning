#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:12:20 2023

@author: bmatougui
"""

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time
import numpy as np


x_train = torch.tensor([0,0.5,1])
y_train = torch.tensor([0, 1,0])


def beforactivation_topnode(x):
    return x * (3.34) - 1.43


def beforactivation_bottomnode(x):
    return x*(-3.53)+0.57 


def dssr_dw3(y_train,y_pred,y1_i):
    return torch.sum(-2 * y1_i * (y_train-y_pred))


def dssr_dw4(y_train,y_pred,y2_i):
    return torch.sum(-2 * y2_i * (y_train-y_pred))


def dssr_db3(y_train,y_pred):
    return torch.sum(-2 * (y_train-y_pred))


def passforward(x_train,b3,w3,w4):
    sftplus = nn.Softplus()
    yi_1 = sftplus(beforactivation_topnode(x_train)) 
    yi_2 = sftplus(beforactivation_bottomnode(x_train))
    final_output = yi_1 * w3 + yi_2 * w4 + b3
    return final_output,yi_1,yi_2


def prediction(x,b3,w3,w4):
    
    sftplus = nn.Softplus()
    
    blue_curve_y= sftplus(beforactivation_topnode(x)) * w3
    orange_curve_y= sftplus(beforactivation_bottomnode(x)) * w4
    green_curve_y = blue_curve_y + orange_curve_y + b3
    
    return blue_curve_y,orange_curve_y,green_curve_y
    


#intialise parameters
learning_rate = 0.1
b3 = 0
w = torch.normal(mean =0, std = 1, size=(1,2))
w3 = w[0][0]
w4 =w[0][1]

#plot graphic settings
x = torch.from_numpy(np.linspace(0, 1, 100))
y= prediction(x, b3, w3, w4)
plt.ion()
figure, ax = plt.subplots(figsize=(20, 20))
line1, = ax.plot(x, y[2],color= "green")
line2, = ax.plot(x, y[0],color= "blue")
line3, = ax.plot(x, y[1],color= "orange")
ax.scatter(x_train, y_train, s=100)
ax.margins(y=1)

for i in range (250): 
    y_pred,yi_1,yi_2 = passforward(x_train,b3,w3,w4)
    
    print('y_train:' , y_train)
    print('y_pred:' , y_pred)
    
    #Optimisation
    #Gradient decent
    dssr_b3 = dssr_db3(y_train,y_pred)
    dssr_w3 = dssr_dw3(y_train,y_pred,yi_1)
    dssr_w4 = dssr_dw4(y_train,y_pred,yi_2)
    
    print('b3:' ,b3)
    print('w3:' ,w3)
    print('w4:' ,w4)
    
    step_size = torch.tensor([dssr_b3 * learning_rate,dssr_w3*learning_rate,dssr_w4 * learning_rate])
    print('step_size:' , step_size)
    b3,w3,w4 = torch.tensor([b3,w3,w4]) - step_size
    b3,w3,w4 = b3.item(),w3.item(),w4.item()
    
    print('new b3:' , b3)
    print('new w3:' , w3)
    print('new w4:' , w4)
    
    ##prediction
    blue_curve_y,orange_curve_y,green_curve_y = prediction(x, b3, w3, w4)
    
    #plot
    line1.set_xdata(x)
    line1.set_ydata(green_curve_y)
    
    line2.set_xdata(x)
    line2.set_ydata(blue_curve_y)
    
    line3.set_xdata(x)
    line3.set_ydata(orange_curve_y)
    
    
    figure.canvas.flush_events()
    time.sleep(0.1)


    






