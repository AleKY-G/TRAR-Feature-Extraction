import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os

def Transpose_tensor(input_tensor):
    '''
        转置操作: h < w
    '''
    b,c,h,w = input_tensor.size()
    if(h>w):
        input_tensor = input_tensor.transpose(2,3)
    return input_tensor

def Padding2D(input, height, width):
    input_height = input.size(2)
    input_width = input.size(3)
    output = input
    if input_height < height:
        padding_height = height - input_height
        padding_upside = padding_height//2
        padding_downside = padding_height-padding_upside
        output = F.pad(output, (0,0,padding_upside,padding_downside), mode="constant",value=0)
    else:
        output = output[:,:,:height,:]
    if input_width < width:
        padding_width = width - input_width
        padding_leftside = padding_width//2
        padding_rightside = padding_width - padding_leftside
        output = F.pad(output, (padding_leftside,padding_rightside,0,0), mode="constant", value=0)
    else:
        output = output[:,:,:,:width]
    return output


def TRAR_Preprocess(img_feat, downsample=2):
    img_feat = Transpose_tensor(img_feat)
    img_feat = Padding2D(img_feat, 32, 32)
    if downsample == 2:
        img_feat = F.max_pool2d(img_feat, kernel_size=2, stride=2)
    elif downsample == 4:
        img_feat = F.max_pool2d(img_feat, kernel_size=2, stride=2)
        img_feat = F.max_pool2d(img_feat, kernel_size=2, stride=2)
    img_feat = img_feat.cpu()
    return img_feat[0].numpy()