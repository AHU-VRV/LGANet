#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author   : Guo Qingqing
# @Date     : 2022/10/12 下午8:15
# @Software : PyCharm

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from utils.loader import *
import yaml
from PIL import Image
import imageio
from tqdm import tqdm
from models.model import LGANet

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Hyper parameters
config         = yaml.load(open('./configs/config_skin_isic2018.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
patience       = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = config['path_to_data']

test_dataset = isic_loader(path_Data = data_path, train = False, Test = True)
test_loader  = DataLoader(test_dataset, batch_size = 1, shuffle= True)


# In[3]:
model_name = 'LGANet'


save_result_path = './results/'+model_name +'/prediction'+ config['save_result']
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)

save_result_pred_path = './results/'+model_name +'/prob'+ config['save_result']
if not os.path.exists(save_result_pred_path):
    os.makedirs(save_result_pred_path)

Net = LGANet()
Net = Net.to(device)
Net.load_state_dict(torch.load('./weights/LGANet/isic18/best_loss_weight_path/weights_isic18.model', map_location='cpu')['model_weights'])

with torch.no_grad():
    print('val_mode')
    val_loss = 0
    Net.eval()
    for itter, batch in tqdm(enumerate(test_loader)):
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask']
        index = batch['index'].item()
        msk_pred,_,_,_ = Net(img)
        msk_pred = torch.sigmoid(msk_pred)

        name = 'test_'+str(index)+'.png'
        print(name)

        msk_pred_output = msk_pred.cpu().detach().numpy()[0,0]>0.5
        msk_pred_output = (msk_pred_output*255).astype(np.uint8)
        # name = 'test_'+str(index)+'.png'
        # print(name)
        imageio.imwrite(save_result_path+name,msk_pred_output)

        #概率图
        pred = msk_pred.cpu().detach().numpy()[0, 0]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred = (pred*255).astype(np.uint8)
        imageio.imwrite(save_result_pred_path+name, pred)