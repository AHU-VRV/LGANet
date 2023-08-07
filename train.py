#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.loader import *
import pandas as pd
import glob
import nibabel as nib
import numpy as np
import copy
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange, repeat
from models.model import LGANet


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


## Loader
## Hyper parameters
config         = yaml.load(open('./configs/config_skin_isic2017_New.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
best_Dice = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = config['path_to_data']


train_dataset = isic_loader(path_Data = data_path, train = True)
train_loader  = DataLoader(train_dataset, batch_size = int(config['batch_size_tr']), shuffle= True)

val_dataset   = isic_loader(path_Data = data_path, train = False,Test=config['is_test'])
val_loader    = DataLoader(val_dataset, batch_size = int(config['batch_size_va']), shuffle= False)

test_dataset = isic_loader(path_Data = data_path, train = False, Test = True)
test_loader  = DataLoader(test_dataset, batch_size = 1, shuffle= True)


# In[3]:
model_name = 'LGANet'
Net = LGANet(channel=32,n_classes = number_classes)

Net = Net.to(device)
if int(config['pretrained']):
    Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
    best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']
optimizer = optim.Adam(Net.parameters(), lr= float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = config['patience'])
criteria  = torch.nn.BCELoss()
# criteria_boundary  = torch.nn.BCELoss()
# criteria_region = torch.nn.MSELoss()


# In[ ]:
def test(model):
    with torch.no_grad():
        print('test_mode')
        model.eval()
        num1 = len(test_loader)
        DSC = 0.0
        for itter, batch in tqdm(enumerate(test_loader)):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask']
            # msk_pred,side_out = model(img)
            msk_pred,_,_,_ = model(img)

            msk_pred = torch.sigmoid(msk_pred)
            # eval Dice
            msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
            msk_pred  = np.where(msk_pred>=0.5, 1, 0)

            input = msk_pred
            target = msk.numpy()[0, 0]

            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

    return DSC / num1

for ep in range(int(config['epochs'])):
    Net.train()
    epoch_loss = 0
    for itter, batch in enumerate(train_loader):
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask'].to(device)

        mask_type = torch.float32
        msk = msk.to(device=device, dtype=mask_type)
        msk_pool2 = rearrange(msk, 'b c (h n) (w m) -> b c h w (n m)', h=8, w=8) #patch_size=4
        msk_pool2 = torch.sum(msk_pool2,dim=-1)
        msk_pool2[msk_pool2==0] = 0
        msk_pool2[msk_pool2==16] = 0
        msk_pool2[msk_pool2 > 0] = 1

        msk_pool3 = rearrange(msk, 'b c (h n) (w m) -> b c h w (n m)', h=4, w=4) #patch_size=4
        msk_pool3 = torch.sum(msk_pool3,dim=-1)
        msk_pool3[msk_pool3==0] = 0
        msk_pool3[msk_pool3==16] = 0
        msk_pool3[msk_pool3 > 0] = 1

        msk_pool4 = rearrange(msk, 'b c (h n) (w m) -> b c h w (n m)', h=2, w=2) #patch_size=4
        msk_pool4 = torch.sum(msk_pool4,dim=-1)
        msk_pool4[msk_pool4==0] = 0
        msk_pool4[msk_pool4==16] = 0
        msk_pool4[msk_pool4 > 0] = 1

        msk_pred,s2,s3,s4 = Net(img)

        # msk_pred = torch.sigmoid(msk_pred)
        loss_seg = structure_loss(msk_pred, msk)
        loss_score2 = criteria(s2,msk_pool2)
        loss_score3 = criteria(s3,msk_pool3)
        loss_score4 = criteria(s4,msk_pool4)

        tloss  = 0.7*loss_seg + 0.1*loss_score2 + 0.1*loss_score3 + 0.1*loss_score4
        optimizer.zero_grad()
        tloss.backward()
        optimizer.step()
        epoch_loss += loss_seg.item()
        if itter%int(float(config['progress_p']) * len(train_loader))==0:
            print(f' Epoch>> {ep+1} and itteration {itter+1} Loss>> {((epoch_loss/(itter+1)))}')

    #测试
    if ep >= 0:
        Dice = test(Net)
        if Dice > best_Dice:
            best_Dice = Dice
            save_best_dice_model_path = './weights/'+model_name+ config['weight_path'] + 'best_dice_weight_path/'
            if not os.path.exists(save_best_dice_model_path):
                os.makedirs(save_best_dice_model_path)
            torch.save(Net.state_dict(), save_best_dice_model_path +'epoch'+str(ep+1)+ '.pth')
            torch.save(Net.state_dict(), save_best_dice_model_path +'best_dice.pth')
            print(f' test on epoch>> {ep+1} dice >> {(Dice)}')
            print('New best dice, saving...')

    ## Validation phase
    with torch.no_grad():
        print('val_mode')
        val_loss = 0
        Net.eval()
        for itter, batch in enumerate(val_loader):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            # msk_pred,side_out = Net(img)
            msk_pred,_,_,_ = Net(img)
            # msk_pred = torch.sigmoid(msk_pred)
            loss = structure_loss(msk_pred, msk)
            val_loss += loss.item()
        print(f' validation on epoch>> {ep+1} dice loss>> {(abs(val_loss/(itter+1)))}')
        mean_val_loss = (val_loss/(itter+1))
        # Check the performance and save the model
        if (mean_val_loss) < best_val_loss:
            print('New best loss, saving...')
            best_val_loss = copy.deepcopy(mean_val_loss)
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'val_loss': best_val_loss})

            save_best_loss_model_path = './weights/'+model_name+ config['weight_path'] +'best_loss_weight_path/'
            if not os.path.exists(save_best_loss_model_path):
                os.makedirs(save_best_loss_model_path)

            torch.save(state,save_best_loss_model_path+ str(ep+1)+'.model')
            torch.save(state, save_best_loss_model_path+config['saved_model'])

    scheduler.step(mean_val_loss)
    
print('Trainng phase finished')    

