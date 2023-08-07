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
# from utils.loader_TMUNet import *
import utils.metrics as metrics
from medpy.metric.binary import hd, dc, assd, jc,hd95
import glob
import numpy as np
import copy
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from models.model import LGANet



# In[2]:


## Hyper parameters
config         = yaml.load(open('../configs/config_skin_isic2018.yml'), Loader=yaml.FullLoader)
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
Net = LGANet(n_classes = number_classes)
Net = Net.to(device)
Net.load_state_dict(torch.load('./weights/LGANet/isic18/best_loss_weight_path/weights_isic18.model', map_location='cpu')['model_weights'])



predictions = []
gt = []

with torch.no_grad():
    print('val_mode')
    val_loss = 0
    numm = 0
    HD95_value = 0
    ASSD_value = 0
    dice_value = 0
    iou_value = 0
    SE = 0
    SP = 0
    ACC = 0
    Pre = 0

    Net.eval()
    for itter, batch in tqdm(enumerate(test_loader)):
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask']
        msk_pred,_,_,_ = Net(img)
        # msk_pred = Net(img)
        # print(torch.max(msk_pred))
        msk_pred = torch.sigmoid(msk_pred)

        gt.append(msk.numpy()[0, 0])
        msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
        predictions.append(msk_pred)
        mask_pred = np.where(msk_pred>0.5, 1, 0)
        label = msk.cpu().numpy()[0, 0] > 0.5

        #因为霍夫距离中要求不能只有一个值
        if np.count_nonzero(label) == 0 or np.count_nonzero(mask_pred) == 0:
            HD95 = 0.
            ASSD = 0.
        else:
            HD95 = hd95(mask_pred, label)
            ASSD = assd(mask_pred, label)
        HD95_value += HD95
        ASSD_value += ASSD
        numm += 1

# 衡量边界的指标
HD95_ave = HD95_value / numm
ASSD_ave = ASSD_value / numm

print("HD95_ave: " + str(HD95_ave))
print("ASSD_ave: " + str(ASSD_ave))


predictions = np.array(predictions)
gt = np.array(gt)

y_scores = predictions.reshape(-1)
y_true   = gt.reshape(-1)

y_scores2 = np.where(y_scores>0.5, 1, 0)
y_true2   = np.where(y_true>0.5, 1, 0)

#F1 score
F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
print ("\nF1 score (F-measure) or DSC: " +str(F1_score))

confusion = confusion_matrix(np.int32(y_true2), y_scores2)
print (confusion)
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print ("Accuracy: " +str(accuracy))
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print ("Specificity: " +str(specificity))
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print ("Sensitivity: " +str(sensitivity))
Precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    Precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print("Precision: " +str(Precision))
IOU = 0
if float(np.sum(confusion)-confusion[0,0])!=0:
    IOU = float(confusion[1,1]) / float(np.sum(confusion)-confusion[0,0])
print("IOU: " +str(IOU))



