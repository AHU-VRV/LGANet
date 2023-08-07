# -*- coding: utf-8 -*-


import numpy as np
import scipy.io as sio
import imageio
import scipy.misc as sc
import glob
from skimage.transform import resize

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2018 data set #################################################
Dataset_add = './Skin Dataset/ISIC2018_processed_data/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 2594 training samples
Data_train_2018    = np.zeros([2594, height, width, channels])
Label_train_2018   = np.zeros([2594, height, width])

print('Reading ISIC 2018')
for idx in range(len(Tr_list)):
    print(idx+1)
    img = imageio.imread(Tr_list[idx])
    # img = np.double(resize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    img = np.double(resize(img, [height, width, channels]))
    Data_train_2018[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'ISIC2018_Task1_Training_GroundTruth/' + b +'_segmentation.png')    
    img2 = imageio.imread(add)
    # img2 = np.double(resize(img2, [height, width], interp='bilinear'))
    img2 = np.double(resize(img2, [height, width]))
    Label_train_2018[idx, :,:] = img2    
         
print('Reading ISIC 2018 finished')

################################################################ Make the train and test sets ########################################    
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img      = Data_train_2018[0:1815,:,:,:]
Validation_img = Data_train_2018[1815:1815+259,:,:,:]
Test_img       = Data_train_2018[1815+259:2594,:,:,:]

Train_mask      = Label_train_2018[0:1815,:,:]
Validation_mask = Label_train_2018[1815:1815+259,:,:]
Test_mask       = Label_train_2018[1815+259:2594,:,:]


np.save(Dataset_add+'data_train', Train_img)
np.save(Dataset_add+'data_test' , Test_img)
np.save(Dataset_add+'data_val'  , Validation_img)

np.save(Dataset_add+'mask_train', Train_mask)
np.save(Dataset_add+'mask_test' , Test_mask)
np.save(Dataset_add+'mask_val'  , Validation_mask)


