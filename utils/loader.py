from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from einops.layers.torch import Rearrange
from scipy.ndimage.morphology import binary_dilation

# ===== normalize over the dataset 
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized
       
    
class weak_annotation(torch.nn.Module):
    def __init__(self, patch_size = 16, img_size = 256):
        super().__init__()
        self.arranger = Rearrange('c (ph h) (pw w) -> c (ph pw) h w', c=1, h=patch_size, ph=img_size//patch_size, w=patch_size, pw=img_size//patch_size)
    def forward(self, x):
        x = self.arranger(x)
        x = torch.sum(x, dim = [-2, -1])     #每个区域的像素求和.
        x = x/x.max()  #归一化
        return x
    
def Bextraction(img):
    img = img[0].numpy()
    img2 = binary_dilation(img, structure=np.ones((7,7))).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis = 0)
    return torch.tensor(img3.copy())

## Temporary
class isic_loader(Dataset):
    """ dataset class for Brats datasets
    """
    def __init__(self, path_Data, train = True, Test = False):
        super(isic_loader, self)
        self.train = train
        if train:
          self.data   = np.load(path_Data+'data_train.npy')
          self.mask   = np.load(path_Data+'mask_train.npy')
        else:
          if Test:
            self.data   = np.load(path_Data+'data_test.npy')
            self.mask   = np.load(path_Data+'mask_test.npy')
          else:
            self.data   = np.load(path_Data+'data_val.npy')
            self.mask   = np.load(path_Data+'mask_val.npy')          
          
          
        self.data   = dataset_normalized(self.data)
        self.mask   = np.expand_dims(self.mask, axis=3)
        # self.mask   = self.mask /255.
        # self.weak_annotation = weak_annotation(patch_size = 16, img_size = 256)   #区域重要性系数的监督
         
    def __getitem__(self, indx):
        img = self.data[indx]
        seg = self.mask[indx]
        seg[seg >= 0.5] = 1
        seg[seg <= 0.5] = 0
        if self.train:
            img, seg = self.apply_augmentation(img, seg)
        
        seg = torch.tensor(seg.copy())
        img = torch.tensor(img.copy())
        img = img.permute( 2, 0, 1)
        seg = seg.permute( 2, 0, 1)
        
        # weak_ann = self.weak_annotation(seg)
        # boundary = Bextraction(seg)
        return {'image': img,
                # 'weak_ann': weak_ann,
                # 'boundary': boundary,
                'mask' : seg,
                'index':indx}
               
    def apply_augmentation(self, img, seg):
        if random.random() < 0.5:
            img  = np.flip(img,  axis=1)
            seg  = np.flip(seg,  axis=1)
        return img, seg

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # train_dataset = isic_loader(path_Data = './dataset_isic17/processed_data/isic2017/', train = False, Test = True)
    # train_loader  = DataLoader(train_dataset, batch_size = 2, shuffle= True)
    # for itter, batch in enumerate(train_loader):
    #     if itter == 0:
    #         img = batch['image']
    #         msk = batch['mask']
    #         weak_ann = batch['weak_ann']
    #         boundary = batch['boundary']
    #         img1 = img[0].permute( 1,2,0).numpy().astype(np.int8)
    #         plt.subplot(2,2,1)
    #         plt.imshow(img1)
    #         plt.subplot(2,2,2)
    #         plt.imshow(msk[0].permute( 1,2,0).numpy().squeeze(),cmap="gray")
    #         plt.subplot(2,2,3)
    #         plt.imshow(boundary[0].permute( 1,2,0).numpy().squeeze(),cmap="gray")
    #         plt.show()
    path_Data = '/media/qing/My Book/paper_code/Skin_Segmentation/dataset/Skin Dataset/ISIC2017/processed_data/isic2017/'
    data  = np.load(path_Data+'data_test.npy')
    print(np.max(data))
    data_norm = dataset_normalized(data)
    print(np.max(data_norm))
    mask   = np.load(path_Data+'mask_test.npy')
    mask   = np.expand_dims(mask, axis=3)
    print(np.max(mask))
    mask   = mask /255.

    img = mask[2].transpose( 2, 0, 1)
    print(np.max(img))
    img1 = img[0]
    img2 = binary_dilation(img1, structure=np.ones((7,7))).astype(img.dtype)
    img3 = img2 - img
    print(img3.shape)
    print(np.max(img3))
    plt.subplot(1,2,1)
    plt.title('boudary')
    plt.imshow(img3[0],cmap="gray")
    plt.subplot(1,2,2)
    plt.title('mask')
    plt.imshow(img[0],cmap="gray")
    # plt.subplot(2,2,3)
    # plt.imshow(img2,cmap="gray")
    plt.show()

