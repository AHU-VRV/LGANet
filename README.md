

# LGANet: Local-Global Augmentation Network for Skin Lesion Segmentation
This repo is the official implementation  for the paper: **"LGANet: Local-Global Augmentation Network for Skin Lesion Segmentation"** at ISBI 2023.

This paper proposes a novel framework, LGANet, for skin lesion segmentation. Particularly, two module, LFM and GAM are constructed. LFM aims at learning local inter-pixel correlations to augment local detailed information around boundary regions, while GAM aims at learning global context at a finer level to augment global information.
## Architecture
![Network](https://img-blog.csdnimg.cn/bf41c11f82ec4cd382d3dd916829de98.png#pic_center)Fig.2. The structure of the proposed LGANet. LFM and GAM are integrated into the Transformer encoder based framework to learn local detailed information around boundary and augment global context respectively, where dense concatenations are used for final pixel-level prediction.



## Requirements

 - Python 3.6
 - pytorch 1.6.0
 - torchvision 0.7.0
 - einops 0.3.0
 - numpy 1.16.6
 - scipy 1.2.1
 - tqdm 4.61.0
 - yaml 0.2.5

## Datasets

 - The ISIC 2018  and ISIC 2016 dataset can be acquired from [the official site](https://challenge.isic-archive.com/data/).
 - Run Prepare_data.py for data preperation.


## Pretrained model
You could download the pretrained model from [here](https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV).  Please put it in the " **./pretrained**" folder for initialization.
## Training
python  train.py
## Testing
python test.py
## Evaluation
python evaluate.py

## References
Some of the codes in this repo are borrowed from:
 - [TMUNet](https://github.com/rezazad68/TMUnet)     
 - [PVT](https://github.com/whai362/PVT)


 

