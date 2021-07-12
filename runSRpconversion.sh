#!/bin/bash
reset;
source ~/anaconda/etc/profile.d/conda.sh;
conda activate srRockEnv;
#we consider case when all project saved in  '/GAN_PhysRock&Roll/' folder
#first convert image files to numpy array,
#High Resolution Images for train stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_train_HR'
# With names 0001-9999
#Low Resolution Images stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_train_unknown_X4'
#'unknown' in folder name because we not defined Upsampling method argument --downgrade  by default='unknown'
#High Resolution Images for valid stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_valid_HR'
#Low Resolution Images for valid stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_valid_unknown_X4'
# train and validation ids respesented by arguments --trainIDs 1-9 --valIDs 10-10

pythonw sr3dydw.py --preprocess True  --trainIDs 1-9 --valIDs 10-10 --dataset '../GAN_PhysRock&Roll/Shuffled3D_BIN' --outdir '../GAN_PhysRock&Roll/shuffled3D_BIN'  --indir '../GAN_PhysRock&Roll/DRSRD3/DRSRD3_3D/shuffled3D';
#for example 0001x4.png,0002x4.png, ..., 0010x4.png file of 4 times downsampled images of size 510x339
#for example 0001.png,0002.png, ..., 0010.png file of original high resolution images of size 2040x1356

#Train setting DIV2K images USIGN 2D (valH * valW) SUPER RESOLUTION
#pythonw sr3dydw.py --train True --batch-size 4 --depth 1 --iterMax 100 --valW 1356  --valH 2040  --trainIDs 1-8 --valIDs 9-10;

#FOR TRAIN 3D  (valH * valW * depth)
#pythonw sr3dydw.py --train True --batch-size 4 --valW Y --valH X  --depth Z --trainIDs '1-8' --valIDs '8-10' ;


#after training save network weight in 3 files
# Example
# epoch-250-PSNR-26.706900875139965.ckpt.meta
# epoch-250-PSNR-26.706900875139965.ckpt.index
# epoch-250-PSNR-26.706900875139965.ckpt.data-00000-of-00001
# just copy trained weigh files to '/validatedCheckpoints' folder
# replace in  sr3dydwTestOnly.py line with model restore  (170 to 200)
# for example ResNet architecture (not GAN setting)
#   if dim == '2D Photo':
#        flatFlag = True
#        if ganFlag:
#            restore = './validatedCheckpoints/SRGAN2DRock.ckpt'
#        else:
#            restore = './validatedCheckpoints/epoch-250-PSNR-26.706900875139965.ckpt'
# Toy trained model using 9 images and validation using 1 image from div2K dataset
# and run runSRInference.sh