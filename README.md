# Train and Inference Super Resolution by Deep Neural Networks (SRCNN ResNet and Gans: EDSR, SRGAN, WDSR ) for 2D/3D Photo and 2D/3D Rock Physics images


## Introduction
Rewritten by me unification project based on forks of https://github.com/yingDaWang-UNSW/GUI_Prototype_SR and https://github.com/yingDaWang-UNSW/EDSRGAN-3D  for doing end-to-end Train and Inference Super resolution of both 2D and 3D images using SRCNN Resnet, SRGANs: ESRGAN, SRGAN, WDSRGAN. 
For detail read Wang Ying Da PH.D THESIS  Machine Learning Methods and Computationally Efficient Techniques in Digital Rock Analysis, 2020 and all related articles.





## Authors
[YingDa Wang](https://github.com/yingDaWang-UNSW "GitHub Account")<br>
with several small modification by USA_VS for END-to-END Train and Inference SRDNN.
Please put a star to https://github.com/yingDaWang-UNSW to support his work.


Preparation of Env for Train

DNN  use old Tensorflow 1.12 with Tensorlayer 1.11 (deleted at 2017.03.02).

Let consider Mac OS X, another platforms require similar dependency.

Root folder of project './GAN_PhysRock&Roll/' 

1. run sh script  installSRMac.sh

installSRMac.sh 

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-MacOSX-x86_64.sh;
bash Anaconda3-2019.03-MacOSX-x86_64.sh -b;

~/anaconda/bin/conda init bash;
source ~/.bash_profile; # just in case;

conda create --name srRockEnv;

source ~/anaconda/etc/profile.d/conda.sh;
conda activate srRockEnv;
conda info --envs;


conda install tensorflow=1.12; 
conda install pillow;
conda install -c conda-forge gooey;
pip install tensorlayer==1.11;
```


2. Convert your image using script runSRpconversion.sh
```
source ~/anaconda/etc/profile.d/conda.sh;
conda activate srRockEnv;
we consider case when all project saved in  '/GAN_PhysRock&Roll/' folder
first convert image files to numpy array,
High Resolution Images for train stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_train_HR'
 With names 0001-9999
Low Resolution Images stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_train_unknown_X4'
'unknown' in folder name because we not defined Upsampling method argument --downgrade  by default='unknown'
High Resolution Images for valid stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_valid_HR'
Low Resolution Images for valid stored in 'DRSRD3/DRSRD3_3D/shuffled3D/GAN_valid_unknown_X4'
train and validation ids respesented by arguments --trainIDs 1-9 --valIDs 10-10

pythonw sr3dydw.py --preprocess True  --trainIDs 1-9 --valIDs 10-10 --dataset '../GAN_PhysRock&Roll/Shuffled3D_BIN' --outdir '../GAN_PhysRock&Roll/shuffled3D_BIN'  --indir '../GAN_PhysRock&Roll/DRSRD3/DRSRD3_3D/shuffled3D';
```

For example 0001x4.png,0002x4.png, ..., 0010x4.png file of 4 times downsampled images of size 510x339 and 0001.png,0002.png, ..., 0010.png file of original high resolution images of size 2040x1356.


3.Train using your images. Below train on toy example, 9 image for train and 1 to validation from Div2k set




Train setting DIV2K images USIGN 2D (valH * valW) SUPER RESOLUTION
```
pythonw sr3dydw.py --train True --batch-size 4 --depth 1 --iterMax 100 --valW 1356  --valH 2040  --trainIDs 1-8 --valIDs 9-10;
```
FOR TRAIN 3D  (valH * valW * depth)
```
pythonw sr3dydw.py --train True --batch-size 4 --valW Y --valH X  --depth Z --trainIDs '1-8' --valIDs '8-10' ;
```



After training  network weight stored in 3 files:
epoch-250-PSNR-26.706900875139965.ckpt.meta, epoch-250-PSNR-26.706900875139965.ckpt.index, epoch-250-PSNR-26.706900875139965.ckpt.data-00000-of-00001.



Copy trained weigh files to '/validatedCheckpoints' folder,


replace in  sr3dydwTestOnly.py line with model restore  (170 to 200)
```
'for example ResNet architecture (not GAN setting)
   if dim == '2D Photo':
        flatFlag = True
        if ganFlag:
            restore = './validatedCheckpoints/SRGAN2DRock.ckpt'
        else:
            restore = './validatedCheckpoints/epoch-250-PSNR-26.706900875139965.ckpt'
```         
 Toy trained model using 9 images and validation using 1 image from div2K dataset.
Run  download_2k_dataset.sh to download  DIV2K full dataset.




4. Use inference Gui application


runSRInference.sh 

source ~/anaconda/etc/profile.d/conda.sh;
conda activate srRockEnv;
pythonw sr3dydwTestOnly.py;


The folder containing the Trained models and testing images is here: https://drive.google.com/file/d/13o3Vz65YlByJjw8zMvX0C11TXCvKGOXS/view?usp=sharing



##Running Guideline##

A desktop shortcut icon should have been created as part of the installation. Please double click it. In case we can't get that to work:<br>
Open a terminal window at the directory where the file “runProgram.sh” is located and type “bash runProgram.sh”.

**Input Format**
There are 5 different formats of input image: .png, .jpg, .mat, .nc, .tif.

**2D Images**
All input 5 input formats listed above are acceptable for 2D images resolution.

**3D Images**
The format of .mat, .nc and .tif are acceptable for 3D image super resolution.

|parameter|usage|
|----------| :-------: |
|Input images|Folder which containing input images, named “srtestfolder”|
|Input format|The format of input image|
|Scale factor|Up sampling factor|
|Output format|The format of output image|
|Bit depth|Bit depth of your input images|
|Image dimension|2D or 3D dimension of input images|
|Use CPU|Force the network to use the CPU as default, if no compatible GPU is detected|
|Use GAN|Force the network to use the GAN as default|

**Output format and Visualisation**

The output images are stored in “srtestfolder\srOutputs”.

For 2D images, output is slice-by-slice, and readable my standard image reading software.

For 3D images, we recommend ImageJ: https://imagej.net/Fiji/Downloads

**Examples**

***2D Images***

1. choose the folder containing the input images on your computer;

2. choose the up-sampling scale factor you want, such as 4, 16 or 64;

3. when you choose to resolution 2D images, it may not be necessary to use 3D patches;

4. there are only two types of 2D output formats, “.png” and “.jpg”, could be used;

5. bit depth: uint8 means unsigned 8-bits integer;

6. choose 2D as image dimension;

7. if you choose “yes” as “USE CPU”, the software will force the network to use CPU rather than GPU. And CPU will as default if you do not choose;

8. there are two types of neural network could be provided, such as CNN and GAN. And the CNN will as default if you do not choose;

9. determine to use checkpoints for training 2D images or not.

10. Start to resolution your images!

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step2d.png)

***3D Images***

The steps for 3D images are almost similar with those for 2D images, except:

-	use 3D patches as recommended for 3D images;

-	there are three types of 3D output formats, “.mat”, “.nc” and “.tiff”, could be used;

-	choose 3D as image dimension;

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step3d.png)
