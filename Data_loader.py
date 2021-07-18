import Configure as Con;
import torch as th;
import torch.nn as nn;
import torch.nn.functional as F;
from torchvision import datasets,transforms;
from torch.utils.data import Dataset,DataLoader,TensorDataset;
from PIL import Image;
import os;
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;


class Custom_data_loader():

    def __init__(self,type_):

        path = 'C:/Users/ablyv/Documents/Web Devlop/Autonise/WEEK6_VE';
        self.type_ = type_;
        self.classes = sorted(os.listdir('/'.join([path,Con.base[self.type_]])));
        self.image_paths=[];
        self.targets=[];

        for i,div in enumerate(self.classes):
            imgs = sorted(os.listdir('/'.join([path,Con.base[self.type_],div,'image'])));

            # if self.type_=='train':
            #     imgs = imgs[0:int(0.9*len(imgs))];
            # else:
            #     imgs = imgs[int(0.9*len(imgs)):];

            for img in imgs:
                self.image_paths.append('/'.join([path,Con.base[self.type_],div,'image',img]));
                self.targets.append(i);


    def Image_crop(self,image):
        x,y=np.where(image!=0)
        maxX,minX,maxY,minY = max(x),min(x),max(y),min(y);
        return image[minX:maxX,minY:maxY];

    def Re_resize(self,image):
        # fimage = np.zeros([Con.resz_fig,Con.resz_fig]);
        lenX , lenY = image.shape;
        maxlen = max(lenX, lenY);
        fimage = np.zeros([maxlen, maxlen]);
        y_min = (maxlen-image.shape[0])//2;
        y_max = y_min + image.shape[0];
        x_min = (maxlen-image.shape[1])//2;
        x_max = x_min +image.shape[1];
        fimage[y_min:y_max,x_min:x_max]=image;
        fimage = Image.fromarray(fimage);
        fimage = fimage.resize([Con.resz_fig,Con.resz_fig]);
        fimage = (np.array(fimage)>0.1).astype(np.float32);

        # padLeft =int((Con.resz_fig-lenX)/2);
        # padRight = int((Con.resz_fig - lenY) / 2);
        # fimage[padLeft:(padLeft+lenX),padRight:(padRight+lenY)]= image;
        return fimage;

    def processing(self,paths):
        img_gs=plt.imread(paths)[:,:,3]; # coversion to gray scale image by just choosing the 4th channel of opacity;
        # since we have to resize it we have to convert it into the PIL image
        img_pil  =Image.fromarray(img_gs); # converted to pil image ,it will show you 4 channels again,
        # opacity channel will be entirely set to 255 , and img_gs 1d element will be replicated thrice(will be in between 0 and 1 )
        #in this case the entire image will be read from a scale of 0 - 255.
        # since RGB value is in 0-1 hence the image will be entirely balck.
        img_rsz = img_pil.resize((Con.resz_fig,Con.resz_fig));
        # thresholding is important else that image would be blured out and would affect the accuracy;
        # lets have a threshold of 0.1;
        img_rsz = np.array(img_rsz);
        fimage=(img_rsz>0.1).astype(np.float32);## this may give accuracy of 90%
        # but in order to increase the accuracy to even further ,
        # we can maginify the element to entire range of nearly Con.resz_fig x Con.resz_fig;
        # hence we have to first crop the image;
        fimage=self.Image_crop(fimage);
        fimage = self.Re_resize(fimage);
        return fimage;


    def __getitem__(self,item):
        # print(self.image_paths[item]);
        image = self.processing(self.image_paths[item]);
        return image[None,:,:], self.targets[item],self.image_paths[item];

    def __len__(self):
        return len(self.image_paths);

def getDataLoader(type_="train"):
    return DataLoader(
        Custom_data_loader(type_=type_ ),
        batch_size = Con.batchSize[type_],
        num_workers=Con.numworkers[type_],
        shuffle= type_=='train'
    )




