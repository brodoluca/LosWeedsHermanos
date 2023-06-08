import os
import numpy as np
import cv2
import time
import glob
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms
import shutil
from matplotlib import pyplot as plt
import ast

class PointSet(Dataset):
    def __init__(self, transform, path=None,label_path = None, preprocess = True) -> None:
            self.path = path if path[-1] == "/" else path+"/"
            self.names = os.listdir(path)
            self.names.sort()
            
            self.label =  pd.read_csv(label_path)

            #self.label['X'] = self.label['X'].apply(ast.literal_eval)
            #self.label['Y'] = self.label['Y'].apply(ast.literal_eval)
            
            self.transform = transform
            
            self.length = len(self.label['Image'])
    def __len__(self):
          return self.length

    def __getitem__(self, index):
        name = self.label['Image'][index]
        img_path = self.path +name
        img = self.transform(Image.open(img_path)) if self.transform != None else Image.open(img_path)

        label = [self.label['X'][index], self.label['Y'][index]]

        return img, label
    



class CompletePointSet(Dataset):
    def __init__(self, label_path,transform, ) -> None:
            self.path  = ""
            for x in label_path.split('/')[:-1]:
                 self.path+=x+"/"
            self.path = self.path if self.path[-1] == "/" else self.path+"/"
            self.label =  pd.read_csv(label_path)
            self.transform = transform
            
            self.length = len(self.label.index)
    def __len__(self):
          return self.length-1

    def __getitem__(self, index):
        #print(index)
        name = self.label['Image'][index]
        folder = self.label['Folder'][index]
        img_path = self.path +folder +"/"+name
        img = self.transform(Image.open(img_path)) if self.transform != None else Image.open(img_path)

        label = [self.label['X'][index], self.label['Y'][index]]
        #print(label)
        return img, label



class NameSet(Dataset):
        def __init__(self, path, transform = None) -> None:
                super().__init__()
                self.path = path
                self.transform = transform
                self.images_names = os.listdir(path)

        def __len__(self):
              return len(self.images_names)
        def __getitem__(self, index):
                im_name = self.images_names[index]
                x, y,_ = im_name.split("_")
                img_path = self.path + im_name
                img = self.transform(Image.open(img_path)) if self.transform != None else Image.open(img_path)
                return img, np.array([x, y])

if __name__ == "__main__":
    #test_set  = PointSet(path="./Data/VID_20230517_115928",label_path="./Data/VID_20230517_115928.csv", transform=None)
    test_set = NameSet("./road_following_data_gathering_final_A/apex/")
    print(len(test_set))
    #print(test_set[-1])
    for idx,_ in enumerate(test_set):
       #print(idx)
       pass
       #print(te[1][1])

