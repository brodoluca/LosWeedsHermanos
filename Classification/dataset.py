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

import utils

class TrackSet(Dataset):
    def __init__(self, transform, path=None, preprocess = True) -> None:
            self.path = path if path[-1] == "/" else path+"/"
            self.names = os.listdir(path)
            self.names.sort()
            self.length = len(self.names)
            
            self.transform = transform
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        name = self.names[index]
        steering, throttle = int(name.split("_")[0])/10,int(name.split("_")[1])
        img_path = self.path +name
        img = self.transform(Image.open(img_path))
        #print(steering, throttle)
        label = utils.transl_action_env2agent(np.array([[steering, throttle]]),False)#transforms.ToTensor()(), False))
        #print(label)
        return img, label[0]
    
    
class AnasTrackSet(Dataset):
    def __init__(self, transform, path=None,label_path = None, preprocess = True) -> None:
            self.path = path if path[-1] == "/" else path+"/"
            self.names = os.listdir(path)
            self.names.sort()
            
            self.label =  pd.read_csv(label_path)
            self.label['action'] = self.label['action'].apply(ast.literal_eval)
            self.transform = transform
            
            self.length = len(self.label['action'])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        name = self.label['name'][index]
        img_path = self.path +name
        img = self.transform(Image.open(img_path)) if self.transform != None else Image.open(img_path)
        #print(steering, throttle)
        #print(self.label['action'][index])
        label = utils.transl_action_env2agent(np.array(self.label['action'][index]),False)#transforms.ToTensor()(), False))
        #print(label.shape)
        return img, label[0]
    

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
    def __init__(self, label_path,transform ) -> None:
            self.path  = ""
            for x in label_path.split('/')[:-1]:
                 self.path+=x+"/"
            self.path = self.path if self.path[-1] == "/" else self.path+"/"
            self.label =  pd.read_csv(label_path)
            self.transform = transform
            
            self.length = len(self.label.index)
            
            self.img_paths = []
            for name, folder in zip(self.label['Image'], self.label['Folder']):
                img_path = self.path +folder +"/"+name
                self.img_paths += [img_path]
            #print(self.img_paths)
    def __len__(self):
          return self.length

    def __getitem__(self, index):
        #print(index)
        #name = self.label['Image'][index]
        #folder = self.label['Folder'][index]
        #img_path = self.path +folder +"/"+name
        img_path = self.img_paths[index]
        img = self.transform(Image.open(img_path)) if self.transform != None else Image.open(img_path)

        label = np.array([self.label['X'][index], self.label['Y'][index]])
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
                img = self.transform(Image.open(img_path)) # if self.transform != None else Image.open(img_path)
                #print(type(img), x,y, index, im_name)
                return img, torch.tensor([int(x), int(y)])
