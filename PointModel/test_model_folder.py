import torch
import cv2
import os
import pandas as pd
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import timm

#from torch2trt import torch2trt
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224),antialias=None),
        ]
        )


path = "./road_following_data_gathering_final_A/apex/"

images = os.listdir(path)
images.sort()




radius = 10
color_gt = [255,0,0]
color_pred = [0,255,0]
sach_color_pred = [0,0,255]
thickness = -1

def get_model():
#loading model
    model  = models.resnet18()
    num_features = model.fc.in_features
    #model.fc = nn.Linear(num_features, 2)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2),
        #nn.ReLU(),
    )
    return model

model = timm.create_model('mobilenetv3_large_100')
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)
#model = get_model()

sach_model = get_model()
sach_model.fc = nn.Linear(512, 2)
model_check = torch.load("./mobilenet_points_final.pth",map_location=torch.device('cpu'))
sach_model_check = torch.load("./road_following_model9.pth",map_location=torch.device('cpu'))
#print(model_check.keys())
sach_model.load_state_dict(sach_model_check)
sach_model.eval()

model.load_state_dict(model_check)
model.eval()


for image in images:
    img = cv2.imread(path + image)
    x, y,_ = image.split("_")
    print(x,y)
    #centerOfCircle = label
    predicted_center = model(transform(img).unsqueeze(0)).detach().numpy()
    sac_predicted_center = sach_model(transform(img).unsqueeze(0)).detach().numpy()
    img = cv2.circle(img, (int(x),int(y)), radius, color_gt, thickness)
    
    x,y = sac_predicted_center[0]
    x = int(224 * (x / 2.0 + 0.5))
    y = int(224 * (y / 2.0 + 0.5))

    xp,yp = predicted_center[0]
    #xp = int(224 * (xp / 2.0 + 0.5))
    #yp = int(224 * (yp/ 2.0 + 0.5))
    xp = int(xp)
    yp = int(yp)
    print(sac_predicted_center)


    img = cv2.circle(img, (xp,yp), radius, color_pred, thickness)
    img = cv2.circle(img, (y,x), radius, sach_color_pred, thickness)
    cv2.imshow("IMage", img)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()
   