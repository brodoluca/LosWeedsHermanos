import torch
import cv2
import os
import pandas as pd
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import timm
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224),antialias=None),
        ]
        )


path = "./Data/"
label_path = "VID_20230517_115928.csv"
folder_path = path + label_path.split(".")[0]+"/"


label =  pd.read_csv(path+label_path)


radius = 50
color_gt = [255,0,0]
color_pred = [0,255,0]
thickness = -1



#loading model
#model  = models.resnet50()
#num_features = model.fc.in_features
#model.fc = nn.Linear(num_features, 2)


model = timm.create_model('mobilenetv3_large_100')
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)

model_check = torch.load("./mobilenet_point.pth",map_location=torch.device('cpu'))
model.load_state_dict(model_check)
model.eval()


for image,x,y in zip(label['Image'],label['X'],label['Y']):
    img = cv2.imread(folder_path + image)
    #centerOfCircle = label
    img = cv2.circle(img, (x,y), radius, color_gt, thickness)
    predicted_center = model(transform(img).unsqueeze(0)).detach().numpy()
    
    #print(np.int_(predicted_center[0]))
    img = cv2.circle(img, np.int_(predicted_center[0]), radius, color_pred, thickness)
    cv2.imshow("IMage", img)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()
   