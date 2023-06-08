import torch
import cv2
import os
import pandas as pd
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import timm
import cv2
import utils
def add_text_to_image(image, text):

    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    font_thickness = 3

    # Determine the text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Set the text position
    text_position = (10, text_size[1] + 10)  # Offset from the top-left corner

    # Add the text to the image
    cv2.putText(image, text, text_position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
    return image

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224),antialias=None),
        ]
        )


data_path = "./road_following_data_gathering_final_A/apex/"
model_path = "./models/mobilenet_cat1-2.pth"
images = os.listdir(data_path)
images.sort()

#model = models.resnet.resnet50()
#num_features = model.fc.in_features
#model.fc = nn.Linear(num_features, 25)

model = timm.create_model('mobilenetv3_large_100', pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 25)

model_check = torch.load(model_path,map_location=torch.device('cpu'))
model.load_state_dict(model_check)
model.eval()


for image_name in images:
    image_path = data_path+image_name
    image = cv2.imread(image_path)
    

    preds = model( transform(image).unsqueeze(0))
    idx = torch.argmax(preds).item()
    action = utils.possible_actions[idx]
    text = f"{action[0]}, {action[1]}"
    image = add_text_to_image(image, text)

    cv2.imshow("Image with Text", image)
    if cv2.waitKey(0) == ord('q'):
        break
cv2.destroyAllWindows()

