import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import timm


possible_actions =  np.array([
    [-0.1, 0], #move n. 0
[-0.2, 0], #move n. 1
[-0.3, 0], #move n. 2
[-0.4, 0], #move n. 3
[-0.5, 0], #move n. 4
[-0.6, 0], #move n. 5
[0, 0], #move n. 6
[0.1, 0], #move n. 7
[0.2, 0], #move n. 8
[0.3, 0], #move n. 9
[0.4, 0], #move n. 10
[0.5, 0], #move n. 11
[0.6, 0], #move n. 12
[-0.1, 1], #move n. 13
[-0.2, 1], #move n. 14
[-0.3, 1], #move n. 15
[-0.4, 1], #move n. 16
[-0.5, 1], #move n. 17
[-0.6, 1], #move n. 18
[0.0, 1], #move n. 19
[0.1, 1], #move n. 20
[0.2, 1], #move n. 21
[0.3, 1], #move n. 22
[0.4, 1], #move n. 23
[0.5, 1], #move n. 24
[0.6, 1], #move n. 25
])

n_actions = len(possible_actions)-1

def action2index(action):
    action = np.expand_dims(action, axis=0) if action.shape == (2,) else action
    # print(action)
    ids = np.where(np.all(possible_actions == action[:, np.newaxis], axis=2))[1]
    
    return ids

def action2index2(action):
    action = np.expand_dims(action, axis = 0) if action.shape == (2,) else action
    #print(action)
    ids = []
    for a in action:
        id = np.where(np.all(possible_actions==a, axis=1))
        #print(id[0])
        ids.append(id[0][0])
    return np.array(ids)

def index2action(ids):
    """ Converts action from id to array format (as understood by the environment) """
    return possible_actions[ids]


def one_hot(labels):
    """ One hot encodes a set of actions """
    one_hot_labels = np.zeros(labels.shape + (n_actions,))
    for c in range(n_actions):
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels


def transl_action_env2agent(acts, p = False):
    """ Translate actions from environment's format to agent's format """
    act_ids = action2index(acts)
    if(p):
        print(act_ids)
    return one_hot(act_ids)

def get_mobilenet(device):
    model = timm.create_model('mobilenetv3_large_100', pretrained=True).to(device)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, n_actions).to(device)
    return model
def get_resnet34(device):
    model = models.resnet34(pretrained=True).to(device)

    # Get the number of input features for the last fully connected layer
    num_features = model.fc.in_features

    # Modify the last fully connected layer for 143 classes
    model.fc = nn.Linear(num_features, n_actions).to(device)
    return model

def get_efficientnet(version='b7', pretrained=True):
    if version not in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
        raise ValueError(f"Invalid version '{version}'. Available versions are: b0, b1, b2, b3, b4, b5, b6, b7.")

    model = getattr(models, f'efficientnet_{version}')(pretrained=pretrained)
    return model

def get_efficientnet_2(device,version='b4', pretrained=True):
    if version not in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
        raise ValueError(f"Invalid version '{version}'. Available versions are: b0, b1, b2, b3, b4, b5, b6, b7.")

    model = getattr(models, f'efficientnet_{version}')(pretrained=pretrained).to(device)
    
    model.classifier =nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=1792, out_features=n_actions, bias=True)
      ).to(device)
    
    return model



def get_resnet50(device):
    model = models.resnet50(pretrained=True).to(device)

# Get the number of input features for the last fully connected layer
    num_features = model.fc.in_features

    # Modify the last fully connected layer for 143 classes
    model.fc = nn.Linear(num_features, n_actions).to(device)
    return model