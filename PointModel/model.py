import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features2 = nn.Sequential(   
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.features3 = nn.Sequential(   
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.features4 = nn.Sequential(   
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features1(x)
        #print(x.shape) 
        x = self.features2(x)
        #print(x.shape) 
        x = self.features3(x)
        #print(x.shape) 
        x = self.features4(x)
        #print(x.shape) 
        x = x.view(x.size(0), -1)
        print(x.shape) 
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn((63,3,224,224))
    model = SimpleCNN(4)
    model(x)