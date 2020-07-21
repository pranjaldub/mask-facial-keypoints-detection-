import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # This network takes in a square (same width and height), grayscale image as input
        # and it ends with a linear layer that represents the keypoints
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=2, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)  
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1)  
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=512, kernel_size=(3, 3), stride=1, padding=0)  
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=0)  
        # maxpooling layers, multiple conv layers, fully-connected layers,
        # and other layers (such as dropout) to avoid overfitting
        # max-pool layer 
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # linear layers
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=136)
        # dropout 
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout6 = nn.Dropout(p=0.6)      
    
    def forward(self, x):
        # x is the input image
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout2(x)
        # flatten
        x = x.view(x.size(0), -1) 
        # fully connected layers
        x = F.elu(self.fc1(x))
        x = self.dropout4(x)
        x = F.elu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)        
        # a modified x, having gone through all the layers of your model
        return x