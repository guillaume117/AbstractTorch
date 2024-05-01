import torch
import torch.nn as nn
from src.abstractModule import AbstractLinear as AL
from src.abstractModule import AbstractReLU as AR


class AbstractNN2(nn.Module):
    
    def __init__(self,num_depth=1,device=torch.device("cpu")):

        super(AbstractNN2,self).__init__()
       
      
        self.num_depth = num_depth
        self.device = device
        self.conv1=nn.Conv2d(self.num_depth,16,3,device=self.device)
        self.conv2=nn.Conv2d(16,16,3,device=self.device)
        self.conv3=nn.Conv2d(16,32,3,device=self.device) 
        self.conv4=nn.Conv2d(32,32,3,device=self.device)

        self.fc1=nn.Sequential(nn.Flatten(),nn.Linear(12800,6272,device=self.device))
        self.fc2=nn.Sequential(nn.Flatten(),nn.Linear(6272,6272,device=self.device))
        self.fc3=nn.Sequential(nn.Flatten(),nn.Linear(6272,6272,device=self.device))
        self.fc4=nn.Sequential(nn.Flatten(),nn.Linear(6272,6272,device=self.device))
        self.fc5=nn.Sequential(nn.Flatten(),nn.Linear(6272,512,device=self.device))
        self.fc6=nn.Sequential(nn.Flatten(),nn.Linear(512,256,device=self.device))
        self.fc7=nn.Sequential(nn.Flatten(),nn.Linear(256,8,device=self.device))

    
    def forward(self,x,add_symbol=False,device = torch.device("cpu")):

        self.device = device
        x_true = x
        x_true = x_true[0].unsqueeze(0)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_conv2D(self.conv1,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
       
        x,x_min,x_max,x_true = AL.abstract_conv2D(self.conv2,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_conv2D(self.conv3,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_conv2D(self.conv4,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_linear(self.fc1,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_linear(self.fc2,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_linear(self.fc3,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_linear(self.fc4,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_linear(self.fc5,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AL.abstract_linear(self.fc6,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")

        x,x_min,x_max,x_true = AL.abstract_linear(self.fc7,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AR.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        return x,x_min,x_max,x_true
    

