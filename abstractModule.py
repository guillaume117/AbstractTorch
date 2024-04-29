import torch
import torch.nn as nn 
import copy
from typing import List, Union, Tuple


class AbstractLinear(nn.Module):
        
        def __init__(self):
    
            super(AbstractLinear,self).__init__()

        @staticmethod    
        def abstract_linear(lin:nn.Module,
                            x:torch.tensor,
                            x_true:torch.tensor,
                            device:torch.device=torch.device("cpu")):
            lin=lin.to(device)
            x =x.unsqueeze(1).to(device)
            x_true=x_true.to(device)
            lin_abs = copy.deepcopy(lin).to(device)
            lin_abs[1].weight.data =torch.abs(lin[1].weight.data)
        
            
            x_value = x[0].unsqueeze(1)
            x_epsilon= x[1:-1].unsqueeze(1)
            x_noise =x[-1].unsqueeze(1)
        
            x=lin(x)    
            x_true = lin(x_true)
            x[0]=lin(x_value)
            x[1:-1]=lin(x_epsilon)-lin(torch.zeros_like(x_epsilon))
            x[-1]=lin_abs(x_noise)-lin_abs(torch.zeros_like(x_noise))
            x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
            x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
            print(x_min.type())
            return x,x_min,x_max,x_true
        

        @staticmethod
        def abstract_conv2D(conv:nn.Module,
                            x:torch.tensor,
                            x_true:torch.tensor,
                            device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
            conv = conv.to(device)
            x=x.to(device)
            x_true = x_true.to(device)
            print(f"x.shape={x.shape}")
        
            conv_abs = copy.deepcopy(conv).to(device)
            conv_abs.weight.data = torch.abs(conv.weight.data)
        
        
            x_value = x[0]
            x_epsilon= x[1:-1]
            x_noise = x[-1]
            x=conv(x)
            x[0]=conv(x_value)
            x_true = conv(x_true)
            x[1:-1]=conv(x_epsilon)-conv(torch.zeros_like(x_epsilon).to(device))
            x[-1]=conv_abs(x_noise)-conv_abs(torch.zeros_like(x_noise).to(device))
            x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
            x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
            
            return x,x_min,x_max,x_true






class AbstractReLU(nn.Module):
    def __init__(self,max_symbols:Union[int,bool]=False):
        super(AbstractReLU,self).__init__()

    @staticmethod
    def abstract_relu(x:torch.Tensor,
                      x_min:torch.Tensor,
                      x_max:torch.Tensor,
                      x_true:torch.Tensor,
                      add_symbol:bool=False,
                      device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
        
        num_symbols = len(x)
      
        sgn_min = torch.sign(x_min)
        sgn_max = torch.sign(x_max)
        sgn = sgn_min+sgn_max
        p = x_max/(torch.abs(x_max)+torch.abs(x_min))
        q = x_max*(1-p)/2
        d = torch.abs(q)
        x_true  = nn.ReLU()(x_true)
        copy_x_for_approx = x

        #mask for values that will be approximated by the linear approximation
        mask_p = (sgn==0)*1
        #mask for the values for those the output is the same as the input (y=x)
        mask_1 =(sgn==2)*1+ (sgn==1)*1
        #expand the mask to the number of symbols
        mask_p = mask_p.unsqueeze(0).expand(num_symbols,-1)
        mask_1 = mask_1.unsqueeze(0).expand(num_symbols,-1)
        #approximation of the center  
        copy_x_for_approx[0]= mask_p[0]*(p*copy_x_for_approx[0]+q)+mask_1[0]*copy_x_for_approx[0]
        #uptade of the epsilon
        copy_x_for_approx[1:-1]=p*mask_p[1:-1]*copy_x_for_approx[1:-1] + mask_1[1:-1]*copy_x_for_approx[1:-1]
        #update of the noise symbol -> projection 0, |W|*espilon_noise or new noise if new linear approximation
        copy_x_for_approx[-1]=d*mask_p[-1] +copy_x_for_approx[-1]*mask_p[-1] + mask_1[-1]*copy_x_for_approx[-1]
        x=copy_x_for_approx

        x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
        x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
        
        if add_symbol:
            """if we want to generate new symbols, we must project each of the new symbol on a specific layer"""
            """
            new_symbols_indexes =torch.where(x[-1]!=0)


            for index,value in enumerate(new_symbols_indexes[0]):
            
            
                x =torch.cat((x,x[-1].unsqueeze(0)),dim=0)
                
                x[-2]=torch.zeros_like(x[-2])
                
                x[-2][value]=x[-1][value]
            x[-1]=torch.zeros_like(x[-1])
            
            """
            new_eps =torch.where(x[-1]!=0)[0].to(device)
            
            index = torch.arange(len(new_eps)).to(device)
            new_eps_batch_shape = x[-1].expand(len(new_eps)+1,-1).shape
            new_eps_batch = torch.zeros(new_eps_batch_shape).to(device)
            new_eps_batch[index,new_eps]=x[-1][new_eps]

            x=x[:-1]

            x = torch.cat((x,new_eps_batch),dim=0)

        return x,x_min,x_max,x_true
    

    @staticmethod
    def abstract_relu_conv2D(x:torch.tensor,
                             x_min:torch.tensor,
                             x_max:torch.tensor,
                             x_true:torch.tensor,
                             add_symbol:bool=False,
                             device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:

        num_symbols = len(x)
        sgn_min = torch.sign(x_min)
        sgn_max = torch.sign(x_max)
        sgn = sgn_min+sgn_max
        p = x_max/(torch.abs(x_max)+torch.abs(x_min))
        q = x_max*(1-p)/2
        d = torch.abs(q)
        x_true  = nn.ReLU()(x_true)
        copy_x_for_approx = x
        mask_p = (sgn==0)*1
        mask_1 =(sgn==2)*1 + (sgn==1)*1
        mask_p = mask_p.unsqueeze(0).expand(num_symbols,-1,-1,-1)
        mask_1 = mask_1.unsqueeze(0).expand(num_symbols,-1,-1,-1)
        
        #approximation of the center 0, (p*x+q) or the value itself 
        copy_x_for_approx[0]= mask_p[0]*(p*copy_x_for_approx[0]+q)+mask_1[0]*copy_x_for_approx[0]
        
        #update of the epsilon
        copy_x_for_approx[1:-1]=p*mask_p[1:-1]*copy_x_for_approx[1:-1] + mask_1[1:-1]*copy_x_for_approx[1:-1]

        #update of the noise symbol -> projection 0, |W|*espilon_noise or new noise if new linear approximation
        copy_x_for_approx[-1]=d*mask_p[-1] +copy_x_for_approx[-1]*mask_p[-1] +mask_1[-1]*copy_x_for_approx[-1]

        x=copy_x_for_approx
        
        x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
        x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
        
        if add_symbol:
            """
            new_symbols_indexes =torch.where(x[-1]!=0)


            for index,value in enumerate(new_symbols_indexes[0]):
            
            
                x =torch.cat((x,x[-1].unsqueeze(0)),dim=0)
                
                x[-2]=torch.zeros_like(x[-2])
                
                x[-2][value]=x[-1][value]
            x[-1]=torch.zeros_like(x[-1])
            """
                        
            new_eps =torch.where(x[-1]!=0)[0].to(device)
            index = torch.arange(len(new_eps)).to(device)
            new_eps_batch_shape = x[-1].expand(len(new_eps)+1,-1,-1,-1).shape
            new_eps_batch = torch.zeros(new_eps_batch_shape).to(device)
            new_eps_batch[index,new_eps]=x[-1][new_eps]

            x=x[:-1]

            x = torch.cat((x,new_eps_batch),dim=0)            

        return x,x_min,x_max,x_true



























class AbstractNN(nn.Module):
    
    def __init__(self,num_depth=1,device=torch.device("cpu")):

        super(AbstractNN,self).__init__()
       
      
        self.num_depth = num_depth
        self.device = device
        self.conv1=nn.Conv2d(self.num_depth,16,3,device=self.device)
        self.conv2=nn.Conv2d(16,16,3,device=self.device)
        self.conv3=nn.Conv2d(16,32,3,device=self.device) 
        self.conv4=nn.Conv2d(32,32,3,device=self.device)

        self.fc1=nn.Sequential(nn.Flatten(),nn.Linear(51200,6272,device=self.device))
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
        x,x_min,x_max,x_true = AbstractLinear.abstract_conv2D(self.conv1,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
       
        x,x_min,x_max,x_true = AbstractLinear.abstract_conv2D(self.conv2,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_conv2D(self.conv3,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_conv2D(self.conv4,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu_conv2D(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_linear(self.fc1,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_linear(self.fc2,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_linear(self.fc3,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_linear(self.fc4,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_linear(self.fc5,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        x,x_min,x_max,x_true = AbstractLinear.abstract_linear(self.fc6,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")

        x,x_min,x_max,x_true = AbstractLinear.abstract_linear(self.fc7,x,x_true,device=self.device)
        x,x_min,x_max,x_true = AbstractReLU.abstract_relu(x,x_min,x_max,x_true,add_symbol=add_symbol,device =self.device)
        print(f"lenx:{len(x)}")
        return x,x_min,x_max,x_true
        
