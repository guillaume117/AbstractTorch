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
        x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
        x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
      
        sgn_min = torch.sign(x_min)
        sgn_max = torch.sign(x_max)
        sgn = sgn_min+sgn_max
        p = x_max/(torch.abs(x_max)+torch.abs(x_min))
        p = torch.where(torch.isnan(p),torch.zeros_like(p),p)
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
        copy_x_for_approx[-1]=d*mask_p[-1] +torch.abs(p)*mask_p[-1]*copy_x_for_approx[-1] + mask_1[-1]*copy_x_for_approx[-1]
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
        x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
        x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)

        num_symbols = len(x)
        sgn_min = torch.sign(x_min)
        sgn_max = torch.sign(x_max)
        sgn = sgn_min+sgn_max
        p = x_max/(torch.abs(x_max)+torch.abs(x_min))
        p = torch.where(torch.isnan(p),torch.zeros_like(p),p)
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
        copy_x_for_approx[-1]=d*mask_p[-1] +torch.abs(p)*mask_p[-1]*copy_x_for_approx[-1] + mask_1[-1]*copy_x_for_approx[-1]

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
         
            new_eps =torch.where(x[-1].flatten()!=0)[0].to(device)
           
            index = torch.arange(len(new_eps)).to(device)
            new_eps_batch_shape = x[-1].flatten().expand(len(new_eps)+1,-1).shape
            new_eps_batch = torch.zeros(new_eps_batch_shape).to(device)
            new_eps_batch[index,new_eps]=x[-1].flatten()[new_eps]
            new_eps_batch = new_eps_batch.reshape(x[-1].expand(len(new_eps)+1,-1,-1,-1).shape)
            

            x=x[:-1]

            x = torch.cat((x,new_eps_batch),dim=0) 
                  

        return x,x_min,x_max,x_true




class AbstractMaxpool2D(nn.Module):
    def __init__(self,max_symbols:Union[int,bool]=False):
       super(AbstractMaxpool2D,self).__init__()

    @staticmethod
    def abstract_maxpool2D(maxpool:nn.Module,
                           x:torch.tensor,
                           x_true:torch.tensor,add_symbol:bool=False,
                           device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
        maxpool = maxpool.to(device)
        kernel_size = maxpool.kernel_size

        #assert kernel_size[0]==kernel_size[1];"The kernel size must be a square"
        assert kernel_size==2,f"Maxpool2D kernel size {kernel_size}. A kernel size different of 2 is not supported"
        stride = maxpool.stride
        padding = maxpool.padding
        dim_x =len(x[0])
        x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
        x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
        
        x_final =maxpool(x_true)
        conv_0 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_1 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_2 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_3 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        w_0 = torch.tensor([[[[1., -1.], [0, 0.]]]])
        w_1 = torch.tensor([[[[0., 1.], [0, 0.]]]])
        w_2 = torch.tensor([[[[0., 0.], [0., 1.]]]])
        w_3 = torch.tensor([[[[0., 0.], [1., 0.]]]])
        w_0 = w_0.expand(dim_x,-1,-1,-1)
        w_1 = w_1.expand(dim_x,-1,-1,-1)
        w_2 = w_2.expand(dim_x,-1,-1,-1)
        w_3 = w_3.expand(dim_x,-1,-1,-1)
        conv_0.weight.data = w_0
        conv_0.bias.data =  torch.zeros(dim_x)
        conv_1.weight.data = w_1
        conv_1.bias.data =  torch.zeros(dim_x)
        conv_2.weight.data = w_2
        conv_2.bias.data =  torch.zeros(dim_x)
        conv_3.weight.data = w_3
        conv_3.bias.data =  torch.zeros(dim_x)

       


        x_result,x_min_result,x_max_result,x_true_result  = AbstractLinear.abstract_conv2D(conv_0,x,x_true,device=device)
        x_result,x_min_result,x_max_result,x_true_result = AbstractReLU.abstract_relu_conv2D(x_result,x_min_result,x_max_result,x_true_result,device=device)
        x_result_1,x_min_result_1,x_max_result_1,x_true_result_1  = AbstractLinear.abstract_conv2D(conv_1,x,x_true,device=device)
     
        x_result = AbstractBasic.abstract_addition(x_result_1,x_result)
   
        x_min_result += x_min_result_1
        x_max_result += x_max_result_1
        x_true_result += x_true_result_1
    
        x_result_1,x_min_result_1,x_max_result_1,x_true_result_1  = AbstractLinear.abstract_conv2D(conv_2,x,x_true,device=device)
        x_result_1 = AbstractBasic.abstract_substraction(x_result_1, x_result)
        x_min_result_1 -= x_min_result
        x_max_result_1 -= x_max_result
        x_true_result_1 -= x_true_result
        x_result_2,x_min_result_2,x_max_result_2,x_true_result_2  = AbstractReLU.abstract_relu_conv2D(x_result_1,x_min_result_1,x_max_result_1,x_true_result_1,add_symbol=False,device=device)
        x_result_2 = AbstractBasic.abstract_addition(x_result_2, x_result)
        x_min_result_2 += x_min_result
        x_max_result_2 += x_max_result
        x_true_result_2 += x_true_result


        x_result_1,x_min_result_1,x_max_result_1,x_true_result_1  = AbstractLinear.abstract_conv2D(conv_3,x,x_true,device=device)
        x_result_1 =AbstractBasic.abstract_substraction(x_result_1, x_result_2)
        x_min_result_1 -= x_min_result_2
        x_max_result_1 -= x_max_result_2
        x_true_result_1 -= x_true_result_2
        x_result_3,x_min_result_3,x_max_result_3,x_true_result_3  = AbstractReLU.abstract_relu_conv2D(x_result_1,x_min_result_1,x_max_result_1,x_true_result_1,device=device)
        x_result_3 = AbstractBasic.abstract_addition(x_result_3,x_result_2)
        x_min_result_3 += x_min_result_2
        x_max_result_3 += x_max_result_2
        x_true_result_3 += x_true_result_2
        x= x_result_3
        x_min = x_min_result_3
        x_max = x_max_result_3
        x_true = x_final

        if add_symbol:
            """
            new_symbols_indexes =torch.where(x[-1]!=0)


            for index,value in enumerate(new_symbols_indexes[0]):
            
            
                x =torch.cat((x,x[-1].unsqueeze(0)),dim=0)
                
                x[-2]=torch.zeros_like(x[-2])
                
                x[-2][value]=x[-1][value]
            x[-1]=torch.zeros_like(x[-1])
            """
         
            new_eps =torch.where(x[-1].flatten()!=0)[0].to(device)
           
            index = torch.arange(len(new_eps)).to(device)
            new_eps_batch_shape = x[-1].flatten().expand(len(new_eps)+1,-1).shape
            new_eps_batch = torch.zeros(new_eps_batch_shape).to(device)
            new_eps_batch[index,new_eps]=x[-1].flatten()[new_eps]
            new_eps_batch = new_eps_batch.reshape(x[-1].expand(len(new_eps)+1,-1,-1,-1).shape)
            

            x=x[:-1]

            x = torch.cat((x,new_eps_batch),dim=0) 
       
        
        return x,x_min,x_max,x_true







class AbstractBasic(nn.Module):
    def __init__(self):
        super(AbstractBasic,self).__init__()
    
    @staticmethod
    
    def abstract_addition(x:torch.tensor,
                          y:torch.tensor)->torch.tensor:
        """ This function abstracts the addition of two abstract tensors x and y
        the last layer (noise layer is added is the sum of the absolute values of the last layer of x and y
        input : x : tensor : the first tensor
                y : tensor : the second tensor
                output : z : tensor : the result of the addition
                """
        assert x.shape==y.shape; "The two tensors must have the same shape"
        z = x+y
        
        z[-1] = torch.abs(x[-1])+torch.abs(y[-1])
        return z
    @staticmethod

    def abstract_substraction(x:torch.tensor,
                              y:torch.tensor)->torch.tensor:
        """ This function abstracts the substraction of two abstract tensors x and y
        the last layer (noise layer is added is the sum of the absolute values of the last layer of x and y
        input : x : tensor : the first tensor
                y : tensor : the second tensor
                output : z : tensor : the result of the substraction
                """
        assert x.shape==y.shape; "The two tensors must have the same shape"
        z = x-y
        z[-1] = torch.abs(x[-1])+torch.abs(y[-1])
        return z
    


