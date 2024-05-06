import torch
import torch.nn as nn 
import copy
from typing import List, Union, Tuple
import gc



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
            lin_copy_bias_null = copy.deepcopy(lin).to(device)
            lin_copy_bias_null[1].bias.data = torch.zeros_like(lin[1].bias.data)
            lin_abs_bias_null = copy.deepcopy(lin).to(device)
            lin_abs_bias_null[1].weight.data =torch.abs(lin[1].weight.data)
            lin_abs_bias_null[1].bias.data = torch.zeros_like(lin[1].bias.data)
        
            
            x_value = x[0].unsqueeze(1)
            x_epsilon= x[1:-1].unsqueeze(1)
            x_noise =x[-1].unsqueeze(1)
        
            x=lin(x)    
            x_true = lin(x_true)
            x[0]=lin(x_value)
            x[1:-1]=lin_copy_bias_null(x_epsilon)
            x[-1]=lin_abs_bias_null(x_noise)
            x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
            x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
            del x_value,x_epsilon,x_noise
            gc.collect()
          
            return x,x_min,x_max,x_true
        

        @staticmethod
        def abstract_conv2D(conv:nn.Module,
                            x:torch.tensor,
                            x_true:torch.tensor,
                            device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
            conv = conv.to(device)
            x=x.to(device)
            x_true = x_true.to(device)
          
        
            conv_abs_bias_null = copy.deepcopy(conv).to(device)
            conv_copy_bias_null = copy.deepcopy(conv).to(device)
            conv_copy_bias_null.bias.data = torch.zeros_like(conv.bias.data)
            conv_abs_bias_null.weight.data = torch.abs(conv.weight.data)
            conv_abs_bias_null.bias.data = torch.zeros_like(conv.bias.data)
        
        
            x_value = x[0]
            x_epsilon= x[1:-1]
            x_noise = x[-1]
            x=conv(x)
            x[0]=conv(x_value)
            x_true = conv(x_true)
            x[1:-1]=conv_copy_bias_null(x_epsilon)
            x[-1]=conv_abs_bias_null(x_noise)
            x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
            x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
            
            del x_value,x_epsilon,x_noise
            gc.collect()

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

        coef_approx_linear = x_max/(torch.abs(x_max)+torch.abs(x_min))
        coef_approx_linear = torch.where(torch.isnan(coef_approx_linear),torch.zeros_like(coef_approx_linear),coef_approx_linear)
        bias_approx_linear = x_max*(1-coef_approx_linear)/2
        noise_approx_linear = torch.abs(bias_approx_linear)
        x_true  = nn.ReLU()(x_true)
       

        
        mask_p = (sgn==0)
        mask_1 =(sgn==2) + (sgn==1)
        mask_0 = (sgn==-2)+(sgn==-1)
        x[0,mask_p]=(coef_approx_linear[mask_p]*x[0,mask_p]+bias_approx_linear[mask_p])
        x[0,mask_1]=x[0,mask_1]
        x[0,mask_0]=0

        x[1:-1,mask_p]=coef_approx_linear[mask_p]*x[1:-1,mask_p]
        x[1:-1,mask_1]=x[1:-1,mask_1]
        x[1:-1,mask_0]=0
        x[-1,mask_p]=noise_approx_linear[mask_p]+torch.abs(coef_approx_linear[mask_p])*x[-1,mask_p]
        x[-1,mask_1]=x[-1,mask_1]
        x[-1,mask_0]=0
        del sgn_min,sgn_max,sgn,coef_approx_linear,bias_approx_linear,noise_approx_linear, mask_0,mask_1,mask_p
        gc.collect()
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
        
        coef_approx_linear = x_max/(torch.abs(x_max)+torch.abs(x_min))
        coef_approx_linear = torch.where(torch.isnan(coef_approx_linear),torch.zeros_like(coef_approx_linear),coef_approx_linear)
        bias_approx_linear = x_max*(1-coef_approx_linear)/2
        noise_approx_linear = torch.abs(bias_approx_linear)

        x_true  = nn.ReLU()(x_true)
    
      
        
        mask_p = (sgn==0)
        mask_1 =(sgn==2) + (sgn==1)
        mask_0 = (sgn==-2)+(sgn==-1)
        x[0,mask_p]=(coef_approx_linear[mask_p]*x[0,mask_p]+bias_approx_linear[mask_p])
        x[0,mask_1]=x[0,mask_1]
        x[0,mask_0]=0

        x[1:-1,mask_p]=coef_approx_linear[mask_p]*x[1:-1,mask_p]
        x[1:-1,mask_1]=x[1:-1,mask_1]
        x[1:-1,mask_0]=0
        x[-1,mask_p]=noise_approx_linear[mask_p]+torch.abs(coef_approx_linear[mask_p])*x[-1,mask_p]
        x[-1,mask_1]=x[-1,mask_1]
        x[-1,mask_0]=0

        del sgn_min,sgn_max,sgn,coef_approx_linear,bias_approx_linear,noise_approx_linear, mask_0,mask_1,mask_p
        gc.collect()



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

       #max(a,b,c,d) = relu(relu(relu(a-b)+b)+c)+d)


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
        """Since you cannot add an abstract tensor like a vulgare tensor"""
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
    


