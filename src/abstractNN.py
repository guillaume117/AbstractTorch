import torch
import torch.nn as nn
import copy
from typing import List, Union, Tuple
"""
This class aims to make the tools that will be used to generate an abstract neural network
In that abstract Neural Network, weight and bias will be affine function. 
The objectif is to experiment a new way of learning 
--> 
"""


class AbstractWeight(nn.Module):
    def __init__(self):
        super(AbstractWeight, self).__init__()
     
    """This method aims to generate the output of an abstract layer. For each weights that are in the index, 
    an affine expression is generated and passed to the nexte layer (output). 
    This method is used inside a forward pass, before that layer, it is a concrete evaluation,
    after it becomes an abstract evaluation
    The choice of the indexes to be abstracted can be done by the user, or by the use of pruning algorithms.
    ."""
    @staticmethod
    def generate_ztp_from_layer_and_indexes(fully_connected_layer: nn.Module, 
                     index : torch.Tensor,
                     alpha : torch.Tensor, 
                     input : torch.Tensor,
                     device = torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        weight_of_layer = fully_connected_layer[1].weight.data
        bias_of_layer = fully_connected_layer[1].bias.data
        intermediate_layer = copy.deepcopy(fully_connected_layer).to(device)
        intermediate_layer[1].bias.data = torch.zeros_like(intermediate_layer[1].bias.data )
       
        index = index.to(device)
        output=[]
        print(f"input.shape: {input.shape}")
        x = fully_connected_layer(input)
        print(f"x.shape: {x.shape}")
        output.append(x)
        for i in range(1,len(alpha)+1):
            weight_epsilon = torch.zeros_like(weight_of_layer).flatten()
            weight_epsilon[index[i-1]]=alpha[i-1]
            weight_epsilon = weight_epsilon.reshape(weight_of_layer.shape)
            intermediate_layer[1].weight.data = weight_epsilon
            abstract_tensor_layer = intermediate_layer(input)

            output.append(abstract_tensor_layer)
        output.append(torch.zeros_like(x))
        output = torch.stack(output).squeeze(1)

        output_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
        output_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
        print(f"output.shape: {output.shape}")
        return output,output_min,output_max, x



class AbstractNN(nn.Module):
    def __init__(self):
        super(AbstractNN,self).__init__()
    
    @staticmethod
    def abstract_linear(lin:nn.Module,
                            x:torch.tensor,
                            x_true:torch.tensor,
                            lin_weights_alphas_index_and_values: Tuple[torch.tensor],
                            lin_bias_alphas_index_and_values: Tuple[torch.tensor],
                            device:torch.device=torch.device("cpu")):
            
        with torch.no_grad():

            x_center = x[0]
           

            
            
            
            
            
            lin=lin.to(device)
            x =x.unsqueeze(1).to(device)
            x_true=x_true.to(device)
            lin_weight_shape = lin[1].weight.data.shape
            lin_copy_bias_null = copy.deepcopy(lin).to(device)
            lin_copy_bias_null[1].bias.data = torch.zeros_like(lin[1].bias.data)
            lin_abs_bias_null = copy.deepcopy(lin).to(device)
            lin_abs_bias_null[1].weight.data =torch.abs(lin[1].weight.data)
            lin_abs_bias_null[1].bias.data = torch.zeros_like(lin[1].bias.data)
            lin_bias_shape =lin[1].bias.data.shape
             
            x_true = lin(x_true)
            x_value=lin(x[0])
            x_epsilon=lin_copy_bias_null(x[1:-1])
            x_noise=lin_abs_bias_null(x[-1])

            del x
            x = torch.cat((x_value,x_epsilon,x_noise),dim=0)

            index = lin_weights_alphas_index_and_values.indices.to(device)
            if len(index)>0:
         
                values =lin_weights_alphas_index_and_values.values

                add_eps =[]
                for indice in range(len(index)):
                    weight_epsilon= torch.zeros(lin_weight_shape).flatten()
                    weight_epsilon[index[indice]] =values[indice]
                    weight_epsilon = weight_epsilon.reshape(lin_weight_shape)
                    lin_copy_bias_null[1].weight.data = weight_epsilon
                    abstract_tensor_layer = lin_copy_bias_null(x_center.unsqueeze(0))   

                    add_eps.append(abstract_tensor_layer)
            
                add_eps = torch.stack(add_eps).squeeze(1)
                print(add_eps.shape)
                x=torch.cat((x[:-1],add_eps,x[-1].unsqueeze(0)))
            
            

            index = lin_bias_alphas_index_and_values.indices.to(device)
            if len(index)>0:
                values = lin_bias_alphas_index_and_values.values.to(device)
                add_eps = []
                for indice in range(len(index)):
                    lin_copy_bias_null[1].weight.data = torch.zeros(lin_weight_shape)
                    bias_epsilon = torch.zeros(lin_bias_shape)
                    bias_epsilon[index[indice]]= values[indice]
                    lin_copy_bias_null[1].bias.data =bias_epsilon
                    abstract_tensor_layer = lin_copy_bias_null(torch.zeros_like(x_center).unsqueeze(0))
                    add_eps.append(abstract_tensor_layer)
                add_eps = torch.stack(add_eps).squeeze(1)
                print(add_eps.shape)
                x=torch.cat((x[:-1],add_eps,x[-1].unsqueeze(0)))
                                                               
                                                    

     
        
           
            del x_value,x_epsilon,x_noise,lin_abs_bias_null,lin_copy_bias_null,
    
            
            x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
            x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
          
            return x,x_min,x_max,x_true
    @staticmethod
    def abstract_conv2D(conv:nn.Module,
                        x:torch.tensor,
                        x_true:torch.tensor,
                        lin_weights_alphas_index_and_values,
                        lin_bias_alphas_index_and_values,
                        device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
        conv = conv.to(device)
        x=x.to(device)
        x_true = x_true.to(device)
        x_center = x[0]
        with torch.no_grad():

            conv_abs_bias_null = copy.deepcopy(conv).to(device)
            conv_copy_bias_null = copy.deepcopy(conv).to(device)
            conv_copy_bias_null.bias.data = torch.zeros_like(conv.bias.data)
            conv_abs_bias_null.weight.data = torch.abs(conv.weight.data)
            conv_abs_bias_null.bias.data = torch.zeros_like(conv.bias.data)
            conv_weight_shape = conv.weight.data.shape
            conv_bias_shape = conv.bias.data.shape

        
        
            x_true = conv(x_true)
            x_value=conv(x[0]).unsqueeze(0)
            x_epsilon=conv_copy_bias_null(x[1:-1])
    
            x_noise=conv_abs_bias_null(x[-1]).unsqueeze(0)
            del x
            x = torch.cat((x_value,x_epsilon,x_noise),dim=0)

            index = lin_weights_alphas_index_and_values.indices.to(device)
            if len(index)>0:
         
                values =lin_weights_alphas_index_and_values.values

                add_eps =[]
                for indice in range(len(index)):
                    weight_epsilon= torch.zeros(conv_weight_shape).flatten()
                    weight_epsilon[index[indice]] =values[indice]
                    weight_epsilon = weight_epsilon.reshape(conv_weight_shape)
                    conv_copy_bias_null.weight.data = weight_epsilon
                    abstract_tensor_layer = conv_copy_bias_null(x_center.unsqueeze(0))   

                    add_eps.append(abstract_tensor_layer)
            
                add_eps = torch.stack(add_eps).squeeze(1)
                print(add_eps.shape)
                x=torch.cat((x[:-1],add_eps,x[-1].unsqueeze(0)))
            
            

            index = lin_bias_alphas_index_and_values.indices.to(device)
            if len(index)>0:
                values = lin_bias_alphas_index_and_values.values.to(device)
                add_eps = []
                for indice in range(len(index)):
                    conv_copy_bias_null.weight.data = torch.zeros(conv_weight_shape)
                    bias_epsilon = torch.zeros(conv_bias_shape)
                    bias_epsilon[index[indice]]= values[indice]
                    conv_copy_bias_null.bias.data =bias_epsilon
                    abstract_tensor_layer = conv_copy_bias_null(torch.zeros_like(x_center).unsqueeze(0))
                    add_eps.append(abstract_tensor_layer)
                add_eps = torch.stack(add_eps).squeeze(1)
                print(add_eps.shape)
                x=torch.cat((x[:-1],add_eps,x[-1].unsqueeze(0)))





            del x_value,x_epsilon,x_noise,conv_abs_bias_null, conv_copy_bias_null
            x_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
            x_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
            return x,x_min,x_max,x_true 