import torch
import torch.nn as nn
import copy
from typing import List, Union, Tuple


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
        input = input.flatten().to(device)
        index = index.to(device)
        output=[]
        x = fully_connected_layer(input)
        output.append(x)
        for i in range(1,len(alpha)+1):
            weight_epsilon = torch.zeros_like(weight_of_layer)
            weight_epsilon[index]=alpha[i-1]
            fully_connected_layer[1].weight.data = weight_of_layer
            abstract_tensor_layer = fully_connected_layer(input)-fully_connected_layer(torch.zeros_like(input))     
            output.append(abstract_tensor_layer)
        output.append(torch.zeros_like(x))

        output_min = x[0] - torch.sum(torch.abs(x[1:]),dim=0)
        output_max = x[0] + torch.sum(torch.abs(x[1:]),dim=0)
        return output,output_min,output_max, x
       