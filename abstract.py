import torch


class abstractTensor:
    def __init__(self, tensor: torch.Tensor, alpha: torch.Tensor ):
        self.tensor = tensor
        self.alpha = alpha
      


    def abstract_tensor(self):
        #assert len(self.alpha)==len(self.tensor.flatten()), "The length of alpha should be equal to the length of the flatten tensor"
       
        
        flatten_tensor  = self.tensor.flatten()
     
        abstract_tensor=[]
        abstract_tensor.append(self.tensor)
        for i in range(1,len(self.alpha)+1):
           
            abstract_tensor_layer = torch.zeros_like(flatten_tensor)
        
            abstract_tensor_layer[i-1]=self.alpha[i-1]
            abstract_tensor_layer = abstract_tensor_layer.reshape(self.tensor.shape)
            abstract_tensor.append(abstract_tensor_layer)
        
        abstract_tensor.append(torch.zeros_like(self.tensor))
        abstract_tensor= torch.stack(abstract_tensor)

        
        return abstract_tensor