def make_indice_and_values_tupple():
    tupple = torch.tensor([])
    tupple.indices = torch.tensor([])
    tupple.values = torch.tensor([])
    return tupple


import matplotlib.pyplot as plt
import numpy as np
def plot_dominance(result,x_min,x_max,x_true):
       y_min       =  np.array(x_min)
       y_max       =  np.array(x_max)
       center_Ztp  =  np.expand_dims(np.array(result[0]),axis =1)
       y_true      =  np.expand_dims(np.array(x_true[:])[0],axis =1)



       




       x_scale = np.arange(len(y_min))
       D =np.stack((y_min,y_max),axis=1)

   
       # plot:

       fig,ax = plt.subplots(1, 1, figsize=(8,4), tight_layout=True)
       ax.eventplot(D, orientation="vertical", linewidth=1,color='blue',linelengths=0.3)
       ax.eventplot(y_true, orientation="vertical", linewidth=0.50,color='green',linelengths=0.4)
       ax.eventplot(center_Ztp, orientation="vertical", linewidth=1,color='red',linelengths=0.5)

       ax.set(xlim=(-0.5, 10),xticks=x_scale,xticklabels=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag,","Ankle boot"],
              ylim=(min(-1,np.min(D)-1), max(1,p.max(D)+1)))
       plt.ylabel("Value of the abstract domain")
       plt.title("Dominance interval for the 10 classes of Fashion MNIST .\n Fully abstracted network")
       plt.legend()
       plt.show()
