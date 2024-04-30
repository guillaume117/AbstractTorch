import torch
from torch.utils.data import DataLoader
import time
import copy
from torch.nn.functional import one_hot





class CustomTrainer:
    """
    This class is used to train the model.  print(f"labels.size ={labels.size()}")
    It is used to train the model on the train_dataset and validate it on the val_dataset.
    input : model : tensor : the model to train
            criterion : tensor : the criterion to use
            optimizer : tensor : the optimizer to use
            scheduler : tensor : the scheduler to use
            num_epochs : int : the number of epochs to train the model
            learning_rate : float : the learning rate to use
            batch_size : int : the batch size to use
            device : str : the device to use
            nan_mask_label : tensor : the mask to use
            running_instance : str : the name of the running instance
    output : model : tensor : the trained model
    """
    def __init__(self,model,device):
        self.device = device
        self.model = model
        

    
    



    def train_model(self,train_dataset,val_dataset,criterion,
                 optimizer,scheduler,num_epochs,learning_rate,batch_size,verbose=True,resname='model'):
        """
        This function is used to train the model.
        input : train_dataset : tensor : the train dataset
                val_dataset : tensor : the validation dataset
        output : model : tensor : the trained model
        """
        self.criterion=criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.learning_rate =learning_rate
      
        val_acc = []
        val_loss = []
        train_acc = []
        train_loss = []
        epoch=0

        train_loader  = DataLoader(dataset = train_dataset, batch_size=self.batch_size, shuffle =True,drop_last=True)
        val_loader = DataLoader(dataset = val_dataset, batch_size=self.batch_size,shuffle=True, drop_last=True)

        dataloaders = {'train': train_loader, 'val': val_loader}
        start = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0
        list = {'train': {'acc': train_acc, 'loss': train_loss}, 
            'val':{'acc': val_acc, 'loss': val_loss}}
       

        
        for epoch in range(self.num_epochs):

            print('-' * 100)
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 100)

            for phase in ['train','val']:
                if phase =='train':
                    self.model.train()
                else:
                    self.model.eval()
                running_corrects = 0.0
                batch_number = 0
                batch_acc_single_element = 0
                for inputs, labels in dataloaders[phase]:
            
                    #print(inputs.size())
                    #print(f"labels from dataset={labels}")
                    batch_number+=1
                   
                    #print(f"label after one hot encoding  = {label}")

                    if self.device.type == 'mps':
                       

                        inputs = inputs.to(self.device).float()
                        label = labels.to(self.device).float()
                    else :
                        #if the device is cuda or cpu, we need to convert the inputs and labels to the device

                        inputs = inputs.to(self.device).float()
                        label = labels.to(self.device).float()
                        

                    self.optimizer.zero_grad()
                    #forward    with torch.no_grad() if phase == 'val' else torch.set_grad_enabled(phase == 'train'):
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
          
                     #   print(f'outputs.size() = {outputs.size()}')
                      #  print(f'outputs={outputs}')
                        _,preds = torch.max(outputs, 1)
                        
                      

                       # print(f"preds{preds}")
                    
                        loss = self.criterion(outputs,label)
                
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()



                        if (batch_number%100==0 )|(phase == 'val')|(batch_number ==1):
                            batch_acc_single_element +=1

                          
            
                            batch_accuracy =torch.sum(torch.where((label==preds),torch.tensor(1).to(self.device),torch.tensor(0).to(self.device)))
                            #print(batch_accuracy)
                           
                            accuracy = batch_accuracy/self.batch_size*100
                            if verbose:
                                print(f"batch_loss {batch_number}_Epoch_{epoch}={loss.item():.2f}, accuracy = {accuracy:.2f}%")
                            running_corrects +=batch_accuracy
                epoch_acc =100*running_corrects/(batch_acc_single_element*self.batch_size)
                print(f"EPOCH ACCURACY = {epoch_acc:.2f} %")
                        
                            
                
                
                list[phase]['loss'].append(loss)
                list[phase]['acc'].append(epoch_acc)
                print("*"*100)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss.item(), epoch_acc))
                print("*"*100)
        
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    
            
            print()
            
        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), f'dataset/{resname}_{epoch}_valacc_{epoch_acc}.pth')
        
            
        return self.model
    


    def evaluate_model(self,test_dataset,verbose=True):
        """
        This function is used to evaluate the model.
        input : test_dataset : tensor : the test dataset
        output : model accuarcy : float : the model accuracy over the test dataset
        """
        self.model.eval()
        running_corrects = 0.0
        batch_number = 0
        batch_acc_single_element = 0
        test_loader = DataLoader(dataset = test_dataset, batch_size=len(test_dataset),shuffle=True, drop_last=True)
        for inputs, labels in test_loader:
        
            label =one_hot(labels,10)
            if self.device.type == 'mps':
                inputs = inputs.to(self.device).float()
                label = label.to(self.device).float()
            else :
                inputs = inputs.to(self.device).float()
                label = label.to(self.device).float()
            with torch.no_grad():
                outputs = self.model(inputs)
                _,preds = torch.max(outputs, 1)
                batch_accuracy =torch.sum(torch.where((labels==preds),torch.tensor(1).to(self.device),torch.tensor(0).to(self.device)))
           
                running_corrects +=batch_accuracy
        epoch_acc =100*running_corrects/(len(test_dataset))
        if verbose:
            print(f"EPOCH ACCURACY = {epoch_acc:.2f} %")
        return epoch_acc
