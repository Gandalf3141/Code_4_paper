import torch
from torch.utils.data import Dataset

#this return (u1,u2,s,v), (s,v)
class custom_simple_dataset(Dataset):
 
    def __init__(self, data, window_size):
 
        self.data = data
        self.ws = window_size
 
    def __len__(self):
        return self.data.size(0)
 
    def __getitem__(self, idx):
 
        inp = self.data[idx, :, :]
        label = self.data[idx, self.ws:, 2:]

        return inp, label

# no change needed since other system was (p,s,v) with label (s,v)
# LSTM and TCN without OR
class CustomDataset(Dataset):


    def __init__(self, data, window_size):

        self.data = data
        self.ws = window_size

    def __len__(self):
        return self.data.size(0)*self.data.size(1) - (self.ws + 1)

    def __getitem__(self, idx):

        j = int(idx/self.data.size(1))  

        k = int((idx + self.ws) / self.data.size(1))

        m = (idx + self.ws) - k * self.data.size(1)

        index = idx % self.data.size(1)

        if j < k :
            
            if m < 0: 
                inp = self.data[j, index : index + self.ws, :]
            else: 
                inp=torch.cat((self.data[j, index : self.data.size(1) , :],
                          self.data[j, self.data.size(1) - 1, :].repeat(m, 1)))
                
            label = self.data[j, self.data.size(1) - 1, :]
                
        else:

            inp = self.data[j, index : index + self.ws, :]

            label = self.data[j, index + self.ws, :]

        return inp, label

    def get_all_data(self):
        return self.data
    
# We need this for MLPs without OR
class CustomDataset_mlp(Dataset):


    def __init__(self, data, window_size):

        self.data = data
        self.ws = window_size
        

    def __len__(self):
        return self.data.size(0)*self.data.size(1) - (self.ws + 1)

    def __getitem__(self, idx):

        j = int(idx/self.data.size(1))  

        k = int((idx + self.ws) / self.data.size(1))

        m = (idx + self.ws) - k * self.data.size(1)

        index = idx % self.data.size(1)

        if j < k :
            
            if m < 0: 
                inp = self.data[j, index : index + self.ws, :]
            else: 
                inp=torch.cat((self.data[j, index : self.data.size(1) , :],
                          self.data[j, self.data.size(1) - 1, :].repeat(m, 1)))
                

            label = self.data[j, self.data.size(1) - 1, :]
                
        else:

            inp = self.data[j, index : index + self.ws, :]


            label = self.data[j, index + self.ws, :]

        last = inp[-1:,:]

        inp = torch.cat((inp[:,0], inp[:,1], inp[:,2], inp[:,3]))
        
        return inp, last, label