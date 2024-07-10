import os
import pickle
import torch
from torch.utils.data import Dataset


def read_session(session_id,path):
    file_path = os.path.join(path,str(session_id)) 
    file_path = os.path.join(file_path,'all_reg.pickle') 
    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)
    return b 


step = 50
size_bin = int(50/step)

structures=  ['VISp', 'VISl', 'VISal']

with open("data/good_sessions.pickle", 'rb') as handle:
        good_sessions = pickle.load(handle)

class VBNDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, 
                 session_ids=good_sessions,
                 structures = structures,
                 file_path=None ,
                 change="change",
                 preprocess= False,
                 time_bin = 0,
                 dim = 50):
        
        full_data = {
            area: torch.Tensor() for area in structures
        }
 
            
        self.session_ind =[0]
                           
        for session_id in session_ids:
            data = read_session(session_id,file_path)
            for i,area in enumerate(structures):
                #if aggregate:
                #size_bin = int( data[area][change].shape[-1]//5 )
                to_add =data[area][change].mean(axis=1)
                to_add = to_add [:,time_bin*dim:(time_bin+1)*dim ]
                if i ==0:
                    self.session_ind.append( self.session_ind[-1] + to_add.shape[0] )
                    
                full_data[area] = torch.cat([full_data[area],torch.Tensor(to_add) ],axis=0)
        
        if preprocess:
            for area in structures:
                full_data[area] = ( full_data[area] - full_data[area].mean(axis=0) )/ full_data[area].std(axis=0) 
            
        self.data = full_data     
        self.areas =   structures 
        
        
    def get_sessions(self):
        out = []
        for idx , i in enumerate(self.session_ind[:-1]) :
            sess ={}
            for area in self.areas:
                sess[area] = self.data[area][self.session_ind[idx] :self.session_ind[idx+1],:]
            out.append(sess)
        return out
        
         
    def __len__(self):
        return self.data[self.areas[0]].shape[0] 

    def __getitem__(self, idx):
        
        return { area : self.data[area][idx]
                for area in self.areas} 

