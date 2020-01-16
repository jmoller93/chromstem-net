#!/env/python

"""
This is the dataset loader for all chromstem data.

Args:
    csv_file (string) : The master csv file where the data and labels are located
    root_dir (string) : Right now this is just the current directory, but I plan to split the data between training and validation
    transform (optional, callable) : Transforms the dataset, but probably unnecessary with this dataset, may be removed

"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ChromstemDataset(Dataset):
    """Chromstem Dataset"""

    def __init__(self,csv_file,root_dir,transform=None):
        """
        Args:
            csv_file (string): Master CSV file that will hold all the data
            root_dir (string): Directory where the datasets are labeled
            transform (callable, optional): Optional transformations to samples
        """

        f = open(csv_file,'r')
        csv_length = len(f.readlines()[-1].split(','))
        f.close()
        col_names = ['File'] + ['Nucl%d' % i for i in range(csv_length-1)]
        self.nucl_coords_frame_ = pd.read_csv(csv_file,engine='python',header=None,names=col_names)
        self.root_dir_ = root_dir
        self.transform_ = transform

    # Reads the '.dat' data file
    def read_chromstem(self,fnme):
        x,y,z,rho = np.loadtxt(fnme,comments='#',unpack=True)
        tensor = torch.zeros_like(torch.empty(1,int(x[0]),int(y[0]),int(z[0])))

        for i,arr in enumerate(zip(x[1:],y[1:],z[1:])):
            tensor[0,int(arr[0]),int(arr[1]),int(arr[2])] = rho[i]

        # Convert to sparse matrix as data will be sparse
        #tensor = torch.Sparse.FloatTensor(tensor)
        return tensor

    def __len__(self):
        return(len(self.nucl_coords_frame_))

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        chromstem_name = os.path.join(self.root_dir_,self.nucl_coords_frame_.iloc[idx,0])
        chromstem = self.read_chromstem(chromstem_name)
        nucl_coords = self.nucl_coords_frame_.iloc[idx,1:]
        nucl_coords = np.asarray([nucl_coords])
        nucl_coords = nucl_coords.astype('float').reshape(-1,3)
        num_nucls = int(chromstem_name.split('/data/')[1].split('/')[0])

        sample = {'chromstem' : chromstem, 'nucl_coords' : nucl_coords, 'num_nucls' : num_nucls}

        if self.transform_:
            sample = self.transform_(sample)

        return sample

