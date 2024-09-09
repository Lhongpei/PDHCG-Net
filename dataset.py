import numpy as np
import pickle
import quadratic_programming as qpy
from datetime import datetime
import os
import gzip
from tqdm import tqdm
import random
import torch
class PDHCG_Dataset:
    def __init__(self, root, num_samples = None, save_flag = False, reprocess = False):
        self.root = root
        self.raw_root = os.path.join(root, 'raw')
        self.data = []
        self.num_samples = num_samples if num_samples is not None else len(os.listdir(self.raw_path))
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory {root} not found")
        pro_list = random.sample(os.listdir(self.raw_root), self.num_samples) if self.num_samples is not None else os.listdir(self.raw_root)
        if (not reprocess) and os.path.exists(os.path.join(self.processed_path, 'dataset.pth')):
            self.data = torch.load(os.path.join(self.processed_path, 'dataset.pth'))
        else:
            if reprocess and os.path.exists(os.path.join(self.processed_path, 'dataset.pth')):
                os.remove(os.path.join(self.processed_path, 'dataset.pth'))
            for file in tqdm(pro_list, desc="Loading dataset"):
                if file.endswith('.pkl.gz'):
                    with gzip.open(os.path.join(self.raw_root, file), 'rb') as f:
                        data_loaded = pickle.load(f)
                        data_loaded['problem'] = qpy.from_base_to_torch(data_loaded['problem'])
                        data_loaded['primal_solution'] = torch.tensor(data_loaded['primal_solution'], dtype=torch.float64)
                        data_loaded['dual_solution'] = torch.tensor(data_loaded['dual_solution'], dtype=torch.float64)
                        self.data.append(data_loaded)
        print(f"-----------------Loading END------------------")
        print(f"-----------------Dataset Info-----------------")
        print(f"Number of samples: {len(self.data)}")
        print(f"----------------------------------------------")
        if save_flag:
            
            if not os.path.exists(self.processed_path):
                os.makedirs(self.processed_path)
            # Save the dataset
            if not os.path.exists(os.path.join(self.processed_path, 'dataset.pth')):
                torch.save(self.data, os.path.join(self.processed_path, 'dataset.pth'))
            print(f"Dataset saved at {os.path.join(self.processed_path, 'dataset.pth')}")
    

    @property
    def processed_path(self):
        return os.path.join(self.root, 'processed')
     
    @property
    def raw_path(self):
        return os.path.join(self.root, 'raw')
     
    def len(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __get_slice__(self, s):
        return self.data[s]
    

class PDHCG_DataScheduler:
    def __init__(self, root):
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)
    
    def save_pair(self, problem: qpy.QuadraticProgrammingBase, primal, dual, time=None, iters=None, name=None, zipped=True):
        if name is None:
            name = f"scale_{problem.n}_{problem.m}_date_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl.gz"
        time_info = time if time is not None else 'not given'
        iters_info = iters if iters is not None else 'not given'
        if zipped:
            with gzip.open(os.path.join(self.root, name), 'wb') as f:
                pickle.dump(
                    {
                        'problem': problem,
                        'primal_solution': primal,
                        'dual_solution': dual,
                        'time_info': time_info,
                        'iters_info': iters_info
                    }, f
                )
        else:
            with open(os.path.join(self.root, name), 'wb') as f:
                pickle.dump(
                    {
                        'problem': problem,
                        'primal_solution': primal,
                        'dual_solution': dual,
                        'time_info': time_info,
                        'iters_info': iters_info
                    }, f
                )
        
    def loader(self):
        for file in os.listdir(self.root):
            if file.endswith('.pkl.gz'):
                with gzip.open(os.path.join(self.root, file), 'rb') as f:
                    yield pickle.load(f)
            elif file.endswith('.pkl'):
                with open(os.path.join(self.root, file), 'rb') as f:
                    yield pickle.load(f)
    
    def dataset(self, num_samples = None):
        return PDHCG_Dataset(self.root, num_samples)

if __name__ == '__main__':
    data_scheduler = PDHCG_DataScheduler("random_qp")
    loader = data_scheduler.loader()
    data = next(loader)
    print(data)
