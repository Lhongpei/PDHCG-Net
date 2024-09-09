from juliacall import Main as jl
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
jl.seval('using CUDA')
import torch
from pdhcg import PDHCG
import numpy as np
import torch
from pympler import asizeof
from models import PDHCGNet
from omegaconf import OmegaConf
from quadratic_programming import from_base_to_torch

import time
from tqdm import tqdm
import pandas as pd


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
pdhcg = PDHCG(gpu_flag=False)
config = OmegaConf.load("config.yaml")
config = OmegaConf.to_container(config, resolve=True)
config['model_params']['device'] = device
model = PDHCGNet(config['model_params'])
model.load_state_dict(torch.load(os.path.join('saved_models', os.listdir('saved_models')[0]), map_location=device))
model.to(device)


df = pd.DataFrame(columns=['Original Solving Time', 'Initial Given Solving Time', 'Original Iterations', 'Initial Given Iterations'])
average_time = 0
#pdhcg.ruiz_rescaling_(10)
for i in range(100):
    pdhcg.setGeneratedProblem("randomqp", 100000, 1e-4, i+1000)

    initial_primal = np.zeros(100000)
    initial_dual = np.zeros(50000)
    pdhcg.solve(warm_start_flag=False, warm_start_primal=initial_primal, warm_start_dual=initial_dual)

    original_time = pdhcg.solve_time_sec
    original_iter = pdhcg.iteration_count

    start_time = time.time()
    qp = pdhcg.toPyProblem()
    qp = from_base_to_torch(qp, device)
    primal, dual = model(qp)
    used_time = time.time() - start_time
    average_time += used_time
    pdhcg.solve(warm_start_flag=True, warm_start_primal=primal.detach().cpu().numpy(), warm_start_dual=dual.detach().cpu().numpy())
    initial_given_time = pdhcg.solve_time_sec
    initial_given_iter = pdhcg.iteration_count
    df = pd.concat([df, pd.DataFrame([[original_time, initial_given_time, original_iter, initial_given_iter]], columns=['Original Solving Time', 'Initial Given Solving Time', 'Original Iterations', 'Initial Given Iterations'])], ignore_index=True)
    df.to_csv('result.csv')
print(average_time/100)