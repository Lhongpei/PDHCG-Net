from juliacall import Main as jl
jl.seval('using CUDA')
from pdhcg import PDHCG
import numpy as np
import torch
from pympler import asizeof
from models import PDHCGNet
from omegaconf import OmegaConf
import os
from tqdm import tqdm
from dataset import PDHCG_DataScheduler
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system('unset LD_LIBRARY_PATH')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ.pop('LD_LIBRARY_PATH')
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
generate_path = "/data/hongpei/small_mpc/raw"
if not os.path.exists(generate_path):
    os.makedirs(generate_path)
data_scheduler = PDHCG_DataScheduler("/data/hongpei/small_mpc/raw")
for i in tqdm(range(5000)):
    scale = np.random.randint(63, 200)
    pdhcg = PDHCG(gpu_flag=True, warm_up_flag=True)
    pdhcg.setGeneratedProblem("mpc", scale, 1e-4, i + 10000)
    pdhcg.solve()
    data_scheduler.save_pair(pdhcg.toPyProblem(), np.array(pdhcg.primal_solution), 
                             np.array(pdhcg.dual_solution), pdhcg.solve_time_sec, pdhcg.iteration_count, zipped=False)
    


