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
from torch.optim.lr_scheduler import OneCycleLR
from dataset import PDHCG_DataScheduler
from sklearn.model_selection import train_test_split
import math
from validate import validate
from quadratic_programming import collate_fn
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system('unset LD_LIBRARY_PATH')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ.pop('LD_LIBRARY_PATH')
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
set_seed(0)
print("Using device: ", device)
pdhcg = PDHCG(gpu_flag=True)
config = OmegaConf.load("config.yaml")
config = OmegaConf.to_container(config, resolve=True)
config['model_params']['device'] = device
config['validate_params']['device'] = device
model = PDHCGNet(config['model_params'])
#model.load_state_dict(torch.load(os.path.join('saved_models', os.listdir('saved_models')[0]), map_location=device))
state_dict = torch.load(os.path.join('saved_models', os.listdir('saved_models')[0]), map_location=device)
sub_state_dict = {k: v for k, v in state_dict.items() if 'sub' in k}
model.load_state_dict(sub_state_dict, strict=False)
datascheduler = PDHCG_DataScheduler("/data/hongpei/small_mpc")
train_params = config['train_params']
epochs = train_params['epochs']
validate_params = config['validate_params']
batchsize = 128#train_params['batchsize']
update_size = 20
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), train_params['lr'], betas=(0.9, 0.999)) 
#scheduler = OneCycleLR(optimizer, max_lr=train_params['max_lr'], total_steps=epochs)
dataset = datascheduler.dataset(4000).data
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
shuffled_idxes = np.random.permutation(len(train_dataset))
best_loss = 1e9
for ep in tqdm(range(epochs), desc="Training"):
    loss_epoch = 0
    run_num = 0
    times_complete_batch = math.floor(len(shuffled_idxes) / batchsize)
    for i in range(times_complete_batch + 1):
        if i < times_complete_batch:
            start_idx = i * batchsize
            end_idx = (i + 1) * batchsize
        else:
            start_idx = i * batchsize
            end_idx = len(shuffled_idxes) + 1
        if start_idx >= end_idx - 1:
            continue
        instance = collate_fn((dataset[start_idx:end_idx]))
        #instance = dataset[start_idx]
        problem = instance['problem'].to(device)
        primal_label = instance['primal_solution'].to(device)
        dual_label = instance['dual_solution'].to(device)
        primal, dual = model(problem)
        loss_per_epoch = loss(primal, primal_label) + loss(dual, dual_label)
        run_num += end_idx - start_idx
        # if run_num % update_size == 0:
        #     optimizer.zero_grad()
        #     loss_per_epoch.backward()
        #     optimizer.step()
        #     loss_epoch += loss_per_epoch.item()
        #     loss_per_epoch = 0
        # else:
        #     loss_per_epoch.backward()
        #     loss_epoch += loss_per_epoch.item()
        optimizer.zero_grad()
        loss_per_epoch.backward()
        optimizer.step()
        loss_epoch += loss_per_epoch.item()
        loss_per_epoch = 0
    loss_epoch /= len(shuffled_idxes)
    # scheduler.step()
    # print(f'LR: {scheduler.get_last_lr()}')
    print(f"Loss: {loss_epoch}")
    if ep % 10 == 0:
        loss_validate = validate(model, test_dataset, validate_params)
        if loss_validate < best_loss and loss_validate < 1:
            best_loss = loss_validate
            torch.save(model.state_dict(), f"best_model_{best_loss}.pth")
    model.train()
            

        



        