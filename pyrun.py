from pdhcg import PDHCG
import numpy as np
import torch
from pympler import asizeof
from models import PDHCGNet
from omegaconf import OmegaConf
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.system('unset LD_LIBRARY_PATH')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ.pop('LD_LIBRARY_PATH')
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
pdhcg = PDHCG(gpu_flag=True)
config = OmegaConf.load("config.yaml")
config = OmegaConf.to_container(config, resolve=True)
config['model_params']['device'] = device
model = PDHCGNet(config['model_params'])
pdhcg.setGeneratedProblem("randomqp", 1000, 1e-2, 0)
print("generated problem")
# initial_primal = np.zeros(1000)
# initial_dual = np.zeros(500)
# pdhcg.ruiz_rescaling_(10)
# # # pdhcg.solve(warm_start_flag=True, warm_start_primal=initial_primal, warm_start_dual=initial_dual)
# # # print("Objective value: ", pdhcg.objective_value)
# # # print("Primal solution Norm: ", np.linalg.norm(pdhcg.primal_solution))
# # # print("Dual solution Norm: ", np.linalg.norm(pdhcg.dual_solution))
# # # print("Iteration count: ", pdhcg.iteration_count)
# # # print("Solve time: ", pdhcg.solve_time_sec)
# optim = torch.optim.Adam(model.parameters(), lr=1e-4)
# qp = pdhcg.toPyProblem()
# print('Inital_py')
# qp.toTorch_(device=device)
# #print("Initial KKT: ", qp.kkt(torch.zeros(1000, device=device, dtype=torch.float64), torch.zeros(500, device=device, dtype=torch.float64)).item())
# print(asizeof.asizeof(qp))
# for _ in tqdm(range(10000)):
#     primal, dual = model(qp)
#     kkt = qp.kkt(primal, dual)

#     model.zero_grad()
#     kkt.backward()

            
#     optim.step()
#     print("KKT: ", kkt.item())
    
# pdhcg.solve(warm_start_flag=True, warm_start_primal=primal.detach().cpu().numpy(), warm_start_dual=dual.detach().cpu().numpy())

pdhcg.solve(warm_start_flag=False)