from juliacall import Main as jl
jl.seval('using CUDA')
from pdhcg import PDHCG
import numpy as np
import os
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
jl.seval('push!(LOAD_PATH, "src/.")')
jl.seval('using PDHCG')

# pdhcg = PDHCG(gpu_flag=True)
# pdhcg.setGeneratedProblem("randomqp", 1000, 1e-2, 0)
# print("generated problem")


# pdhcg.solve(warm_start_flag=False)
# print(pdhcg.result)

qp = jl.PDHCG.generateProblem("randomqp", n = 6, density = 1., seed = 0)
jl.PDHCG.pdhcgSolve(qp, gpu_flag = True)
# 