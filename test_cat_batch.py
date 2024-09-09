import torch 
from utils import sparse_matrix_batch_cat
t = torch.tensor([[1, 0], [4, 0], [7, 8]])
p = torch.tensor([[0, 2], [0, 5], [0, 0]])
t = t.to_sparse_csr()
p = p.to_sparse_csr()
csr = sparse_matrix_batch_cat([t, p,t])
csr.to_dense()
