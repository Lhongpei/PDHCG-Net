from juliacall import Main as jl
import torch
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn.functional as F
jl.seval('using SparseArrays')
def convert_scip_sparse_to_torch_sparse(
        scip_sparse,
        device: torch.device = torch.device("cpu")
        ) -> torch.sparse_coo_tensor:

    coo = scip_sparse
    torch_sparse = torch.sparse_coo_tensor(
        torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long),
        torch.tensor(coo.data, dtype=torch.float64),
        coo.shape,
        device=device
    )
    torch_sparse = torch_sparse.to_sparse_csr()
    return torch_sparse

def jl_to_scip(jl_matrix):
        row_indices, col_indices, values = jl.findnz(jl_matrix)
        return coo_matrix((np.array(values), (np.array(row_indices) - 1, np.array(col_indices) - 1)), shape = jl.size(jl_matrix))
    
def sparse_matrix_batch(matrix_list: list):
    return sparse_matrix_batch_cat(matrix_list)

def sparse_matrix_batch_csr(matrix_list: list):
    max_size = max([i.col_indices().shape[-1] for i in matrix_list])
    crow_batch = torch.stack([i.crow_indices() for i in matrix_list], dim=0)
    col_batch = torch.stack([F.pad(i.col_indices(), (0, max_size - len(i.col_indices())), 'constant', 0) for i in matrix_list], dim=0)
    values_batch = torch.stack([F.pad(i.values(), (0, max_size - i.values().size(-1)), 'constant', 0) for i in matrix_list], dim=0)
    return torch.sparse_csr_tensor(crow_batch, col_batch, values_batch)

def sparse_matrix_batch_coo(matrix_list: list):
    sizes = np.array([i.indices().shape[-1] for i in matrix_list])
    batch_idx = torch.repeat_interleave(torch.arange(len(matrix_list)), torch.from_numpy(sizes))
    indices_batch = torch.cat([i.indices() for i in matrix_list], dim=1)
    indices_batch = torch.cat([batch_idx.unsqueeze(0), indices_batch], dim=0)
    values_batch = torch.cat([i.values() for i in matrix_list], dim=0)
    return torch.sparse_coo_tensor(indices_batch, values_batch)

def sparse_matrix_batch_cat(matrix_list: list):
    col_sizes = np.array([i.shape[-1] for i in matrix_list])
    col_sizes = np.insert(col_sizes, 0, 0)
    col_sizes = np.cumsum(col_sizes)
    var_sizes = np.array([i.values().shape[-1] for i in matrix_list])
    var_sizes = np.insert(var_sizes, 0, 0)
    var_sizes = np.cumsum(var_sizes)
    crow_batch = torch.cat([matrix_list[i].crow_indices() + var_sizes[i]  if i == 0 else matrix_list[i].crow_indices()[1:] + var_sizes[i] for i in range(len(matrix_list))], dim=0)  
    col_batch = torch.cat([matrix_list[i].col_indices() + col_sizes[i] for i in range(len(matrix_list))], dim=0) 
    values_batch = torch.cat([i.values() for i in matrix_list], dim=0)
    return torch.sparse_csr_tensor(crow_batch, col_batch, values_batch, size=(len(crow_batch) - 1, col_sizes[-1]))

def batch_mm(matrix, matrix_batch):
    """
    see: https://github.com/pytorch/pytorch/issues/14489
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)