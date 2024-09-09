import torch
from utils import convert_scip_sparse_to_torch_sparse
from scipy.sparse import coo_matrix
from utils import sparse_matrix_batch
import numpy as np
"""
Notice that QuadraticProgrammingBase is a class that holds the data of a quadratic programming problem.
while QuadraticProgrammingTorch is a class that holds the data of a quadratic programming problem in torch tensors.
Don't modify the QuadraticProgrammingBase class. You can modify the QuadraticProgrammingTorch class.

"""
class QuadraticProgrammingBase:
    def __init__(
                self, 
                objective_matrix, 
                objective_vector, 
                objective_constant, 
                constraint_matrix, 
                constraint_lower_bound, 
                variable_lower_bound = None, 
                variable_upper_bound = None
                 ):
        self.objective_matrix = coo_matrix(objective_matrix)
        self.objective_vector = objective_vector
        self.objective_constant = objective_constant
        self.constraint_matrix =  coo_matrix(constraint_matrix)
        self.constraint_lower_bound = constraint_lower_bound
        self.variable_lower_bound = variable_lower_bound
        self.variable_upper_bound = variable_upper_bound
        self.n = objective_matrix.shape[0]
        self.m = constraint_matrix.shape[0]
class QuadraticProgrammingTorch(QuadraticProgrammingBase):
    def __init__(
                self, 
                objective_matrix, 
                objective_vector, 
                objective_constant, 
                constraint_matrix, 
                constraint_lower_bound, 
                variable_lower_bound = None, 
                variable_upper_bound = None,
                device: torch.device = torch.device("cpu"),
                vars_ptr = None,
                cons_ptr = None
                 ):
        self.objective_matrix = objective_matrix
        self.objective_vector = objective_vector
        self.objective_constant = objective_constant
        self.constraint_matrix = constraint_matrix
        self.constraint_lower_bound = constraint_lower_bound
        self.variable_lower_bound = variable_lower_bound
        self.variable_upper_bound = variable_upper_bound
        self.n = objective_matrix.shape[-1]
        self.m = constraint_matrix.shape[-2]
        self.device = device
        self.vars_ptr = vars_ptr
        self.cons_ptr = cons_ptr
        
    
    def primal_dual_gap(self, primal, dual):
        return torch.abs(torch.dot(primal, torch.sparse.mm(self.objective_matrix, primal.unsqueeze(-1)).squeeze(-1)) + \
            torch.dot(self.objective_vector, primal) - torch.dot(self.constraint_lower_bound, dual))
    
    def primal_residual(self, primal):
        return torch.clamp_min(torch.sparse.mm(self.constraint_matrix, primal.unsqueeze(-1)).squeeze(-1) - self.constraint_lower_bound, 0)
    
    def dual_residual(self, primal, dual):
        return torch.abs(torch.sparse.mm(self.objective_matrix, primal.unsqueeze(-1)).squeeze(-1)) + self.objective_vector - \
            torch.sparse.mm(self.constraint_matrix.t(), dual.unsqueeze(-1)).squeeze(-1)
    
    def dual_residual_2(self, dual):
        return torch.clamp_min(dual, 0)
    
    def kkt(self, primal, dual):
        values = [
            torch.norm(self.primal_residual(primal)),
            torch.norm(self.dual_residual(primal, dual)),
            torch.norm(self.dual_residual_2(dual)),
            self.primal_dual_gap(primal, dual)
        ]
        values_tensor = torch.stack(values)
        return torch.norm(values_tensor)
    
    def to_base(self) -> QuadraticProgrammingBase:
        return QuadraticProgrammingBase(
                self.objective_matrix, 
                self.objective_vector, 
                self.objective_constant, 
                self.constraint_matrix, 
                self.constraint_lower_bound, 
                self.variable_lower_bound, 
                self.variable_upper_bound
        )
    def to(self, device):
        self.objective_matrix = self.objective_matrix.to(device)
        self.objective_vector = self.objective_vector.to(device)
        self.objective_constant = self.objective_constant.to(device)
        self.constraint_matrix = self.constraint_matrix.to(device)
        self.constraint_lower_bound = self.constraint_lower_bound.to(device)
        self.variable_lower_bound = self.variable_lower_bound.to(device)
        self.variable_upper_bound = self.variable_upper_bound.to(device)
        self.device = device
        if self.vars_ptr is not None:
            self.vars_ptr = self.vars_ptr.to(device)
        if self.cons_ptr is not None:
            self.cons_ptr = self.cons_ptr.to(device)
        return self
                
        
    
def from_base_to_torch(qp: QuadraticProgrammingBase, device: torch.device = torch.device("cpu")) -> QuadraticProgrammingTorch:
    objective_matrix = convert_scip_sparse_to_torch_sparse(qp.objective_matrix, device)
    objective_vector = torch.tensor(qp.objective_vector, dtype=torch.float64, device=device)
    objective_constant = torch.tensor(qp.objective_constant, dtype=torch.float64, device=device)
    constraint_matrix = convert_scip_sparse_to_torch_sparse(qp.constraint_matrix, device)
    constraint_lower_bound = torch.tensor(qp.constraint_lower_bound, dtype=torch.float64, device=device)
    if qp.variable_lower_bound is not None:
        variable_lower_bound = torch.tensor(qp.variable_lower_bound, dtype=torch.float64, device=device)
    else:
        variable_lower_bound = None
    if qp.variable_upper_bound is not None:
        variable_upper_bound = torch.tensor(qp.variable_upper_bound, dtype=torch.float64, device=device)
    else:
        variable_upper_bound = None
    return QuadraticProgrammingTorch(
                objective_matrix, 
                objective_vector, 
                objective_constant, 
                constraint_matrix, 
                constraint_lower_bound, 
                variable_lower_bound, 
                variable_upper_bound,
                device
        )
def collate_fn(batch, device = torch.device("cpu")):
    problems = from_list_to_batch([item['problem'] for item in batch]).to(device)
    primal_labels = torch.cat([item['primal_solution'] for item in batch], dim=0).to(device)
    dual_labels = torch.cat([item['dual_solution'] for item in batch], dim=0).to(device)
    return {
        'problem': problems,
        'primal_solution': primal_labels,
        'dual_solution': dual_labels
    }
            
def from_list_to_batch(qp_list: list) -> QuadraticProgrammingTorch:
    objective_matrix = sparse_matrix_batch([i.objective_matrix for i in qp_list])
    constraint_matrix = sparse_matrix_batch([i.constraint_matrix for i in qp_list])
    vars_size = torch.tensor(np.array([i.objective_vector.shape[0] for i in qp_list]))
    cons_size= torch.tensor(np.array([i.constraint_lower_bound.shape[0] for i in qp_list]))
    vars_ptr = torch.repeat_interleave(torch.arange(vars_size.shape[0]), vars_size)
    cons_ptr = torch.repeat_interleave(torch.arange(cons_size.shape[0]), cons_size)
    
    objective_vector = torch.cat([i.objective_vector for i in qp_list], dim=0)
    objective_constant = torch.stack([i.objective_constant for i in qp_list], dim=0)
    constraint_lower_bound = torch.cat([i.constraint_lower_bound for i in qp_list], dim=0)
    variable_lower_bound = torch.cat([i.variable_lower_bound for i in qp_list], dim=0)
    variable_upper_bound = torch.cat([i.variable_upper_bound for i in qp_list], dim=0)
    return QuadraticProgrammingTorch(
                objective_matrix, 
                objective_vector, 
                objective_constant, 
                constraint_matrix, 
                constraint_lower_bound, 
                variable_lower_bound, 
                variable_upper_bound,
                vars_ptr = vars_ptr,
                cons_ptr = cons_ptr
        )
    
