import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch import Tensor
from torch.nn import Parameter
import os
import psutil
from quadratic_programming import QuadraticProgrammingTorch
import torch_geometric
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphNorm
import copy
import math
class FourierEncoder(torch.nn.Module):
    """Node encoder using Fourier features.
    """
    def __init__(self, level, include_self=True):
        super(FourierEncoder, self).__init__()
        self.level = level
        self.include_self = include_self

    def multiscale(self, x, scales):
        return torch.hstack([x / i for i in scales])

    def forward(self, x):
        device, dtype, orig_x = x.device, x.dtype, x
        scales = 2 ** torch.arange(-self.level / 2, self.level / 2, device=device, dtype=dtype)
        lifted_feature = torch.cat((torch.sin(self.multiscale(x, scales)), torch.cos(self.multiscale(x, scales))), 1)
        return lifted_feature
class SubPrimalNet(nn.Module):
    def __init__(self, config):
        super(SubPrimalNet, self).__init__()
        self.config = config
        self.iters = config['inner_iter']
        self.device = config['device']
        self.hidden_dim = config['hidden_dim']
        self.Theta1 =nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device) for _ in range(self.iters)]
        )
        self.Theta2 = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device) for _ in range(self.iters)]
        )
        self.Theta3 = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device) for _ in range(self.iters)]
        )
        self.delta = Parameter(torch.tensor(2.))
        
    def forward(self, primal, dual, obj_matrix, obj_vector, cons_matrix):
            for i in range(self.iters):
                primal = F.leaky_relu(
                                    self.Theta1[i](primal) - self.delta * 
                                    (self.Theta2[i](torch.sparse.mm(obj_matrix, primal)) + 
                                    obj_vector - self.Theta3[i](torch.sparse.mm(cons_matrix.t(), dual)))
                                    )
            return primal
    
class SubDualNet(nn.Module):
    def __init__(self, config):
        super(SubDualNet, self).__init__()
        self.config = config
        self.device = config['device']
        self.hidden_dim = config['hidden_dim']
        self.Theta1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device)
        self.Theta2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device)
        self.Theta3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device)
        self.sigma = Parameter(torch.tensor(2.))
        
    def forward(self, primal, last_primal, dual, cons_matrix, right_hand_side):
        dual = F.leaky_relu(
                            self.Theta1(dual) + self.sigma * 
                            (torch.sparse.mm(cons_matrix, self.Theta2(primal) - self.Theta3(last_primal)) - right_hand_side)
                            )
        return dual
    

class PDHCGNet(nn.Module):
    def __init__(self, config):
        super(PDHCGNet, self).__init__()
        self.config = config
        self.device = config['device']
        self.hidden_dim = config['hidden_dim']
        self.sub_primal = nn.ModuleList([SubPrimalNet(config) for _ in range(config['outer_iter'])])
        self.sub_dual = nn.ModuleList([SubDualNet(config) for _ in range(config['outer_iter'])])
        self.iters = config['outer_iter']
        self.primal_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device),
            #nn.LayerNorm(self.hidden_dim, device=self.device),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1, bias=False, device=self.device)
        )
        self.dual_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device),
            #nn.LayerNorm(self.hidden_dim, device=self.device),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1, bias=False, device=self.device)
        )
        self.encoder = GNNEncoder(config)
        self._init_weights()
        
    def forward(self, qp:QuadraticProgrammingTorch):
        primal, dual = self.encoder(qp)
        for i in range(self.iters):
            last_primal = primal.clone()
            dual = self.sub_dual[0](primal, last_primal, dual, qp.constraint_matrix, qp.constraint_lower_bound.unsqueeze(-1))
            
            primal = self.sub_primal[0](primal, dual, qp.objective_matrix, qp.objective_vector.unsqueeze(-1), qp.constraint_matrix)        
        return self.primal_output(primal).squeeze(), self.dual_output(dual).squeeze()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.normal_(m.weight, mean=0.0, std=0.001)
                #nn.init.full_(m.bias, 0.0)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
class GNNEncoder(nn.Module):
    def __init__(self, config):
        super(GNNEncoder, self).__init__()
        self.config = config
        self.device = config['device']
        self.hidden_dim = config['gnn_params']['hidden_dim']
        self.num_layers = config['gnn_params']['num_layers']
        self.head = config['gnn_params']['heads']
        self.out_dim = config['hidden_dim']
        self.convs = nn.ModuleList()
        self.feature_encoder_vec = FourierEncoder(math.ceil(self.hidden_dim/6))
        self.feature_encoder_con = FourierEncoder(math.ceil(self.hidden_dim/2))
        var_enc_dim = math.ceil(self.hidden_dim/6)*6
        con_enc_dim = math.ceil(self.hidden_dim/2)*2
        self.convs.append(
            HeteroConv({
                ('vars', 'obj', 'vars'): GATv2Conv(-1, self.hidden_dim//self.head, heads=self.head, edge_dim=1, add_self_loops=False),
                ('cons', 'to', 'vars'): GATv2Conv((-1, -1), self.hidden_dim//self.head, heads=self.head, edge_dim=1, add_self_loops=False),
                ('vars', 'to', 'cons'): GATv2Conv((-1, -1), self.hidden_dim//self.head, heads=self.head, edge_dim=1, add_self_loops=False)
            })
        )
        for _ in range(self.num_layers-1):
            self.convs.append(
                HeteroConv({
                    ('vars', 'obj', 'vars'): GATv2Conv(-1, self.hidden_dim//self.head, heads=self.head, edge_dim=1, add_self_loops=False),
                    ('cons', 'to', 'vars'): GATv2Conv((-1, -1), self.hidden_dim//self.head, heads=self.head, edge_dim=1, add_self_loops=False),
                    ('vars', 'to', 'cons'): GATv2Conv((-1,-1), self.hidden_dim//self.head, heads=self.head, edge_dim=1, add_self_loops=False)
                })
            )
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True, device=self.device),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.out_dim, bias=True, device=self.device)
        )
        self.convs = self.convs.to(self.device)
        self.graph_norm = GraphNorm(self.hidden_dim).to(self.device)
        
    def forward(self, qp:QuadraticProgrammingTorch):
        hdata = HeteroData()
        varibale_lower_bound = copy.deepcopy(qp.variable_lower_bound)
        varibale_upper_bound = copy.deepcopy(qp.variable_upper_bound)
        varibale_lower_bound[torch.isposinf(varibale_lower_bound)] = 1e2
        varibale_upper_bound[torch.isposinf(varibale_upper_bound)] = 1e2
        varibale_lower_bound[torch.isneginf(varibale_lower_bound)] = -1e2
        varibale_lower_bound[torch.isneginf(varibale_upper_bound)] = -1e2
        hdata['vars'].x = torch.stack([qp.objective_vector, varibale_lower_bound, varibale_upper_bound], dim = 0).t()
        hdata['cons'].x = qp.constraint_lower_bound.unsqueeze(-1)
        hdata['vars'].x = self.feature_encoder_vec(hdata['vars'].x)
        hdata['cons'].x = self.feature_encoder_con(hdata['cons'].x)
        coo_obj_matrix = qp.constraint_matrix.to_sparse_coo().coalesce()
        coo_cons_matrix = qp.constraint_matrix.to_sparse_coo().coalesce()
        hdata['vars', 'obj', 'vars'].edge_index = coo_obj_matrix.indices()
        hdata['vars', 'obj', 'vars'].edge_attr = coo_obj_matrix.values()
        hdata['cons', 'to', 'vars'].edge_index = coo_cons_matrix.indices()
        hdata['cons', 'to', 'vars'].edge_attr = coo_cons_matrix.values()
        hdata['vars', 'to', 'cons'].edge_index = coo_cons_matrix.indices()[[1,0]]
        hdata['vars', 'to', 'cons'].edge_attr = coo_cons_matrix.values()
        
        x_dict = hdata.x_dict
        edge_index_dict = hdata.edge_index_dict
        edge_attr_dict = hdata.edge_attr_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict['vars'] = self.graph_norm(x_dict['vars'], qp.vars_ptr)
            x_dict['cons'] = self.graph_norm(x_dict['cons'], qp.cons_ptr)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x_vars = x_dict['vars']
        x_cons = x_dict['cons']
        
        initial_primal = x_vars
        initial_dual = x_cons
        
        return initial_primal, initial_dual

            
        
        
        