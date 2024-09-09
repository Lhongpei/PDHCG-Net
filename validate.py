import torch
from quadratic_programming import from_base_to_torch
import numpy as np
import math
from quadratic_programming import collate_fn
def validate(model, dataset, config):
    loss = torch.nn.SmoothL1Loss()
    loss_epoch = 0
    device = config['device']
    batchsize = config['batchsize']
    shuffled_idxes = np.random.permutation(len(dataset))

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
        problem = instance['problem'].to(device)
        primal_label = instance['primal_solution'].to(device)
        dual_label = instance['dual_solution'].to(device)
        primal, dual = model(problem)
        loss_per_epoch = loss(primal, primal_label) + loss(dual, dual_label)
        loss_epoch += loss_per_epoch.item()
    loss_epoch /= len(shuffled_idxes)
    print(f"Validation Loss: {loss_epoch}")
    return loss_epoch