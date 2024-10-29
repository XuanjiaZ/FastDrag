import torch
import time

def split_into_N_equal_parts(L, steps):
    if L==0:
        return L.repeat(int(steps))
    op = L/torch.abs(L)
    L = torch.abs(L)
    base_value = L // steps
    remainder = L % steps
    result = base_value.repeat(int(steps))
    result[int(remainder):] += 1
    result *=op

    return result   # tensor([3., 3., 3., 2., 2.])

