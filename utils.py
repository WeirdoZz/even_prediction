import torch
import random

def sample_event(distribution,epsilon):
    if random.uniform(0, 1) > epsilon:
        sample=torch.argmax(distribution,dim=-1)
    else:
        sample=torch.randint(0,distribution.shape[1],(distribution.shape[0],1))

    return sample