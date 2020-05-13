import torch

import time

from emd import EMDLoss

dist =  EMDLoss()

p1 = torch.tensor([1,5,3,2,0,4.])[:,None,None].cuda()
p2 = torch.tensor([0,1,2,3,4,5.])[:,None,None].cuda()
import ipdb ; ipdb.set_trace()
p1.requires_grad = True
p2.requires_grad = True

s = time.time()
cost = dist(p1, p2)
emd_time = time.time() - s

import ipdb ; ipdb.set_trace()
print('Time: ', emd_time)
print(cost)
loss = torch.sum(cost)
print(loss)
loss.backward()
print(p1.grad)
print(p2.grad)
