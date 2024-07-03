import torch

print(torch.__version__)

a = torch.tensor([1.5, 2.5, 3.5])
b = torch.tensor([2.5, 3.5, 4.5])
c = torch.add(a, b)
print(c)