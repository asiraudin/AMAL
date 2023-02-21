import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
torch.autograd.gradcheck(mse, (yhat, y))

#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)
x = torch.randn(10, 4, requires_grad=True, dtype=torch.float64)
w = torch.randn(4, 8, requires_grad=True, dtype=torch.float64)
b = torch.randn(8, requires_grad=True, dtype=torch.float64)

torch.autograd.gradcheck(linear, (x, w, b))