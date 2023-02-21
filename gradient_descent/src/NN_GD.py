import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
import torch.nn.functional as F

# writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax, dtype=torch.float)
datay = torch.tensor(datay, dtype=torch.float).reshape(-1,1)


class DummyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.MSELoss()
    
    def forward(x, y):
        yhat = self.linear2(self.activation(self.linear1(x)))
        return self.loss(yhat, y)

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.MSELoss()
        self.layers = self.Sequential(self.linear1, self.activation, self.linear2)
    
    def forward(x, y):
        yhat = self.layers(x)
        return self.loss(x, y)

model = DummyNN(datax.size(1), 32, datay.size(1))
optim = torch.optim.Adam(model.parameters())
optim.zero_grad()


for epoch in range(100):
    loss = model(datax, datay)
    loss.backward()
    optim.zero_grad()
    optim.step()

