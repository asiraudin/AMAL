
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
from torch.nn import functional as F


def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    mask = (~(target == padcar)).int()
    loss = F.cross_entropy(output, target, reduction='none') * mask
    return loss / mask.sum()



class RNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.linear_x = nn.Linear(in_dim, h_dim)
        self.linear_h = nn.Linear(h_dim, h_dim)
        self.linear_d = nn.Linear(h_dim, out_dim)


    def forward(self, x, h):
        '''
            x : length, bs, in_dim
            h : bs, h_dim
        '''
        out = self.one_step(x[0, ...], h).unsqueeze(0)
        for i in range(1, x.size(0)):
            h = self.one_step(x[i], h)
            out = torch.cat((out, h.unsqueeze(0)), dim=0)
        return out

    def one_step(self, x, h):
        '''
            x : bs, in_dim
            h : bs, h_dim

            return:
                h_next : bs, h_dim
        '''
        return torch.tanh(self.linear_x(x) + self.linear_h(h))

    def decode(self, h):
        '''
            h : bs, h_dim
        '''
        # Softmax included in cross_entropy_loss
        return self.linear_d(h)
    

class LSTM(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.linear_f = nn.Linear(in_dim + h_dim, h_dim)
        self.linear_i = nn.Linear(in_dim + h_dim, h_dim)
        self.linear_c = nn.Linear(in_dim + h_dim, h_dim)
        self.linear_o = nn.Linear(in_dim + h_dim, h_dim)
        self.linear_d = nn.Linear(h_dim, out_dim)
        
    def forward(self, x, h, c):
        '''
            x : length, bs, in_dim
            h : bs, h_dim
            c : bs, h_dim
        '''
        out, c = self.one_step(x[0], h, c).unsqueeze(0)
        for i in range(1, x.size(0)):
            h, c = self.one_step(x[i], h, c)
            out = torch.cat((out, h.unsqueeze(0)), dim=0)
        return out

    def one_step(self, x, h, c):
        '''
            x : bs, in_dim
            h : bs, h_dim

            return:
                h_next : bs, h_dim
        '''
        f_t = F.softmax(self.linear_f(torch.cat((h, x), dim=1)), dim=1)
        i_t = F.softmax(self.linear_i(torch.cat((h, x), dim=1)), dim=1)
        c = f_t * c + i_t * torch.tanh(self.linear_c(torch.cat((h, x), dim=1)))
        o_t = F.softmax(self.linear_o(torch.cat((h, x), dim=1)), dim=1)
        return o_t * torch.tanh(c), c
        
    def decode(self, h):
        '''
            h : bs, h_dim
        '''
        # Softmax included in cross_entropy_loss
        return self.linear_d(h)

class GRU(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.linear_z = nn.Linear(in_dim + h_dim, h_dim)
        self.linear_r = nn.Linear(in_dim + h_dim, h_dim)
        self.linear_h = nn.Linear(in_dim + h_dim, h_dim)
        self.linear_d = nn.Linear(h_dim, out_dim)

    def forward(self, x, h):
        '''
            x : length, bs, in_dim
            h : bs, h_dim
        '''
        out = self.one_step(x[0], h).unsqueeze(0)
        for i in range(1, x.size(0)):
            h = self.one_step(x[i], h)
            out = torch.cat((out, h.unsqueeze(0)), dim=0)
        return out

    def one_step(self, x, h):
        '''
            x : bs, in_dim
            h : bs, h_dim

            return:
                h_next : bs, h_dim
        '''
        print(x.size()), 
        z_t = F.softmax(self.linear_z(torch.cat((h, x), dim=1)), dim=1)
        r_t = F.softmax(self.linear_r(torch.cat((h, x), dim=1)), dim=1)
        rh = r_t * h
        return (1 - z_t) * h + z_t * F.tanh(self.linear_h(torch.cat((rh, x), dim=1)))
    
    def decode(self, h):
        '''
            h : bs, h_dim
        '''
        # Softmax included in cross_entropy_loss
        return self.linear_d(h)


#Longueur des séquences
LENGTH = 50
# Hidden dimension
DIM_HIDDEN = 64
#Taille du batch
BATCH_SIZE = 16
# Interval between evaluation
EVAL_EVERY = 10


with open('data/trump_full_speech.txt') as f:
    raw_txt = f.read()

ds = TextDataset(raw_txt)
loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=128)

model = RNN(in_dim=64, h_dim=DIM_HIDDEN, out_dim=len(id2lettre))
embedding = nn.Embedding(num_embeddings=len(id2lettre), embedding_dim=64)
optim = torch.optim.Adam(model.parameters())

for epoch in range(100):
    epoch_loss = 0
    n_samples = 0
    for i, x in enumerate(loader):
        if i >= 3:
            break
        optim.zero_grad()
        emb = embedding(x)
        init_hstate = torch.zeros(x.size(1), DIM_HIDDEN)
        states = model(emb, init_hstate)
    
        xhat = model.decode(states)
        
        x = x.flatten()
        xhat = xhat.flatten(end_dim=-2)
        
        loss = maskedCrossEntropy(xhat, x, 0).mean()       
        # Log
        epoch_loss += loss.sum().item()
        n_samples += x.size(0)
        
        loss.backward()
        optim.step()
    if epoch % EVAL_EVERY == 0:
        sample = generate(model, embedding, "", EOS_IX)
        print("Generated sequence : ", sample)
    print(f"Epoch: {epoch}, Train Loss: {epoch_loss/n_samples}")