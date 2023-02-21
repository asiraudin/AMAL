import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]

#Longueur des séquences
LENGTH = 10
# Hidden dimension
DIM_HIDDEN = 64
#Taille du batch
BATCH_SIZE = 16
# Interval between evaluation
EVAL_EVERY = 10

PATH = "data/"

with open('data/trump_full_speech.txt') as f:
    raw_txt = f.read()

ds = TrumpDataset(raw_txt, maxlen=20)
data_train = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

model = RNN(in_dim=len(id2lettre), h_dim=DIM_HIDDEN, out_dim=len(id2lettre))
optim = torch.optim.Adam(model.parameters())

for epoch in range(100):
    epoch_loss = 0
    n_samples = 0
    for i, (x, y) in enumerate(data_train):
        optim.zero_grad()
        
        # x : bs, len
        # y : bs, len

        init_hstate = torch.zeros(x.size(0), DIM_HIDDEN)

        x = nn.functional.one_hot(x.transpose(0, 1), num_classes=len(id2lettre)).float()
        y = nn.functional.one_hot(y.transpose(0, 1), num_classes=len(id2lettre)).float()
        states = model(x, init_hstate)

        yhat = model.decode(states)
        
        loss = torch.nn.functional.cross_entropy(yhat, y, reduction='sum')        
        # Log
        epoch_loss += loss.item()
        n_samples += x.size(0)
        
        loss.backward()
        optim.step()
    if epoch % EVAL_EVERY == 0:
        hstate = torch.zeros(1, DIM_HIDDEN)
        seq = ""
        current_symbol = lettre2id[""]
        while len(seq) < LENGTH:
            # print(seq, len(seq))
            one_hot = nn.functional.one_hot(torch.tensor(current_symbol), num_classes=len(id2lettre)).float()
            current_symbol = model.decode(model.one_step(one_hot.unsqueeze(0), hstate)).softmax(dim=1).argmax().item()
            seq += id2lettre[int(current_symbol)]
        print("Generated sequence : ", seq)
    print(f"Epoch: {epoch}, Train Loss: {epoch_loss/n_samples}")