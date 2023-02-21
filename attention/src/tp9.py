import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from models import BasicAttention, SimpleAttention, SelfAttention

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)

w2i, emb, train_dataset, test_dataset = get_imdb_data()
i2w = dict(zip(w2i.values(),w2i.keys()))

dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


# model = SimpleAttention(emb, 50, 128, 1)
model = SelfAttention(emb, 50, 128, 1)
optim = torch.optim.Adam(model.parameters())
# writer = SummaryWriter("runs/tp8/linear")

for epoch in range(5):
    epoch_loss = 0
    n_samples = 0
    # model.train()
    for i, (x, y) in enumerate(train_dataset):
        if i > 5:
            break
        optim.zero_grad()
        yhat = model(x) 
        loss = F.binary_cross_entropy_with_logits(yhat, torch.tensor([y], dtype=torch.float32))
        
        # Log
        epoch_loss += loss.item()
        n_samples += len(x)

        loss.backward()
        optim.step()

    if epoch % 10 == 0:     
        test_epoch_loss = 0
        test_n_samples = 0
        model.eval()
        for i, (x, y) in enumerate(test_dataset):
            yhat = model(x)
            loss = F.binary_cross_entropy_with_logits(yhat, torch.tensor([y], dtype=torch.float32))
            
            # Log
            test_epoch_loss += loss.item()
            test_n_samples += len(x)
        # writer.add_scalar('test_loss_epoch', test_epoch_loss/test_n_samples, epoch)
        print(f"Epoch: {epoch}, Loss test: {test_epoch_loss/test_n_samples}")
    # writer.add_scalar('train_loss_epoch', epoch_loss/n_samples, epoch)
    print(f"Epoch: {epoch}, Loss : {epoch_loss/n_samples}")