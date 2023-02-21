import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

from tp8_preprocess import TextDataset

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

# tokenizer = spm.SentencePieceProcessor()
# tokenizer.Load(f"wp{vocab_size}.model")
# ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)

batch = next(iter(train_iter))
text = batch.text
labels = batch.labels
print(labels.unique())

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1D(in_channels=1, out_channels=8, kernel_size=3), 
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1D(in_channels=8, out_channels= 16, kernel_size=3),
            nn.MaxPool1d(kernel_size=3, size=2)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#TODO : training loop
