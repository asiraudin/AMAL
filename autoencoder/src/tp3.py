from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
# Téléchargement des données

from datamaestro import prepare_dataset

class CustomMNIST(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        tf_image = F.normalize(self.images[index].float())
        tf_image = tf_image.flatten()
        return tf_image, self.labels[index]

    def __len__(self):
        return self.images.size(0)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation_encoder = nn.ReLU()
        self.linear_transpose = nn.Linear(hidden_dim, input_dim)
        self.activation_decoder = nn.Sigmoid()
        
        self.layers = nn.Sequential(
            self.linear,
            self.activation_encoder,
            self.linear_transpose,
            self.activation_decoder
        )

        self.layers[2].weight.data = self.layers[0].weight.data.transpose(0, 1)

    def forward(self, x):
        return self.layers(x)

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0

ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

images_tensor, labels_tensor = torch.from_numpy(train_images), torch.from_numpy(train_labels)
dataset = CustomMNIST(images_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)

savepath = Path("model.pch")

if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)

else:
    model = AutoEncoder(input_dim=images_tensor.size(1)*images_tensor.size(2), hidden_dim=128)
    optim = torch.optim.Adam(model.parameters())
    optim.zero_grad()
    state = State(model, optim)

for epoch in range(state.epoch, 100):
    epoch_loss = 0
    for i, (x, y) in enumerate(dataloader):
        state.optim.zero_grad()
        yhat = state.model(x)
        loss = F.cross_entropy(yhat, y)
        
        # Log
        epoch_loss += loss.item()
        writer.add_scalar('train_loss/batch', loss, i)

        loss.backward()
        state.optim.step()
        state.iteration += 1
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state, fp)
    writer.add_scalar('train_loss/epoch', epoch_loss, i)
    print(f"Epoch: {epoch}, Loss: {epoch_loss}")