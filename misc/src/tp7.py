import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

from datamaestro import prepare_dataset


# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, h_dim), 
            nn.ReLU(), 
            nn.Linear(h_dim, h_dim), 
            nn.ReLU(), 
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim))
        self.tracked_layers = [0, 2, 4]
    
    def forward(self, x):
        outputs = []
        h = x
        for i, l in enumerate(self.layers):
            h = l(h)
            if (i - 1) in self.tracked_layers:
                outputs.append(store_grad(h))
        return h, outputs


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


writer = SummaryWriter("runs/tp7/tp7_test_10")

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")

train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()
train_tensor, train_labels_tensor = torch.from_numpy(train_images), torch.from_numpy(train_labels)
test_tensor, test_labels_tensor = torch.from_numpy(test_images), torch.from_numpy(test_labels)

# Sous-échantillonnage()
size_train = len(train_images)
size = int(0.05 * size_train)
train_images, train_labels = train_images[:size], train_labels[:size]

tr_dataset = CustomMNIST(train_tensor, train_labels_tensor)
te_dataset = CustomMNIST(test_tensor, test_labels_tensor)
dataloader = DataLoader(tr_dataset, batch_size=300, shuffle=True)
test_dataloader = DataLoader(te_dataset, batch_size=300, shuffle=True)

image_dims = train_images[0].shape

model = MLP(in_dim=image_dims[0]*image_dims[1], h_dim=100, out_dim=10)
optim = torch.optim.Adam(model.parameters())

for epoch in range(1000):
    epoch_loss = 0
    n_samples = 0
    model.train()
    for i, (x, y) in enumerate(dataloader):
        optim.zero_grad()
        yhat, outputs = model(x)
        loss = F.cross_entropy(yhat, y, reduction='sum')
        
        # Log
        epoch_loss += loss.item()
        n_samples += len(x)

        loss.backward()
        optim.step()

    if epoch % 50 == 0:
        for i, l in enumerate(model.tracked_layers):
            writer.add_histogram('weights ' + str(i), model.layers[l].weight, epoch)
            writer.add_histogram('grads ' + str(i), outputs[i].grad, epoch)
        writer.add_histogram('entropy', torch.special.entr(yhat), epoch)
        writer.add_histogram('entropy_random', torch.special.entr(torch.randn_like(yhat)), epoch)
        
        test_epoch_loss = 0
        test_n_samples = 0
        model.eval()
        for i, (x, y) in enumerate(test_dataloader):
            yhat, _ = model(x)
            loss = F.cross_entropy(yhat, y, reduction='sum')
            
            # Log
            test_epoch_loss += loss.item()
            test_n_samples += len(x)
        writer.add_scalar('test_loss_epoch', test_epoch_loss/test_n_samples, epoch)
        print(f"Epoch: {epoch}, Loss test: {test_epoch_loss/test_n_samples}")
    writer.add_scalar('train_loss_epoch', epoch_loss/n_samples, epoch)
    print(f"Epoch: {epoch}, Loss : {epoch_loss/n_samples}")
