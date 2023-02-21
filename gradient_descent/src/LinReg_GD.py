import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
import torch.nn.functional as F

# writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)


def full_batch_GD(x, y):
    # Les paramètres du modèle à optimiser
    w = torch.randn(x.size(-1), y.size(-1), requires_grad=True)
    b = torch.randn(y.size(-1), requires_grad=True)

    lr = 1e-7

    for n_iter in range(100):
        yhat = datax @ w + b
        loss = ((datay - yhat)**2).sum() / datay.size(0)
        
        # `loss` doit correspondre au coût MSE calculé à cette itération
        # on peut visualiser avec
        # tensorboard --logdir runs/
        # writer.add_scalar('Loss/train', loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            w.grad.data.zero_()
            b.grad.data.zero_()

def mini_batch_GD(x, y, bs):
    # Les paramètres du modèle à optimiser
    w = torch.randn(x.size(1), y.size(1), requires_grad=True)
    b = torch.randn(y.size(1), requires_grad=True)

    lr = 1e-7

    for n_iter in range(100):
        samples_left = x.size(0)
        batch_index = 0
        loss_epoch = 0
        while samples_left > 0:
            # Create batch
            batchx = x[batch_index * bs : (batch_index + 1) * bs, :]
            batchy = y[batch_index * bs : (batch_index + 1) * bs, :]
            # Pad if needed
            
            bs_ = min(bs, samples_left)
        
            yhat = batchx @ w + b
            try:
                batch_error = ((batchy - yhat)**2).sum()
            except:
                print(batchy.size(), yhat.size())
                assert False

            loss_batch = batch_error / bs_
            loss_epoch += batch_error

            # `loss` doit correspondre au coût MSE calculé à cette itération
            # on peut visualiser avec
            # tensorboard --logdir runs/
            # writer.add_scalar('Loss/batch', loss_batch, n_iter)

            # Sortie directe
            print(f"Itérations {n_iter}, batch {batch_index}: loss {loss_batch}")

            loss_batch.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad

                w.grad.data.zero_()
                b.grad.data.zero_()
            
            batch_index += 1
            samples_left -= bs_
        
        loss_epoch /= x.size(0)
        print(f"Itérations {n_iter}: loss {loss_epoch}")
        # writer.add_scalar('Loss/epoch', loss_batch, n_iter)

def SGD(x, y):
    # Les paramètres du modèle à optimiser
    w = torch.randn(x.size(1), y.size(1), requires_grad=True)
    b = torch.randn(y.size(1), requires_grad=True)

    lr = 1e-7

    for n_iter in range(100):
        loss_epoch = 0
        for i in range(y.size(0)):
            yhat =  x[i, :] @ w + b
            loss = (y[i, :] - yhat)**2
            loss_epoch += loss.item()

            # `loss` doit correspondre au coût MSE calculé à cette itération
            # on peut visualiser avec
            # tensorboard --logdir runs/
            # writer.add_scalar('Loss/train', loss, n_iter)

            # Sortie directe
                
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad

                w.grad.data.zero_()
                b.grad.data.zero_()
        print(f"Itérations {n_iter}: loss {loss_epoch / x.size(0)}")

mini_batch_GD(datax, datay, 50)