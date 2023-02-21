# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        return (1/y.size(0)) * torch.linalg.matrix_norm(yhat - y)**2
        
    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        return (2 / y.size(0)) * grad_output * (yhat - y), -(2  / y.size(0)) * grad_output * (yhat - y)


class Linear(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, x, w, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(x, w, b)
        return x @ w + b

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        # grad_output : (q, p)
        # return : (q, n), (n, p), (1, p)

        x, w, b = ctx.saved_tensors
        return grad_output @ torch.transpose(w, 0 , 1), torch.transpose(x, 0, 1) @ grad_output, torch.sum(grad_output, dim=0)


## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

