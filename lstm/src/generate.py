from textloader import  string2code, id2lettre
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(model, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur peuvent être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si 
        start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    
    hidden_state = torch.zeros(64)

    # Run the network over the start of the sentence if not empty
    if len(start) > 0:
        s = string2code(start)
        embeddings = emb(s)
        hidden_state = model(embeddings, start_hidden_state)[-1]
    
    seq = ""
    current_symbol = string2code(' ' if start == "" else start[-1])
    while current_symbol != eos or len(seq) < maxlen:
        embedding = emb(torch.tensor([current_symbol]))
        current_symbol = model.decode(model.one_step(embedding, hidden_state).unsqueeze(0)).softmax(dim=1).argmax().item()
        seq += id2lettre[int(current_symbol)]
    return seq
    
    

def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
