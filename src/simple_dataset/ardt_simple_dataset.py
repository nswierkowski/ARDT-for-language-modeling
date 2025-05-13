import numpy as np
from torch.utils.data import Dataset

class ARDTSimpleDataset(Dataset):
    def __init__(self, texts, word2idx, embeddings, context_size=None, alpha=0.5):
        """
        texts: lista zdań (każde to string).
        word2idx: słownik słowo->indeks.
        embeddings: numpy array [vocab_size x embed_dim] z osadzeniami.
        context_size: maksymalna liczba tokenów w kontekście (dla sliding window).
        alpha: współczynnik zaniku wykładniczego.
        """
        self.examples = []
        self.alpha = alpha
        self.context_size = context_size
        self.embeddings = embeddings
        self.word2idx = word2idx
        
        for text in texts:
            tokens = text.split()
            # Budujemy sekwencje treningowe (prefixy → kolejny token)
            for i in range(1, len(tokens)):
                context_tokens = tokens[:i]
                if context_size is not None and i > context_size:
                    context_tokens = context_tokens[-context_size:]  # weź ostatnie context_size tokenów
                # Konwertuj na indeksy i pomiń nieznane słowa
                context_inds = [word2idx[w] for w in context_tokens if w in word2idx]
                if not context_inds:
                    continue
                # Oblicz ważoną średnią
                L = len(context_inds)
                # wagi malejące wykładniczo: ostatni token największa waga = 1
                weights = np.exp(-alpha * (np.arange(L-1, -1, -1)))  
                weights = weights / weights.sum()
                # wektory kontekstu
                vecs = embeddings[context_inds]  # shape (L, D)
                context_vec = np.dot(weights, vecs)  # shape (D,)
                target_token = tokens[i]
                if target_token in word2idx:
                    target_idx = word2idx[target_token]
                    self.examples.append((context_vec.astype(np.float32), target_idx))

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]  # (wektor_kontekstu, label_idx)
