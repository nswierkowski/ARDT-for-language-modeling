import torch
from torch.utils.data import Dataset

class ARDTSimpleDataset(Dataset):
    def __init__(self, texts, word2idx, embeddings, context_size=None, alpha=0.5):
        self.alpha = alpha
        self.context_size = context_size

        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.from_numpy(embeddings)
        self.embeddings = embeddings
        self.examples = []

        max_ctx = context_size or max(len(t.split()) for t in texts)
        decay = torch.arange(max_ctx - 1, -1, -1, dtype=torch.float32)
        decay_weights = torch.exp(-alpha * decay)
        decay_weights = decay_weights / decay_weights.sum()

        for text in texts:
            token_idxs = [word2idx[tok] for tok in text.split() if tok in word2idx]
            if len(token_idxs) < 2:
                continue
            tok_tensor = torch.tensor(token_idxs, dtype=torch.long)

            num_tokens = len(tok_tensor)
            for i in range(1, num_tokens):
                start = max(0, i - context_size) if context_size else 0
                context = tok_tensor[start:i] 
                target = tok_tensor[i]

                weights = decay_weights[-len(context):]
                context_embeds = self.embeddings[context]  
                context_vec = torch.matmul(weights, context_embeds) 

                self.examples.append((context_vec, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
