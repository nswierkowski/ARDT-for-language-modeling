import torch


class InstructionAnswerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, word2idx, embeddings, alpha=0.5):
        self.alpha = alpha
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.examples = []
        
        for input_text, output in zip(inputs, outputs):
            tokens = input_text.split()
            idxs = [word2idx[t] for t in tokens if t in word2idx]
            if not idxs:
                continue 
            decay = torch.arange(len(idxs) - 1, -1, -1, dtype=torch.float32)
            weights = torch.exp(-alpha * decay)
            weights /= weights.sum()
            context_vec = (weights.unsqueeze(0) @ self.embeddings[idxs]).squeeze(0)
            self.examples.append((context_vec, float(output)))  
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
