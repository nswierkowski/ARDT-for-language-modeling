import os
import numpy as np

try:
    from transformers import GPT2Tokenizer, GPT2Model
except ImportError:
    print("Transformers library not found. Please install it with 'pip install transformers'.")
    GPT2Tokenizer = None
    GPT2Model = None

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import os


def load_glove_embeddings(file_path, vocab, embed_dim):
    """
    Reads GloVe embeddings from a file and returns an array of embeddings for the words in the vocabulary.
    file_path: path to the GloVe file (each line: word and embed_dim values).
    vocab: dictionary mapping word to index.
    """
    embeddings = np.random.randn(len(vocab), embed_dim).astype(np.float32)  
    print(f'embeddings: {len(embeddings)}')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                    vec = np.array(parts[1:], dtype=np.float32)
                    print(f'VEC LENGTH: {len(vec)}')
                    print(f'vocab LENGTH: {len(vocab)}')
                    print(f'word: {word}')
                    print(f'vocab[word]: {vocab[word]}')
                    embeddings[vocab[word]] = vec
    except FileNotFoundError:
        print(f"Plik GloVe nie znaleziony: {file_path}. Używam losowych osadzeń.")
    return embeddings

def build_vocab(texts):
    """
    Create a vocabulary from a list of texts.
    texts: list of strings, each string is a text document.
    Returns a dictionary mapping each unique token to a unique index.
    """
    vocab = {}
    idx = 0
    for text in texts:
        for token in text.lower().split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def load_gpt2_embeddings(vocab, cache_path="cached_gpt2_embeddings.npy", model_name="gpt2", device="cpu"):
    if os.path.exists(cache_path):
        print(f"Loading cached GPT-2 embeddings from {cache_path}")
        return torch.from_numpy(np.load(cache_path)).float().to(device)

    print("Generating GPT-2 embeddings (this may take time)...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name).to(device)
    model.eval()

    embedding_dim = model.config.hidden_size
    embeddings = torch.zeros((len(vocab), embedding_dim), dtype=torch.float32).to(device)

    with torch.no_grad():
        for word, idx in vocab.items():
            tokens = tokenizer(word, return_tensors="pt", add_special_tokens=False).to(device)
            outputs = model(**tokens)
            hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)
            word_embedding = hidden_states.mean(dim=1).squeeze(0)  # shape: [hidden_size]
            embeddings[idx] = word_embedding

    np.save(cache_path, embeddings.cpu().numpy())
    print(f"Saved GPT-2 embeddings to {cache_path}")
    return embeddings


def load_minilm_embeddings(texts, cache_path="cached_minilm_embeddings.npy", model_name="all-MiniLM-L6-v2", device="cpu"):
    if os.path.exists(cache_path):
        print(f"Loading cached MiniLM embeddings from {cache_path}")
        return np.load(cache_path)

    print("Generating MiniLM embeddings (this may take time)...")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    np.save(cache_path, embeddings)
    print(f"Saved MiniLM embeddings to {cache_path}")
    return embeddings

def compute_target_embedding(text, embeddings, vocab):
    tokens = text.lower().split()
    vecs = []
    for token in tokens:
        idx = vocab.get(token)
        if idx is not None and idx < embeddings.shape[0]:
            vecs.append(embeddings[idx])
            
    print(f"[INFO] Number of tokens found in embeddings: {len(vecs)}")
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(embeddings.shape[1])