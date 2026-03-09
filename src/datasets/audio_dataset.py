from typing import List
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioARDTDataset(Dataset):
    def __init__(self,
                 audio_paths: List[str],
                 sr: int = 22050,
                 n_mfcc: int = 13,
                 hop_length: int = 512,
                 context_frames: int = 5,
                 alpha: float = 0.5):
        """
        Parameters
        ----------
        audio_paths : List[str]
            List of paths to WAV audio files.
        sr : int, default=22050
            Sample rate for loading audio.
        n_mfcc : int, default=13
            Number of MFCC coefficients per frame.
        hop_length : int, default=512
            Hop length between frames when computing MFCCs.
        context_frames : int, default=5
            Number of previous frames to include in the context window.
        alpha : float, default=0.5
            Exponential decay rate applied to context weights.
        """
        self.alpha = alpha
        self.context_frames = context_frames

        self.examples = [] 
        for path in audio_paths:
            y, _ = librosa.load(path, sr=sr)
            mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T

            for t in range(context_frames, len(mfcc)):
                context = mfcc[t - context_frames:t]   
                target  = mfcc[t]                      
                self.examples.append((context, target))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        context, target = self.examples[idx]

        L = context.shape[0]
        decay = torch.exp(-self.alpha * torch.arange(L-1, -1, -1, dtype=torch.float32))
        weights = decay / decay.sum()

        ctx = torch.from_numpy(context).float()
        context_vec = (weights.unsqueeze(0) @ ctx).squeeze(0)
        return context_vec, torch.from_numpy(target).float()
