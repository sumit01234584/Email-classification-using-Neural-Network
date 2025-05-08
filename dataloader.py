import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle

class SpamDataset(Dataset):
    def __init__(self, data, word_column_dict, vocab_size):
        self.texts = data['text'].values
        self.labels = data['label_tag'].values
        self.word_column_dict = word_column_dict
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        vector = np.zeros(self.vocab_size)
        for word in text.split():
            if word in self.word_column_dict:
                vector[self.word_column_dict[word]] += 1
        return torch.tensor(vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)