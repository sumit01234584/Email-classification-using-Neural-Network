import torch
import torch.nn as nn
import torch.nn.functional as F

class SpamClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(vocab_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x