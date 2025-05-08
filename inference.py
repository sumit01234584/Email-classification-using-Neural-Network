import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from model import SpamClassifier
from dataloader import SpamDataset

def evaluate_checkpoint():
    test_df = pd.read_csv("dataset/test_data.csv")
    with open("dataset/word_column_dict.pkl", "rb") as f:
        word_column_dict = pickle.load(f)
    vocab_size = len(word_column_dict)
    test_dataset    = SpamDataset(test_df, word_column_dict, vocab_size)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SpamClassifier(vocab_size)
    model.load_state_dict(torch.load("checkpoints/spam_classifier_2.pth", map_location=torch.device("cpu")))
    model.eval()

    all_preds = list()
    all_labels = list()

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

if __name__ == "__main__":
    evaluate_checkpoint()