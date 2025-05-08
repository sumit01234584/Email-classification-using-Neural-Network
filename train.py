import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from model import SpamClassifier
from dataloader import SpamDataset

def train(dataloader, model, criterion, optimizer, total_loss):
    model.train()
    for inputs, labels in dataloader:
        labels = labels.view(-1, 1)
        preds  = model(inputs)
        loss   = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, precision, recall, f1

if __name__ == "__main__":
    train_df = pd.read_csv("dataset/training_data.csv")
    test_df = pd.read_csv("dataset/test_data.csv")
    with open("dataset/word_column_dict.pkl", "rb") as f:
        word_column_dict = pickle.load(f)
    vocab_size = len(word_column_dict)

    dataset    = SpamDataset(train_df, word_column_dict, vocab_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    test_dataset    = SpamDataset(test_df, word_column_dict, vocab_size)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model     = SpamClassifier(vocab_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_metrics = dict()
    test_metrics['accuracy'] = list()
    test_metrics['precision'] = list()
    test_metrics['recall'] = list()
    test_metrics['f1'] = list()
    for epoch in tqdm(range(50)):
        total_loss = 0
        total_loss = train(dataloader, model, criterion, optimizer, total_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_dataloader)
        test_metrics['accuracy'].append(test_acc)
        test_metrics['precision'].append(test_prec)
        test_metrics['recall'].append(test_rec)
        test_metrics['f1'].append(test_f1)

        torch.save(model.state_dict(), f"checkpoints/spam_classifier_{epoch}.pth")
    
    print(f"Epoch with max test accuracy: {test_metrics['accuracy'].index(max(test_metrics['accuracy']))} with accuracy {max(test_metrics['accuracy'])}")
    print(f"Epoch with max test precision: {test_metrics['precision'].index(max(test_metrics['precision']))} with precision {max(test_metrics['precision'])}")
    print(f"Epoch with max test precision: {test_metrics['recall'].index(max(test_metrics['recall']))} with recall {max(test_metrics['recall'])}")
    print(f"Epoch with max test F1: {test_metrics['f1'].index(max(test_metrics['f1']))} with F1 {max(test_metrics['f1'])}")