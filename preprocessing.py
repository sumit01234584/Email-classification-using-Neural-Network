from sklearn import preprocessing
from collections import Counter
import numpy as np
import pandas as pd
import pickle

if __name__ == "__main__":
    data = pd.read_csv('dataset/spam.csv', encoding='latin-1')
    data = data.rename(columns={"v1":"label", "v2":"text"})
    data["label_tag"] = data.label.map({'ham':0, 'spam':1})

    training_data = data[0:4572]
    training_data_length = len(training_data.label)
    training_data.to_csv('dataset/training_data.csv', index=False)
    test_data = data[4572:]
    test_data_length = len(test_data.label)
    test_data.to_csv('dataset/test_data.csv', index=False)

    spam_counts = Counter()
    ham_counts = Counter()
    total_counts = Counter()
    spam_ham_ratios = Counter()
    for i in range(training_data_length):
        if(training_data.label[i] == 0):
            for word in training_data.text[i].split(" "):
                ham_counts[word] += 1
                total_counts[word] += 1
        else:
            for word in training_data.text[i].split(" "):
                spam_counts[word] += 1
                total_counts[word] += 1

    for word,count in list(total_counts.most_common()):
        if(count > 100):
            spam_ham_ratio = spam_counts[word] / float(ham_counts[word]+1)
            spam_ham_ratios[word] = spam_ham_ratio

    for word,ratio in spam_ham_ratios.most_common():
        if(ratio > 1):
            spam_ham_ratios[word] = np.log(ratio)
        else:
            spam_ham_ratios[word] = -np.log((1 / (ratio+0.01)))

    vocab = set(total_counts.keys())
    vocab_size = len(vocab)
    vocab_vector = np.zeros((1, vocab_size))
    word_column_dict = {}
    for i, word in enumerate(vocab):
        word_column_dict[word] = i

    with open("dataset/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("dataset/word_column_dict.pkl", "wb") as f:
        pickle.dump(word_column_dict, f)

    with open("dataset/spam_ham_ratios.pkl", "wb") as f:
        pickle.dump(spam_ham_ratios, f)