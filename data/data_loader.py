import pickle
from os.path import isfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ABDataset(Dataset):
    def __init__(self, chains, labels):
        self.chains = chains
        self.labels = labels

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        return self.chains[idx], self.labels[idx]


def ab_loader(train, valid, test):
    return DataLoader(train, batch_size=16), DataLoader(valid, batch_size=16), DataLoader(test, batch_size=16)


def get_dataloaders(max_len):
    train_df = pd.read_parquet('data/paratope_data/sabdab_train.parquet')
    val_df = pd.read_parquet('data/paratope_data/sabdab_val.parquet')
    test_df = pd.read_parquet('data/paratope_data/sabdab_test.parquet')

    dataset_cache = "precomputed/embeddings.p"
    if isfile(dataset_cache):
        with open(dataset_cache, "rb") as f:
            embedding_dict = pickle.load(f)
    else:
        embedding_dict = {}
    for seqs, direct in zip(
            [train_df['sequence'].to_list(), val_df['sequence'].to_list(), test_df['sequence'].to_list()],
            ["train", "valid", "test"]):
        for i, seq in enumerate(seqs):
            if seq not in embedding_dict:
                print(i + 1, 'new')
                embedding_dict[seq] = torch.load("data/embeddings/" + direct + "/" + str(i + 1) + ".pt")
        with open(dataset_cache, "wb") as f:
            pickle.dump(embedding_dict, f)

    def to_binary(input_list):
        return np.array([0. if c == 'N' else 1. for c in input_list] + [0. for _ in range(max_len - len(input_list))])

    chains_train = [x for x in train_df['sequence'].tolist()]
    labels_train = [to_binary(x) for x in train_df['paratope_labels'].tolist()]

    chains_valid = [x for x in val_df['sequence'].tolist()]
    labels_valid = [to_binary(x) for x in val_df['paratope_labels'].tolist()]

    chains_test = [x for x in test_df['sequence'].tolist()]
    labels_test = [to_binary(x) for x in test_df['paratope_labels'].tolist()]

    train_data = ABDataset(chains_train, labels_train)
    valid_data = ABDataset(chains_valid, labels_valid)
    test_data = ABDataset(chains_test, labels_test)

    return ab_loader(train_data, valid_data, test_data)


def cv_train_test_split(dataset, cross_round):
    chains, labels = dataset["chains"], dataset["labels"]

    train_indices = [0 for _ in range(len(chains))]

    if cross_round == 9:
        for i in range(cross_round * (len(chains) // 10), len(chains)):
            train_indices[i] = 1
        for i in range(0, (len(chains) // 10)):
            train_indices[i] = 2
    else:
        for i in range(cross_round * (len(chains) // 10), (cross_round + 1) * (len(chains) // 10)):
            train_indices[i] = 1
        for i in range((cross_round + 1) * (len(chains) // 10), (cross_round + 2) * (len(chains) // 10)):
            train_indices[i] = 2

    chains_train = ["".join(chains[index]) for index in range(len(chains)) if train_indices[index] == 0]
    chains_val = ["".join(chains[index]) for index in range(len(chains)) if train_indices[index] == 1]
    chains_test = ["".join(chains[index]) for index in range(len(chains)) if not train_indices[index] == 2]

    labels_train = [labels[index] for index in range(len(labels)) if train_indices[index] == 0]
    labels_val = [labels[index] for index in range(len(labels)) if train_indices[index] == 1]
    labels_test = [labels[index] for index in range(len(labels)) if not train_indices[index] == 2]

    return chains_test, chains_val, chains_train, labels_test, labels_val, labels_train
