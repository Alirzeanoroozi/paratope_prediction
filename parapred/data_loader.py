from torch.utils.data import Dataset, DataLoader
import numpy as np


def cv_train_test_split(dataset, cross_round):
    cdrs, lbls = dataset["cdrs"], dataset["lbls"]

    train_indices = [0 for _ in range(len(cdrs))]

    for i in range(cross_round * (len(cdrs) // 10), (cross_round + 1) * (len(cdrs) // 10)):
        train_indices[i] = 1
    for i in range((cross_round + 1) * (len(cdrs) // 10), (cross_round + 2) * (len(cdrs) // 10)):
        train_indices[i] = 2

    cdrs_train = ["".join(cdrs[index]) for index in range(len(cdrs)) if train_indices[index] == 0]
    cdrs_val = ["".join(cdrs[index]) for index in range(len(cdrs)) if train_indices[index] == 1]
    cdrs_test = ["".join(cdrs[index]) for index in range(len(cdrs)) if not train_indices[index] == 2]
    lbls_train = [lbls[index] for index in range(len(lbls)) if train_indices[index] == 0]
    lbls_val = [lbls[index] for index in range(len(lbls)) if train_indices[index] == 1]
    lbls_test = [lbls[index] for index in range(len(lbls)) if not train_indices[index] == 2]

    return cdrs_test, cdrs_val, cdrs_train, lbls_test, lbls_val, lbls_train


def train_test_split(dataset):
    cdrs, lbls = dataset["cdrs"], dataset["lbls"]
    size = len(cdrs)

    np.random.seed(0)  # For reproducibility
    indices = np.random.permutation(size)
    test_size = size // 10

    cdrs_train = ["".join(cdrs[i]) for i in indices[:-2 * test_size]]
    lbls_train = [lbls[i] for i in indices[:-2 * test_size]]

    cdrs_valid = ["".join(cdrs[i]) for i in indices[-2 * test_size:-test_size]]
    lbls_valid = [lbls[i] for i in indices[-2 * test_size:-test_size]]

    cdrs_test = ["".join(cdrs[i]) for i in indices[-test_size:]]
    # print(len(cdrs_test))
    lbls_test = [lbls[i] for i in indices[-test_size:]]

    return cdrs_test, cdrs_valid, cdrs_train, lbls_test, lbls_valid, lbls_train


class ABDataset(Dataset):
    def __init__(self, cdrs, lbls):
        self.cdrs = cdrs
        self.lbls = lbls

    def __len__(self):
        return len(self.cdrs)

    def __getitem__(self, idx):
        return self.cdrs[idx], self.lbls[idx]


def ABLoader(training_data, valid_data, test_data):
    return DataLoader(training_data, batch_size=6),\
        DataLoader(valid_data, batch_size=6),\
        DataLoader(test_data, batch_size=6)
