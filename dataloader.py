from torch.utils.data import Dataset, DataLoader
import numpy as np


def structure_ids_to_selection_mask(idx, num_structures):
    mask = np.zeros((num_structures * 6, ), dtype=np.bool)
    offset = idx * 6
    for i in range(6):
        mask[offset + i] = True
    return mask


def cv_train_test_split(dataset, train_ids, test_ids, num_structures):
    cdrs, lbls = dataset["cdrs"], dataset["lbls"]

    train_idx = structure_ids_to_selection_mask(train_ids, num_structures)
    test_idx = structure_ids_to_selection_mask(test_ids, num_structures)

    cdrs_train, lbls_train = cdrs[train_idx], lbls[train_idx]
    cdrs_test, lbls_test = cdrs[test_idx], lbls[test_idx]

    return cdrs_test, cdrs_train, lbls_test, lbls_train


def train_test_split(dataset):
    cdrs, lbls = dataset["cdrs"], dataset["lbls"]
    size = len(cdrs)

    np.random.seed(0)  # For reproducibility
    indices = np.random.permutation(size)
    test_size = size // 10

    cdrs_train = ["".join(cdrs[i]) for i in indices[:-test_size]]
    lbls_train = [lbls[i] for i in indices[:-test_size]]

    cdrs_test = ["".join(cdrs[i]) for i in indices[-test_size:]]
    lbls_test = [lbls[i] for i in indices[-test_size:]]

    return cdrs_test, cdrs_train, lbls_test, lbls_train


class ABDataset(Dataset):
    def __init__(self, cdrs, lbls):
        self.cdrs = cdrs
        self.lbls = lbls

    def __len__(self):
        return len(self.cdrs)

    def __getitem__(self, idx):
        return self.cdrs[idx], self.lbls[idx]


def ABloader(training_data, test_data):
    return DataLoader(training_data, batch_size=16, drop_last=True), DataLoader(test_data, batch_size=16, drop_last=True)
