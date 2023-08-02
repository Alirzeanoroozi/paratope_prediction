from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import ToTensor


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
    return DataLoader(training_data, batch_size=64, shuffle=True, drop_last=True), DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)
