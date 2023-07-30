from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import ToTensor


def train_test_split(dataset):
    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]
    size = len(cdrs)

    np.random.seed(0)  # For reproducibility
    indices = np.random.permutation(size)
    test_size = size // 10

    cdrs_train = cdrs[indices[:-test_size]]
    lbls_train = lbls[indices[:-test_size]]
    masks_train = masks[indices[:-test_size]]

    cdrs_test = cdrs[indices[-test_size:]]
    lbls_test = lbls[indices[-test_size:]]
    masks_test = masks[indices[-test_size:]]

    return cdrs_test, cdrs_train, lbls_test, lbls_train, masks_test, masks_train


class ABDataset(Dataset):
    def __init__(self, cdrs, masks, lbls):
        self.cdrs = cdrs
        self.masks = masks
        self.lbls = lbls
        self.transform = ToTensor()

    def __len__(self):
        return len(self.cdrs)

    def __getitem__(self, idx):
        return self.transform(self.cdrs[idx]), self.transform(self.masks[idx]), self.transform(self.lbls[idx])


def ABloader(training_data, test_data):
    return DataLoader(training_data, batch_size=64, shuffle=True, drop_last=True), DataLoader(test_data, batch_size=64, shuffle=True, drop_last=True)


# class MaskingByLambda(nn.Module):
#     def __init__(self, func):
#         super(MaskingByLambda, self).__init__()
#         self.mask_func = func
#
#     def forward(self, x, mask=None):
#         exd_mask = torch.unsqueeze(self.mask_func(x, mask), dim=-1)
#         return x * exd_mask.float()
#
#
# def mask_by_input(tensor):
#     return lambda input, mask: tensor
#
#
# class MaskedConvolution1D(nn.Conv1d):
#     def __init__(self, *args, **kwargs):
#         super(MaskedConvolution1D, self).__init__(*args, **kwargs)
#
#     def forward(self, x, mask=None):
#         assert mask is not None
#         mask = torch.unsqueeze(mask, dim=-1)
#         x = super(MaskedConvolution1D, self).forward(x)
#         return x * mask.float()