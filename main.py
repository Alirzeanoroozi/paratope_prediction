from os.path import isfile
import torch
from data.data_loader import cv_train_test_split, get_dataloaders
from model import Parabert, train, evaluate


# def run_cv(data, num_iters=10):
#     for i in range(num_iters):
#         print("Cross_validation run", i + 1)
#
#         cdrs_test, cdrs_valid, cdrs_train, lbls_test, lbls_valid, lbls_train = cv_train_test_split(data, i)
#         train_data = ABDataset(cdrs_train, lbls_train)
#         valid_data = ABDataset(cdrs_valid, lbls_valid)
#         test_data = ABDataset(cdrs_test, lbls_test)
#
#         train_dataloader, valid_dataloader, test_dataloader = ab_loader(train_data, valid_data, test_data)
#
#         model = Parabert()
#
#         weights = "precomputed/weights/fold-{}.h5".format(i + 1)
#         if not isfile(weights):
#             train(model, train_dl=train_dataloader, val_dl=valid_dataloader)
#             torch.save(model, weights)
#         else:
#             model = torch.load(weights)
#
#         evaluate(model, test_dataloader, False, i)


def single_run():
    test_dataloader, train_dataloader, valid_dataloader = get_dataloaders(max_len=150)

    model = Parabert()
    model_weight = "precomputed/model_weight.pth"

    if not isfile(model_weight):
        train(model, train_dl=train_dataloader, val_dl=valid_dataloader)
        torch.save(model, model_weight)
    else:
        model = torch.load(model_weight)

    return evaluate(model, test_dataloader, eval_phase=False)


if __name__ == "__main__":
    # run_cv()
    threshold = single_run()
    # show_result(threshold)
