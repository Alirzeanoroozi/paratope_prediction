from os import makedirs
from os.path import isfile
import torch

from data_provider import open_dataset
from data_loader import train_test_split, ABDataset, ABLoader, cv_train_test_split
from model import Parapred, train, evaluate


def run_cv(data, num_iters=10):
    # kf = KFold(n_splits=10)
    # for i, (train_ids, test_ids) in enumerate(kf.split(np.arange(num_structures))):
    # print("Fold: ", i + 1)
    makedirs("precomputed/weights", exist_ok=True)
    for i in range(num_iters):
        print("Cross_validation run", i+1)

        cdrs_test, cdrs_valid, cdrs_train, lbls_test, lbls_valid, lbls_train = cv_train_test_split(data, i)
        train_data = ABDataset(cdrs_train, lbls_train)
        valid_data = ABDataset(cdrs_valid, lbls_valid)
        test_data = ABDataset(cdrs_test, lbls_test)

        train_dataloader, valid_dataloader, test_dataloader = ABLoader(train_data, valid_data, test_data)

        model = Parapred()

        weights = "precomputed/weights/fold-{}.h5".format(i)
        if not isfile(weights):
            train(model, train_dl=train_dataloader, val_dl=valid_dataloader)
            torch.save(model, weights)
        else:
            model = torch.load(weights)

        evaluate(model, test_dataloader, False)


def single_run(dataset):
    cdrs_test, cdrs_valid, cdrs_train, lbls_test, lbls_valid, lbls_train = train_test_split(dataset)

    train_data = ABDataset(cdrs_train, lbls_train)
    valid_data = ABDataset(cdrs_valid, lbls_valid)
    test_data = ABDataset(cdrs_test, lbls_test)

    train_dataloader, valid_dataloader, test_dataloader = ABLoader(train_data, valid_data, test_data)

    model = Parapred()

    # if not isfile("precomputed/chains.pth"):
    #     train(model, train_dl=train_dataloader, val_dl=valid_dataloader)
    #     torch.save(model, "precomputed/chains.pth")
    # else:
    #     model = torch.load("precomputed/chains.pth")
    if not isfile("precomputed/cdrs.pth"):
        train(model, train_dl=train_dataloader, val_dl=valid_dataloader)
        torch.save(model, "precomputed/cdrs.pth")
    else:
        model = torch.load("precomputed/cdrs.pth")

    evaluate(model, test_dataloader, False)


if __name__ == "__main__":
    dataset_path = "../data/merged_data.csv"
    dataset = open_dataset(dataset_path)
    # run_cv(dataset, "cv")
    single_run(dataset)
