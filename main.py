from os import makedirs
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim, nn
from cnn import generate_mask
from preprocessing import encode_batch
from data_provider import open_dataset
from dataloader import train_test_split, ABDataset, ABloader, structure_ids_to_selection_mask, cv_train_test_split
from pytorch_model import Parapred, train, evaluate, PARAPRED_MAX_LEN


def run_cv(dataset, output_folder, num_iters=10):
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset, dataset_cache=cache_file)

    makedirs(output_folder + "/weights", exist_ok=True)
    for i in range(num_iters):
        print("Crossvalidation run", i+1)
        output_file = "{}/run-{}.p".format(output_folder, i)
        weights_template = output_folder + "/weights/run-" + str(i) + "-fold-{}.h5"
        kfold_cv_eval(Parapred, dataset, output_file, weights_template, seed=i)


def kfold_cv_eval(model_func, dataset, output_file="crossval-data.p", weights_template="weights-fold-{}.h5", seed=0):
    model = model_func()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_function = nn.BCELoss()
    kf = KFold(n_splits=10, random_state=seed, shuffle=True)

    all_lbls = []
    all_probs = []

    num_structures = int(len(dataset['cdrs']) / 6)
    for i, (train_ids, test_ids) in enumerate(kf.split(np.arange(num_structures))):
        print("Fold: ", i + 1)

        cdrs_test, cdrs_train, lbls_test, lbls_train = cv_train_test_split(dataset, train_ids, test_ids, num_structures)

        train_data = ABDataset(cdrs_train, lbls_train)
        test_data = ABDataset(cdrs_test, lbls_test)

        train_dataloader, test_dataloader = ABloader(train_data, test_data)

        train(model,
              optimizer=optimizer,
              loss_fn=loss_function,
              train_dl=train_dataloader,
              val_dl=test_dataloader)


        evaluate(model, test_data)

        test_seq_lens = np.sum(np.squeeze(mask_test), axis=1)
        probs_flat = flatten_with_lengths(probs_test, test_seq_lens)
        lbls_flat = flatten_with_lengths(lbls_test, test_seq_lens)

        compute_classifier_metrics([lbls_flat], [probs_flat])


        probs_test = model.predict([cdrs_test, np.squeeze(mask_test)])
        all_lbls.append(lbls_test)
        all_probs.append(probs_test)
        all_masks.append(mask_test)

    lbl_mat = np.concatenate(all_lbls)
    prob_mat = np.concatenate(all_probs)
    mask_mat = np.concatenate(all_masks)

    torch.save(model.state_dict(), "precomputed/sabdab.pth")
    model.save_weights(weights_template.format(i))

    with open(output_file, "wb") as f:
        pickle.dump((lbl_mat, prob_mat, mask_mat), f)


def single_run(dataset_file):
    dataset = open_dataset(dataset_file)

    cdrs_test, cdrs_train, lbls_test, lbls_train = train_test_split(dataset)

    train_data = ABDataset(cdrs_train, lbls_train)
    test_data = ABDataset(cdrs_test, lbls_test)

    train_dataloader, test_dataloader = ABloader(train_data, test_data)

    model = Parapred()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.BCELoss()

    train(model,
          optimizer=optimizer,
          loss_fn=loss_function,
          train_dl=train_dataloader,
          val_dl=test_dataloader)

    torch.save(model.state_dict(), "precomputed/sabdab_new.pth")

    # evaluate(model, test_data)


def predict(cdrs):
    sequences, lengths = encode_batch(cdrs, max_length=PARAPRED_MAX_LEN)

    # Generate a mask for the input
    m = generate_mask(sequences, sequence_lengths=lengths)

    p = Parapred()
    p.load_state_dict(torch.load("precomputed/sabdab_new.pth"))

    # Evaluation mode with no gradient computations
    _ = p.eval()
    with torch.no_grad():
        probabilities = p(sequences, m, lengths)

    return probabilities.squeeze(2).type(torch.float64)


if __name__ == "__main__":
    single_run("data/dataset.csv")

    # dataset = open_dataset("data/dataset.csv")
    #
    # cdrs_test, cdrs_train, lbls_test, lbls_train = train_test_split(dataset)
    #
    # train_data = ABDataset(cdrs_train, lbls_train)
    # test_data = ABDataset(cdrs_test, lbls_test)
    #
    # print(train_data[0])
    # print(train_data[1])
    #
    # print(predict(['TATSSLSSSYLH']))
    #
    # for a, b in zip(train_data[1][1], predict(['TATSSLSSSYLH'])[0]):
    #     print(float(a), "  ", round(float(b), 3))
    #
    # # print(predict(["YCQHFYIYPYTFG", "GVNTFGLY", "YPGRGT"]))
