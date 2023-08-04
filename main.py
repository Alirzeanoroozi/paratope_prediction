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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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

    dataset = open_dataset("data/dataset.csv")

    cdrs_test, cdrs_train, lbls_test, lbls_train = train_test_split(dataset)

    train_data = ABDataset(cdrs_train, lbls_train)
    test_data = ABDataset(cdrs_test, lbls_test)

    print(train_data[0])
    print(train_data[1])

    print(predict(['TATSSLSSSYLH']))

    for a, b in zip(train_data[1][1], predict(['TATSSLSSSYLH'])[0]):
        print(float(a), "  ", round(float(b), 3))

    # print(predict(["YCQHFYIYPYTFG", "GVNTFGLY", "YPGRGT"]))


# tensor([[0.1975, 0.2226, 0.3391, 0.4048, 0.4451, 0.4627, 0.4134, 0.4476, 0.3792,
#          0.4062, 0.3191, 0.2645, 0.1421, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359],
#         [0.1036, 0.1938, 0.2567, 0.2908, 0.2975, 0.2317, 0.2661, 0.2269, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359],
#         [0.1795, 0.1900, 0.2157, 0.3055, 0.2101, 0.1784, 0.0359, 0.0359, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359, 0.0359,
#          0.0359, 0.0359, 0.0359, 0.0359, 0.0359]], dtype=torch.float64)

# tensor([[0.0685, 0.0134, 0.0561, 0.0507, 0.9299, 0.9437, 0.8590, 0.9153, 0.1119,
#          0.7351, 0.0098, 0.0064, 0.0034, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775],
#         [0.1212, 0.3969, 0.8606, 0.8883, 0.9034, 0.7356, 0.1559, 0.8118, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775],
#         [0.2265, 0.0040, 0.0916, 0.4299, 0.3625, 0.3892, 0.4775, 0.4775, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775, 0.4775,
#          0.4775, 0.4775, 0.4775, 0.4775, 0.4775]], dtype=torch.float64)