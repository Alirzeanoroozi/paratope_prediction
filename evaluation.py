from sklearn.model_selection import KFold
import numpy as np
import pickle
import torch
from torch import optim, nn
from dataloader import ABDataset, ABloader
from dev_runner import evaluate, compute_classifier_metrics
from pytorch_model import train


def false_neg(y_true, y_pred):
    return torch.squeeze(torch.clamp(y_true - torch.round(y_pred), 0.0, 1.0), dim=-1)


def false_pos(y_true, y_pred):
    return torch.squeeze(torch.clamp(torch.round(y_pred) - y_true, 0.0, 1.0), dim=-1)


def kfold_cv_eval(model_func, dataset, output_file="crossval-data.p", weights_template="weights-fold-{}.h5", seed=0):
    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]
    kf = KFold(n_splits=10, random_state=seed, shuffle=True)

    all_lbls = []
    all_probs = []
    all_masks = []

    num_structures = int(len(cdrs) / 6)
    for i, (train_ids, test_ids) in enumerate(kf.split(np.arange(num_structures))):
        print("Fold: ", i + 1)

        train_idx = structure_ids_to_selection_mask(train_ids, num_structures)
        test_idx = structure_ids_to_selection_mask(test_ids, num_structures)

        cdrs_train, lbls_train, mask_train = cdrs[train_idx], lbls[train_idx], masks[train_idx]
        cdrs_test, lbls_test, mask_test = cdrs[test_idx], lbls[test_idx], masks[test_idx]

        train_data = ABDataset(cdrs_train, mask_train, lbls_train)
        test_data = ABDataset(cdrs_test, mask_test, lbls_test)

        train_dataloader, test_dataloader = ABloader(train_data, test_data)

        model = model_func()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_function = nn.BCELoss()

        train(model,
              optimizer=optimizer,
              loss_fn=loss_function,
              train_dl=train_dataloader,
              val_dl=test_dataloader)

        torch.save(model.state_dict(), "precomputed/sabdab.pth")

        probs_test = evaluate(model, test_data)

        test_seq_lens = np.sum(np.squeeze(mask_test), axis=1)
        probs_flat = flatten_with_lengths(probs_test, test_seq_lens)
        lbls_flat = flatten_with_lengths(lbls_test, test_seq_lens)

        compute_classifier_metrics([lbls_flat], [probs_flat])

        model.save_weights(weights_template.format(i))

        probs_test = model.predict([cdrs_test, np.squeeze(mask_test)])
        all_lbls.append(lbls_test)
        all_probs.append(probs_test)
        all_masks.append(mask_test)

    lbl_mat = np.concatenate(all_lbls)
    prob_mat = np.concatenate(all_probs)
    mask_mat = np.concatenate(all_masks)

    with open(output_file, "wb") as f:
        pickle.dump((lbl_mat, prob_mat, mask_mat), f)


def structure_ids_to_selection_mask(idx, num_structures):
    mask = np.zeros((num_structures * 6, ), dtype=np.bool)
    offset = idx * 6
    for i in range(6):
        mask[offset + i] = True
    return mask


def open_crossval_results(folder="runs/cv-ab-seq", num_results=10,
                          loop_filter=None, flatten_by_lengths=True):
    class_probabilities = []
    labels = []

    for r in range(num_results):
        result_filename = "{}/run-{}.p".format(folder, r)
        with open(result_filename, "rb") as f:
            lbl_mat, prob_mat, mask_mat = pickle.load(f)

        # Get entries corresponding to the given loop
        if loop_filter is not None:
            lbl_mat = lbl_mat[loop_filter::6]
            prob_mat = prob_mat[loop_filter::6]
            mask_mat = mask_mat[loop_filter::6]

        if not flatten_by_lengths:
            class_probabilities.append(prob_mat)
            labels.append(lbl_mat)
            continue

        # Discard sequence padding
        seq_lens = np.sum(np.squeeze(mask_mat), axis=1)
        p = flatten_with_lengths(prob_mat, seq_lens)
        l = flatten_with_lengths(lbl_mat, seq_lens)

        class_probabilities.append(p)
        labels.append(l)

    return labels, class_probabilities


def flatten_with_lengths(matrix, lengths):
    seqs = []
    for i, example in enumerate(matrix):
        seq = example[:lengths[i]]
        seqs.append(seq)
    return np.concatenate(seqs)
