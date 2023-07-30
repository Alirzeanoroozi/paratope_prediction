import numpy as np
from torch import optim, nn

from data_provider import open_dataset
from parapred.dataloader import ABDataset, train_test_split, ABloader
from parapred.pytorch_model import ab_seq_model, train
from torchsummary import summary


def single_run(dataset_file):
    dataset = open_dataset(dataset_file)

    cdrs_test, cdrs_train, lbls_test, lbls_train, masks_test, masks_train = train_test_split(dataset)

    train_data = ABDataset(cdrs_train, masks_train, lbls_train)
    test_data = ABDataset(cdrs_test, masks_test, lbls_test)

    train_dataloader, test_dataloader = ABloader(train_data, test_data)

    model = ab_seq_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.BCELoss()

    train(model,
          optimizer=optimizer,
          loss_fn=loss_function,
          train_dl=train_dataloader,
          val_dl=test_dataloader)

    example_weight = np.squeeze((lbls_train * 1.7 + 1) * masks_train)
    test_ex_weight = np.squeeze((lbls_test * 1.7 + 1) * masks_test)

    rate_schedule = lambda e: 0.001 if e >= 10 else 0.01

    # history = model.fit([cdrs_train, np.squeeze(masks_train)],
    #                     lbls_train, batch_size=32, epochs=150,
    #                     # Just a trial, not actual evaluation.
    #                     validation_data=([cdrs_test, np.squeeze(masks_test)],
    #                                      lbls_test, test_ex_weight),
    #                     sample_weight=example_weight,
    #                     callbacks=[LearningRateScheduler(rate_schedule),
    #                                EarlyStopping(verbose=1, patience=3)])

    model.save_weights("sabdab.h5")
    probs_test = model.predict([cdrs_test, np.squeeze(masks_test)])

    test_seq_lens = np.sum(np.squeeze(masks_test), axis=1)
    probs_flat = flatten_with_lengths(probs_test, test_seq_lens)
    lbls_flat = flatten_with_lengths(lbls_test, test_seq_lens)

    compute_classifier_metrics([lbls_flat], [probs_flat])


def full_run(dataset, out_weights="weights.h5"):
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset, dataset_cache=cache_file)
    # print(dataset)
    cdrs, lbls, masks = dataset["cdrs"], dataset["lbls"], dataset["masks"]

    sample_weight = np.squeeze((lbls * 1.7 + 1) * masks)
    model = ab_seq_model()
    #
    # rate_schedule = lambda e: 0.001 if e >= 10 else 0.01

    model.fit([cdrs, np.squeeze(masks)],
              lbls, batch_size=32, epochs=18,
              sample_weight=sample_weight,
              callbacks=[LearningRateScheduler(rate_schedule)])

    model.save_weights(out_weights)


def youden_j_stat(fpr, tpr, thresholds):
    j_ordered = sorted(zip(tpr - fpr, thresholds))
    return j_ordered[-1][1]


def compute_classifier_metrics(labels, probs):
    matrices = []
    aucs = []
    mcorrs = []
    jstats = []

    for l, p in zip(labels, probs):
        jstats.append(youden_j_stat(*roc_curve(l, p)))

    jstat_scores = np.array(jstats)
    jstat = np.mean(jstat_scores)
    jstat_err = 2 * np.std(jstat_scores)

    threshold = jstat

    print("Youden's J statistic = {} +/- {}. Using it as threshold.".format(jstat, jstat_err))

    for l, p in zip(labels, probs):
        aucs.append(roc_auc_score(l, p))
        l_pred = (p > threshold).astype(int)
        matrices.append(confusion_matrix(l, l_pred))
        mcorrs.append(matthews_corrcoef(l, l_pred))

    matrices = np.stack(matrices)
    mean_conf = np.mean(matrices, axis=0)
    errs_conf = 2 * np.std(matrices, axis=0)

    tps = matrices[:, 1, 1]
    fns = matrices[:, 1, 0]
    fps = matrices[:, 0, 1]

    recalls = tps / (tps + fns)
    precisions = tps / (tps + fps)

    rec = np.mean(recalls)
    rec_err = 2 * np.std(recalls)
    prec = np.mean(precisions)
    prec_err = 2 * np.std(precisions)

    fscores = 2 * precisions * recalls / (precisions + recalls)
    fsc = np.mean(fscores)
    fsc_err = 2 * np.std(fscores)

    auc_scores = np.array(aucs)
    auc = np.mean(auc_scores)
    auc_err = 2 * np.std(auc_scores)

    mcorr_scores = np.array(mcorrs)
    mcorr = np.mean(mcorr_scores)
    mcorr_err = 2 * np.std(mcorr_scores)

    print("Mean confusion matrix and error")
    print(mean_conf)
    print(errs_conf)

    print("Recall = {} +/- {}".format(rec, rec_err))
    print("Precision = {} +/- {}".format(prec, prec_err))
    print("F-score = {} +/- {}".format(fsc, fsc_err))
    print("ROC AUC = {} +/- {}".format(auc, auc_err))
    print("MCC = {} +/- {}".format(mcorr, mcorr_err))


def flatten_with_lengths(matrix, lengths):
    seqs = []
    for i, example in enumerate(matrix):
        seq = example[:lengths[i]]
        seqs.append(seq)
    return np.concatenate(seqs)


def main():
    single_run("data/dataset.csv")


if __name__ == "__main__":
    main()
