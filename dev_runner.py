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

    torch.save(model.state_dict(), "precomputed/sabdab.pth")

    probs_test = predict(test_data)

    test_seq_lens = np.sum(np.squeeze(masks_test), axis=1)
    probs_flat = flatten_with_lengths(probs_test, test_seq_lens)
    lbls_flat = flatten_with_lengths(lbls_test, test_seq_lens)

    compute_classifier_metrics([lbls_flat], [probs_flat])


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


def run_cv(dataset, output_folder, num_iters=10):
    cache_file = dataset.split("/")[-1] + ".p"
    dataset = open_dataset(dataset, dataset_cache=cache_file)

    makedirs(output_folder + "/weights", exist_ok=True)
    for i in range(num_iters):
        print("Crossvalidation run", i+1)
        output_file = "{}/run-{}.p".format(output_folder, i)
        weights_template = output_folder + "/weights/run-" + str(i) + "-fold-{}.h5"
        kfold_cv_eval(ab_seq_model, dataset, output_file, weights_template, seed=i)


def process_cv_results(cv_result_folder, abip_result_folder, cv_num_iters=10):
    for i, loop in enumerate(["H1", "H2", "H3", "L1", "L2", "L3"]):
        print("Classifier metrics for loop type", loop)
        labels, probs = open_crossval_results(cv_result_folder, cv_num_iters, i)
        compute_classifier_metrics(labels, probs)

    # Plot PR curves
    print("Plotting PR curves")
    labels, probs = open_crossval_results(cv_result_folder, cv_num_iters)
    labels_abip, probs_abip = open_crossval_results(abip_result_folder, 10)

    fig = plot_pr_curve(labels, probs, colours=("#0072CF", "#68ACE5"),
                        label="Parapred")
    fig = plot_pr_curve(labels_abip, probs_abip, colours=("#D6083B", "#EB99A9"),
                        label="Parapred using ABiP data", plot_fig=fig)
    fig = plot_abip_pr(fig)
    fig.savefig("pr.eps")

    # Computing overall classifier metrics
    print("Computing classifier metrics")
    compute_classifier_metrics(labels, probs)


def plot_dataset_fraction_results(results):
    print("Plotting PR curves")
    colours = [("#0072CF", "#68ACE5"),
               ("#EA7125", "#F3BD48"),
               ("#55A51C", "#AAB300"),
               ("#D6083B", "#EB99A9")]

    fig = None
    for i, (file, descr) in enumerate(results):
        labels, probs = open_crossval_results(file, 10)
        fig = plot_pr_curve(labels, probs, colours=colours[i], plot_fig=fig, label="Parapred ({})".format(descr))

    fig.savefig("fractions-pr.eps")


def show_binding_profiles(dataset, run):
    labels, probs = open_crossval_results(run, flatten_by_lengths=False)
    labels = labels[0]  # Labels are constant, any of the 10 runs would do
    probs = np.stack(probs).mean(axis=0)  # Mean binding probability across runs

    contact = binding_profile(dataset, labels)
    print("Contact per-residue binding profile:")
    total = sum(list(contact.values()))
    contact = {k: v / total for k, v in contact.items()}
    print(contact)

    parapred = binding_profile(dataset, probs)
    print("Model's predictions' per-residue binding profile:")
    total = sum(list(parapred.values()))
    parapred = {k: v / total for k, v in parapred.items()}
    print(parapred)

    plot_binding_profiles(contact, parapred)


if __name__ == "__main__":
    single_run("data/dataset.csv")
