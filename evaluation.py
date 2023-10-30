import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve, average_precision_score


def youden_j_stat(fpr, tpr, thresholds):
    j_ordered = sorted(zip(tpr - fpr, thresholds))
    return 1. if j_ordered[-1][1] > 1 else j_ordered[-1][1]


def compute_classifier_metrics(probs, labels, lengths, cv=None):
    probs = probs.detach()

    matrices = []
    aucs = []
    aupr = []
    mcorrs = []
    jstats = [youden_j_stat(*roc_curve(lbl[:l], p[:l])) for lbl, p, l in zip(labels, probs, lengths)]

    jstat_scores = np.array(jstats)
    jstat = np.mean(jstat_scores)
    # jstat_err = 2 * np.std(jstat_scores)

    threshold = jstat

    for lbl, p, l in zip(labels, probs, lengths):
        print(*[1 if x == 1.0 else 0 for x in lbl[:l].tolist()])
        # if sum(lbl) != 0.0:
        aucs.append(roc_auc_score(lbl[:l], p[:l]))
        aupr.append(average_precision_score(lbl[:l], p[:l], pos_label=1.0))
        l_pred = (p[:l] > threshold).numpy().astype(int)
        print(*l_pred.tolist())
        print(confusion_matrix(lbl[:l], l_pred))
        matrices.append(confusion_matrix(lbl[:l], l_pred, labels=[0, 1]))
        mcorrs.append(matthews_corrcoef(lbl[:l], l_pred))

    matrices = np.stack(matrices)
    mean_conf = np.mean(matrices, axis=0)
    # errs_conf = 2 * np.std(matrices, axis=0)

    tps = matrices[:, 1, 1]
    fns = matrices[:, 1, 0]
    fps = matrices[:, 0, 1]

    recalls = tps / (tps + fns)
    precisions = tps / (tps + fps)

    rec = np.mean(recalls)
    # rec_err = 2 * np.std(recalls)

    prec = np.mean(precisions)
    # prec_err = 2 * np.std(precisions)

    fscores = 2 * precisions * recalls / (precisions + recalls)
    fsc = np.mean(fscores)
    # fsc_err = 2 * np.std(fscores)

    auc_scores = np.array(aucs)
    auc = np.mean(auc_scores)
    # auc_err = 2 * np.std(auc_scores)

    aupr_scores = np.array(aupr)
    aupr = np.mean(aupr_scores)
    # aupr_err = 2 * np.std(aupr_scores)

    mcorr_scores = np.array(mcorrs)
    mcorr = np.mean(mcorr_scores)
    # mcorr_err = 2 * np.std(mcorr_scores)

    if cv is None:
        f = open("results/model.txt", "w")
    else:
        f = open("results/cv_{}.txt".format(cv), "w")

    # f.write("Youden's J statistic = {} +/- {}. Using it as threshold.\n".format(jstat, jstat_err))
    f.write("Youden's J statistic = {}. Using it as threshold.\n".format(jstat))

    f.write("Mean confusion matrix and error\n")
    f.write(str(mean_conf) + "\n")
    # f.write(str(errs_conf) + "\n")

    # f.write("Recall = {} +/- {}\n".format(rec, rec_err))
    f.write("Recall = {}\n".format(rec))
    # f.write("Precision = {} +/- {}\n".format(prec, prec_err))
    f.write("Precision = {}\n".format(prec))
    # f.write("F-score = {} +/- {}\n".format(fsc, fsc_err))
    f.write("F-score = {}\n".format(fsc))
    # f.write("ROC AUC = {} +/- {}\n".format(auc, auc_err))
    f.write("ROC AUC = {}\n".format(auc))
    # f.write("pr = {} +/- {}\n".format(aupr, aupr_err))
    f.write("pr = {}\n".format(aupr))
    # f.write("MCC = {} +/- {}\n".format(mcorr, mcorr_err))
    f.write("MCC = {}\n".format(mcorr))

    f.close()
    return threshold
