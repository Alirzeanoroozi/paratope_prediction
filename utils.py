import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_hist(lbls_test):
    def count_one(input_list):
        num = 0
        for x in input_list:
            if x == 1:
                num += 1
        return num

    paratopes = pd.Series([count_one(x) for x in lbls_test])
    # creating the dataset
    data = dict(paratopes.value_counts())
    print(data)
    courses = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(courses, values)
    plt.xticks(np.unique(list(data.keys())))
    plt.show()


def show_result(threshold):
    cdrs_test, cdrs_valid, cdrs_train, lbls_test, lbls_valid, lbls_train = train_test_split(dataset)
    for cdr, pred, lbl in zip(cdrs_test, predict(cdrs_test), lbls_test):
        print(*cdr)
        for p in pred[:len(cdr)]:
            print(0 if p < threshold else 1, end=" ")
        print()
        for l in lbl[:len(cdr)]:
            print(0 if l == 0 else 1, end=" ")
        print()
        print("-----------------------------")