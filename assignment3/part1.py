import numpy as np
import pandas as pd

import sklearn.svm as svm
import sklearn.naive_bayes as nb
import sklearn.ensemble as ensemble


def read_data(filename):
    data = pd.read_csv(filename, header=None, sep='\t').as_matrix()
    np.random.shuffle(data)
    data_y = data[:, 0] != 0
    data_x = data[:, 1:]
    return (data_x, data_y)


def partition(datax, datay, prorportion=[8, 1, 1]):
    """
    partition the data into different perportion,
    [8, 1, 1] will part the data into 80%, 10%, 10%
    into train, validation and test sets
    """
    prorportion = np.array(prorportion[0:3])
    (tr, vl, ts) = (prorportion / np.sum(prorportion) *
                    datay.size).astype(np.int)
    train_x = datax[:tr, :]
    train_y = datay[:tr]
    validation_x = datax[tr:tr + vl, :]
    validation_y = datay[tr:tr + vl]
    test_x = datax[tr + vl:, :]
    test_y = datay[tr + vl:]
    return (train_x, train_y, validation_x, validation_y, test_x, test_y)


def test_case(filename, trainer):
    datax, datay = read_data(filename)
    trainx, trainy, validationx, validationy, testx, testy = partition(
        datax, datay)
    trainer.fit(trainx, trainy)
    return trainer.score(testx, testy)


def run_part1():
    acc1 = test_case('./pubfig_dev_50000_pairs.txt', nb.GaussianNB())
    acc2 = test_case('./pubfig_dev_50000_pairs.txt', svm.SVC(kernel="linear"))
    acc3 = test_case('./pubfig_dev_50000_pairs.txt', ensemble.RandomForestClassifier())
    print("the accuracy of Naive Bayes with Gaussian Distribution is %f"%acc1)
    print("the accuracy of SVM is %f"%acc2)
    print("the accuracy of Random Forest is %f"%acc3)

if __name__ == "__main__":
    run_part1()

