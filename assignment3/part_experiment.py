import numpy as np
from part1 import read_data, read_datax, read_datay
from part2 import split_attrs, ground_truth_classifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
try:
    from spark_sklearn import GridSearchCV
except:
    from sklearn.grid_search import GridSearchCV
from write_result import write_result


def minus_app(x):
    x1, x2 = split_attrs(x)
    return x1 - x2


def minus_abs_app(x):
    x1, x2 = split_attrs(x)
    return np.abs(x1 - x2)


def log_app(x):
    x1, x2 = split_attrs(x)
    return np.abs(np.log((x1 + 1) / (x2 + 1)))


def devide_app(x):
    x1, x2 = split_attrs(x)
    return x1 / x2


def plus_app(x):
    x1, x2 = split_attrs(x)
    return x1 + x2

if __name__ == '__main__':
    run_part1 = False
    run_part2 = False
    run_part3 = True
    eval_set = read_datax('./data/pubfig_kaggle_eval.txt')
    if run_part1:
        print("------- part1 code -----------")
        datax, datay = read_data('./data/pubfig_dev_50000_pairs.txt')
        scaler = StandardScaler().fit(datax)
        datax = scaler.transform(datax)

        testx = read_datax('./data/pubfig_kaggle_1.txt')
        testy = read_datay('./data/pubfig_kaggle_1_solution.txt')
        testx = scaler.transform(testx)

        print("first train with the raw data")
        trainer = LinearSVC(verbose=True)
        trainer.fit(datax, datay)
        score = trainer.score(testx, testy)
        write_result('svm_first.csv', trainer.predict(
            scaler.transform(eval_set)))
        print("the score for raw data is %f" % score)

        print("second train with difference")
        trainer = LinearSVC(verbose=True)
        trainer.fit(minus_app(datax), datay)
        score = trainer.score(minus_app(testx), testy)
        write_result('svm_second.csv', trainer.predict(
            minus_app(scaler.transform(eval_set))))
        print("the score for difference is %f" % score)

        print("third train with ratio")
        trainer = LinearSVC(verbose=True)
        trainer.fit(devide_app(datax), datay)
        score = trainer.score(devide_app(testx), testy)
        write_result('svm_third.csv', trainer.predict(
            scaler.transform(devide_app(eval_set))))
        print("the score for ratio is %f" % score)

        print("forth train with sum")
        trainer = LinearSVC(verbose=True)
        trainer.fit(plus_app(datax), datay)
        score = trainer.score(plus_app(testx), testy)
        write_result('svm_forth.csv', trainer.predict(
            scaler.transform(plus_app(eval_set))))
        print("the score for sum is %f" % score)

        print("NB first train with the raw data")
        trainer = GaussianNB()
        trainer.fit(datax, datay)
        score = trainer.score(testx, testy)
        write_result('nb_first.csv', trainer.predict(
            scaler.transform(eval_set)))
        print("the score for raw data is %f" % score)

        print("NB second train with difference")
        trainer = GaussianNB()
        trainer.fit(minus_app(datax), datay)
        score = trainer.score(minus_app(testx), testy)
        write_result('nb_second.csv', trainer.predict(
            scaler.transform(minus_app(eval_set))))
        print("the score for difference is %f" % score)

        print("NB two point fifth train with abs difference")
        trainer = GaussianNB()
        trainer.fit(minus_abs_app(datax), datay)
        score = trainer.score(minus_abs_app(testx), testy)
        write_result('nb_two_point_five.csv',
                     trainer.predict(scaler.transform(minus_abs_app(eval_set))))
        print("the score for abs difference is %f" % score)

        print("NB third train with ratio")
        trainer = GaussianNB()
        trainer.fit(devide_app(datax), datay)
        score = trainer.score(devide_app(testx), testy)
        write_result('nb_third.csv', trainer.predict(
            scaler.transform(devide_app(eval_set))))
        print("the score for ratio is %f" % score)

        print("NB forth train with sum")
        trainer = GaussianNB()
        trainer.fit(plus_app(datax), datay)
        score = trainer.score(plus_app(testx), testy)
        write_result('nb_forth.csv', trainer.predict(
            scaler.transform(plus_app(eval_set))))
        print("the score for sum is %f" % score)

        print("RF first train with the raw data")
        trainer = RandomForestClassifier()
        trainer.fit(datax, datay)
        score = trainer.score(testx, testy)
        write_result('rf_first.csv', trainer.predict(
            scaler.transform(eval_set)))
        print("the score for raw data is %f" % score)

        print("RF second train with difference")
        trainer = RandomForestClassifier()
        trainer.fit(minus_app(datax), datay)
        score = trainer.score(minus_app(testx), testy)
        write_result('rf_second.csv', trainer.predict(
            scaler.transform(minus_app(eval_set))))
        print("the score for difference is %f" % score)

        print("RF two point fifth train with abs difference")
        trainer = RandomForestClassifier()
        trainer.fit(minus_abs_app(datax), datay)
        score = trainer.score(minus_abs_app(testx), testy)
        write_result('rf_two_point_five.csv',
                     trainer.predict(scaler.transform(minus_abs_app(eval_set))))
        print("the score for abs difference is %f" % score)

        print("RF third train with ratio")
        trainer = RandomForestClassifier()
        trainer.fit(devide_app(datax), datay)
        score = trainer.score(devide_app(testx), testy)
        write_result('rf_third.csv', trainer.predict(
            scaler.transform(devide_app(eval_set))))
        print("the score for ratio is %f" % score)

        print("RF forth train with sum")
        trainer = GaussianNB()
        trainer.fit(plus_app(datax), datay)
        score = trainer.score(plus_app(testx), testy)
        write_result('rf_forth.csv', trainer.predict(
            scaler.transform(plus_app(eval_set))))
        print("the score for sum is %f" % score)

    if run_part2:

        print("=========== part2 code ================")
        trainer = ground_truth_classifier('./data/pubfig_attributes.txt')
        testx = read_datax('./data/pubfig_kaggle_2.txt')
        testy = read_datay('./data/pubfig_kaggle_2_solution.txt')
        score = trainer.score(testx, testy)
        write_result('part2.csv', trainer.predict(eval_set))
        print("the ANN score is %f" % score)

    if run_part3:

        print("========== part3 code ================")
        datax, datay = read_data('./data/pubfig_dev_50000_pairs.txt')
        scaler = StandardScaler().fit(datax)
        datax = scaler.transform(datax)

        testx = read_datax('./data/pubfig_kaggle_eval.txt')
        testy = read_datay('./part2.csv')
        testx = scaler.transform(testx)
        trainer = SVC(kernel='rbf', verbose=True, gamma=0.01)

        trainer.fit(minus_abs_app(datax), datay)
        # params = {"C": [0.1, 1, 10], "gamma": [0.1, 0.01, 0.001]}
        # grid_search = GridSearchCV(trainer, params)
        # grid_search.fit(datax, datay)
        score = trainer.score(minus_abs_app(testx), testy)
        print("the score is %f" % score)
        write_result('part3.csv', trainer.predict(scaler.transform(eval_set)))
