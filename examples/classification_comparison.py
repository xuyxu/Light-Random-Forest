import sys
import time
import pickle
import numpy as np

from lightrf import RandomForestClassifier as light_forest
from sklearn.ensemble import RandomForestClassifier as sklearn_forest

from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file


def load_covtype():
    """
    Dataset available at:
        www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#covtype
    """
    # MODIFY THE PATH TO THE DATASET
    data = load_svmlight_file('../../Dataset/LIBSVM/covtype')
    X = data[0].toarray()
    y = data[1]

    X_train, X_test = X[:406708, :], X[406708:, :]
    y_train, y_test = y[:406708]-1, y[406708:]-1  # [1, 7] -> [0, 6]

    return X_train, y_train.astype(np.int), X_test, y_test.astype(np.int)


if __name__ == "__main__":

    n_estimators = 500
    n_jobs = -1
    random_state = 0

    X_train, y_train, X_test, y_test = load_covtype()

    # Bin the data is necessary because lightrf only handles data of the
    # type `np.uint8`.
    binner = _BinMapper(random_state=random_state)
    X_train = binner.fit_transform(X_train)
    X_test = binner.transform(X_test)

    # Main body
    model = light_forest(n_estimators=n_estimators,
                         n_jobs=n_jobs,
                         random_state=random_state)

    tic = time.time()
    model.fit(X_train, y_train)
    toc = time.time()
    training_time = toc - tic

    handler = pickle.dumps(model)
    size = sys.getsizeof(handler)

    tic = time.time()
    y_pred = model.predict(X_test)
    toc = time.time()
    evaluating_time = toc - tic

    print("Light Testing Accuracy: {:.3f} %".format(
        accuracy_score(y_test, y_pred) * 100.))
    print("Model Size: {:.3f} MB".format(size / (1024 * 1024)))
    print("Training Time: {:.3f} s".format(training_time))
    print("Evaluating Time: {:.3f} s\n".format(evaluating_time))

    model = sklearn_forest(n_estimators=n_estimators,
                           n_jobs=n_jobs,
                           random_state=random_state)

    tic = time.time()
    model.fit(X_train, y_train)
    toc = time.time()
    training_time = toc - tic

    handler = pickle.dumps(model)
    size = sys.getsizeof(handler)

    tic = time.time()
    y_pred = model.predict(X_test)
    toc = time.time()
    evaluating_time = toc - tic

    print("Sklearn Testing Accuracy: {:.3f} %".format(
        accuracy_score(y_test, y_pred) * 100.))
    print("Model Size: {:.3f} MB".format(size / (1024 * 1024)))
    print("Training Time: {:.3f} s".format(training_time))
    print("Evaluating Time: {:.3f} s\n".format(evaluating_time))
