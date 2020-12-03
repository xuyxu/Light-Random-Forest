"""
This testing script aims to ensure that the behavior of the reduced
version of RandomForestClassifier and ExtraTreesClassifier is exactly the same
as those in Scikit-Learn.
"""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from lightrf import RandomForestClassifier
from lightrf import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier as sklearn_RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier as sklearn_ExtraTreesClassifier

from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer    
)


n_estimators = 42
test_size = 42 * 0.01
random_state = 42


@pytest.mark.parametrize('load_func',
                         [load_iris,
                          load_digits,
                          load_wine,
                          load_breast_cancer])
def test_rf_proba(load_func):
    X, y = load_func(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)
    
    # Ours
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_proba = model.predict_proba(X_test_binned)
    
    # Sklearn
    model = sklearn_RandomForestClassifier(n_estimators=n_estimators,
                                           random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_proba = model.predict_proba(X_test_binned)

    assert_array_equal(actual_proba, expected_proba)


@pytest.mark.parametrize('load_func',
                         [load_iris,
                          load_digits,
                          load_wine,
                          load_breast_cancer])
def test_crf_proba(load_func):
    X, y = load_func(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)
    X_test_binned = binner.transform(X_test)

    # Ours
    model = ExtraTreesClassifier(n_estimators=n_estimators,
                                 random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_proba = model.predict_proba(X_test_binned)

    # Sklearn
    model = sklearn_ExtraTreesClassifier(n_estimators=n_estimators,
                                         random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_proba = model.predict_proba(X_test_binned)

    assert_array_equal(actual_proba, expected_proba)


@pytest.mark.parametrize('load_func',
                         [load_iris,
                          load_digits,
                          load_wine,
                          load_breast_cancer])
def test_rf_oob_normal(load_func):

    # We need more estimators when testing the oob_decision_function_ under
    # the normal case, i.e., each sample is unsampled by at least one decision
    # tree in the forest.
    oob_n_estimators = 100

    X_train, y_train = load_func(return_X_y=True)
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)

    # Ours
    model = RandomForestClassifier(n_estimators=oob_n_estimators,
                                   bootstrap=True,
                                   oob_score=True,
                                   random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_proba = model.oob_decision_function_

    # Sklearn
    model = sklearn_RandomForestClassifier(n_estimators=oob_n_estimators,
                                           bootstrap=True,
                                           oob_score=True,
                                           random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_proba = model.oob_decision_function_

    assert_array_equal(actual_proba, expected_proba)


@pytest.mark.parametrize('load_func',
                         [load_iris,
                          load_digits,
                          load_wine,
                          load_breast_cancer])
def test_crf_oob_normal(load_func):
    oob_n_estimators = 100

    X_train, y_train = load_func(return_X_y=True)
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)

    # Ours
    model = ExtraTreesClassifier(n_estimators=oob_n_estimators,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=random_state)
    model.fit(X_train_binned, y_train)
    actual_proba = model.oob_decision_function_

    # Sklearn
    model = sklearn_ExtraTreesClassifier(n_estimators=oob_n_estimators,
                                         bootstrap=True,
                                         oob_score=True,
                                         random_state=random_state)
    model.fit(X_train_binned, y_train)
    expected_proba = model.oob_decision_function_

    assert_array_equal(actual_proba, expected_proba)


@pytest.mark.parametrize('load_func',
                         [load_iris,
                          load_digits,
                          load_wine,
                          load_breast_cancer])
def test_oob_warning(load_func):

    # We need less estimators when testing the oob_decision_function_ under
    # the case that no sample is unsampled by any decision tree in the forest.
    oob_n_estimators = 5
    
    X_train, y_train = load_func(return_X_y=True)
    binner = _BinMapper(random_state=random_state)
    X_train_binned = binner.fit_transform(X_train)

    # Random Forest
    model = RandomForestClassifier(n_estimators=oob_n_estimators,
                                   bootstrap=True,
                                   oob_score=True,
                                   random_state=random_state)
    with pytest.warns(None) as record:
        model.fit(X_train_binned, y_train)
    assert len(record) == 2
    assert "Some inputs do not have OOB scores" in str(record[0].message)
    assert "invalid value encountered in true_divide" in str(record[1].message)

    # Completely Random Forest
    model = ExtraTreesClassifier(n_estimators=oob_n_estimators,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=random_state)
    with pytest.warns(None) as record:
        model.fit(X_train_binned, y_train)
    assert len(record) == 2
    assert "Some inputs do not have OOB scores" in str(record[0].message)
    assert "invalid value encountered in true_divide" in str(record[1].message)
