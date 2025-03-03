from allrank.models.metrics import recall
import torch
import numpy as np


def recall_wrap(y_pred, y_true, ats, min_relevance = 1):
    return recall(torch.tensor([y_pred]), torch.tensor([y_true]), ats=ats, min_relevance=min_relevance).numpy()


def recall_wrap_multiple_slates(y_pred, y_true, ats, min_relevance = 1):
    return recall(torch.tensor(y_pred), torch.tensor(y_true), ats=ats, min_relevance=min_relevance).numpy()


def test_recall_simple_1():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    ats = [1]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 1.0

    assert result == expected


def test_recall_simple_2():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]
    ats = [1]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 0.0

    assert result == expected


def test_recall_simple_3():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]
    ats = [2]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 1.0

    assert result == expected


def test_recall_simple_4():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [2]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 0

    assert result == expected


def test_recall_simple_5():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [3]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 0.5

    assert result == expected


def test_recall_simple_6():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [4]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 1.0

    assert result == expected


def test_recall_simple_7():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [5]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 1.0

    assert result == expected


def test_recall_simple_8():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [6]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 1.0

    assert result == expected


def test_recall_complex_1():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [1, 2, 3, 4, 5, 6]

    result = recall_wrap(y_pred, y_true, ats)
    expected = [0, 0, 0.5, 1, 1, 1]

    assert (result == expected).all()


def test_recall_query_1():
    y_pred = [[0.5, 0.2, 0.7, 0.3], [0.5, 0.2, 0.7, 0.3]]
    y_true = [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    ats = [1, 2, 3, 4, 5, 6]

    result = np.array(recall_wrap_multiple_slates(y_pred, y_true, ats))
    expected = np.array([[0, 0, 0.5, 1, 1, 1], [0, 0, 0.5, 1, 1, 1]])

    assert (result == expected).all()


def test_recall_query_2():
    y_pred = [[0.5, 0.2, 0.7, 0.3], [0.5, 0.2, 0.7, 0.3]]
    y_true = [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    ats = [1, 2, 3, 4, 5, 6]

    result = recall_wrap_multiple_slates(y_pred, y_true, ats)
    expected = np.array([[0., 0., 0.5, 1., 1., 1.],
                         [0.25, 0.5, 0.75, 1., 1., 1.]])

    assert (result == expected).all()

def test_recall_baseline():
    predictions = [[0.9, 0.85, 0.71, 0.63],
                   [0.87, 0.76, 0.64, 0.26],
                   [0.7, 0.65, 0.32, 0.1]]

    ground_truth = [[2.0, 0.0, 2.0, 1.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0]]

    result = recall_wrap_multiple_slates(predictions, ground_truth, ats=[1, 4], min_relevance=2)

    np.testing.assert_almost_equal(result[:,0].mean(), 0.166, decimal=3)
    np.testing.assert_almost_equal(result[:,1].mean(), 0.666, decimal=3)