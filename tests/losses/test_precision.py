import torch
from allrank.models.metrics import precision
import numpy as np
def precision_wrap(y_pred, y_true, ats):
    return precision(torch.tensor([y_pred]), torch.tensor([y_true]), ats)

def precision_wrap_multiple_slates(y_pred, y_true, ats, cutoff = 1):
    return precision(torch.tensor(y_pred), torch.tensor(y_true), ats, cutoff=cutoff)
def test_precision_simple_1():
    y_pred = [0.5, 0.2]
    y_true = [2.0, 0.0]
    ats = [1]

    result = precision_wrap(y_pred, y_true, ats)
    expected = 1.0

    assert result == expected

def test_precision_simple_2():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 2.0]
    ats = [1]

    result = precision_wrap(y_pred, y_true, ats)
    expected = 0.0

    assert result == expected

def test_precision_simple_3():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 2.0]
    ats = [2]

    result = precision_wrap(y_pred, y_true, ats)
    expected = 0.5

    assert result == expected

def test_baseline_precision():
    predictions = [[0.9, 0.85, 0.71, 0.63],
                   [0.87, 0.76, 0.64, 0.26],
                   [0.7, 0.65, 0.32, 0.1]]

    ground_truth = [[2.0, 0.0, 2.0, 1.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0]]

    result = precision_wrap_multiple_slates(predictions, ground_truth, ats=[1, 4], cutoff=2)

    np.testing.assert_almost_equal(result[:,0].mean(), 0.333, decimal=3)
    np.testing.assert_almost_equal(result[:,1].mean(), 0.250, decimal=3)


if __name__ == '__main__':
    test_precision_simple_1()
    test_precision_simple_2()
    test_precision_simple_3()