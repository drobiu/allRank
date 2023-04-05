import torch
import numpy as np
from allrank.models.metrics import map
def map_wrap(y_pred, y_true, ats, cutoff=2):
    return map(torch.tensor([y_pred]), torch.tensor([y_true]), ats, cutoff)

def test_map_simple_0():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    ats = [1]

    result = map_wrap(y_pred, y_true, ats)
    expected = [0]

    assert result == expected

def test_map_simple_1():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]
    ats = [1]

    result = map(torch.tensor([y_pred]), torch.tensor([y_true]), ats, cutoff=1)
    expected = [1.0]

    assert result == expected, result

def test_map_simple_2():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 2.0]
    ats = [1, 2]

    result = map(torch.tensor([y_pred]), torch.tensor([y_true]), ats)
    expected = np.array([[0.0, 0.5]])

    assert np.equal(result, expected).all()

def test_map_two_queries():
    y_pred_1_5 = [0.5, 0.7, 0.9]
    y_pred_3 = [0.5, 0.7, 0.2]
    y_true = [0, 2, 2]
    result = map(torch.tensor([y_pred_1_5, y_pred_3]), torch.tensor([y_true, y_true]), ats=[1, 2, 3])
    expected = np.array([[1, 1, 1],
                [1, 1, 5/6]])

    assert np.allclose(result, expected)

def test_long_query():
    y_pred = [0.9, 0.71, 0.63, 0.36, 0.85, 0.47, 0.24, 0.16]
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]

    ats = [3, 4, 7]

    result = map(torch.tensor([y_pred]), torch.tensor([y_true]), ats, cutoff=1)
    expected = np.array([[0.83, 0.81, 0.77]])

    assert np.allclose(result, expected, atol=0.1)

if __name__ == '__main__':
    test_map_simple_0()
    test_map_simple_1()
    test_map_simple_2()
    test_map_two_queries()
    test_long_query()