import torch
from allrank.models.metrics import precision
def precision_wrap(y_pred, y_true, ats):
    return precision(torch.tensor([y_pred]), torch.tensor([y_true]), ats)

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

if __name__ == '__main__':
    test_precision_simple_1()
    test_precision_simple_2()
    test_precision_simple_3()