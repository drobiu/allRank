from allrank.models.metrics import recall
import torch

def recall_wrap(y_pred, y_true, ats):
    return recall(torch.tensor([y_pred]), torch.tensor([y_true]), ats=ats)


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
    expected = 0.5

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
    expected = 1/3

    assert result == expected

def test_recall_simple_6():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [4]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 0.5

    assert result == expected

def test_recall_simple_7():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]
    ats = [5]

    result = recall_wrap(y_pred, y_true, ats)
    expected = 0.5

    assert result == expected
