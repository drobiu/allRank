import math

import torch
from pytest import approx

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.metrics import ndcg

import numpy as np

def ndcg_wrap(y_pred, y_true, ats=None):
    return ndcg(torch.tensor([y_pred]), torch.tensor([y_true]), ats=ats).numpy()

def ndcg_wrap_multiple_slates(y_pred, y_true, ats=None):
    return ndcg(torch.tensor(y_pred), torch.tensor(y_true), ats=ats).numpy()
def test_ndcg_simple_1():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = ndcg_wrap(y_pred, y_true)

    assert (result == 1.0)


def test_ndcg_simple_2():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]

    result = ndcg_wrap(y_pred, y_true)

    assert (result == 1 / math.log2(3))


def test_ndcg_one_when_no_relevant():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 0.0]

    result = ndcg_wrap(y_pred, y_true)

    assert (result == 1.0)


def test_ndcg_for_multiple_ats():
    y_pred = [0.5, 0.2, 0.1]
    y_true = [1.0, 0.0, 1.0]

    result = ndcg_wrap(y_pred, y_true, ats=[1, 2])

    ndcg_one_relevant_on_top = 1.0 / (1.0 + 1 / math.log2(3))
    expected = [1.0, ndcg_one_relevant_on_top]

    batch_0 = 0
    assert result[batch_0] == approx(expected)


def test_ndcg_with_padded_input():
    y_pred = [0.5, 0.2, 1.0]
    y_true = [1.0, 0.0, PADDED_Y_VALUE]

    result = ndcg_wrap(y_pred, y_true)

    assert result == 1.0


def test_ndcg_with_padded_input_2():
    y_pred = [0.5, 0.2, 1.0]
    y_true = [0.0, 1.0, PADDED_Y_VALUE]

    result = ndcg_wrap(y_pred, y_true)

    assert result == 1 / math.log2(3)

def test_baseline_ndcg():
    predictions = [[0.9, 0.85, 0.71, 0.63],
                   [0.87, 0.76, 0.64, 0.26],
                   [0.7, 0.65, 0.32, 0.1]]

    ground_truth = [[2.0, 0.0, 2.0, 1.0],
                    [0.0, 2.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0]]

    result = ndcg_wrap_multiple_slates(predictions, ground_truth, ats=[4])

    np.testing.assert_almost_equal(result.mean(), 0.857, decimal=3)
