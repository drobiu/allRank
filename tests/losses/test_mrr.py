import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.metrics import mrr


def mrr_wrap_single_slate(y_pred, y_true, ats=None, min_relevance=1):
    return mrr_wrap_multiple_slates([y_pred], [y_true], ats, min_relevance=min_relevance)


def mrr_wrap_multiple_slates(y_pred, y_true, ats=None, min_relevance=1):
    return mrr(torch.tensor(y_pred), torch.tensor(y_true), ats=ats, min_relevance=min_relevance).numpy()


def single_slate_and_ats(result):
    return result[0][0]


def test_mrr_simple_1():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = mrr_wrap_single_slate(y_pred, y_true)

    assert single_slate_and_ats(result) == 1.0


def test_mrr_simple_no_ats():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = mrr_wrap_single_slate(y_pred, y_true, ats=None)

    assert single_slate_and_ats(result) == 1.0


def test_mrr_simple_2():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]

    result = mrr_wrap_single_slate(y_pred, y_true)

    assert single_slate_and_ats(result) == 0.5


def test_mrr_multiple_slates():
    y_pred_1 = [0.2, 0.5]
    y_pred_05 = [0.5, 0.2]
    y_true = [0.0, 1.0]

    result = mrr_wrap_multiple_slates([y_pred_1, y_pred_05], [y_true, y_true])

    assert result[0][0] == 1.0
    assert result[1][0] == 0.5


def test_mrr_multiple_ats():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 1.0]

    result = mrr_wrap_single_slate(y_pred, y_true, ats=[1, 2])

    assert result[0][0] == 0.0
    assert result[0][1] == 0.5


def test_mrr_multiple_slates_multiple_ats():
    y_pred_1 = [0.2, 0.5]
    y_pred_05 = [0.5, 0.2]
    y_true = [0.0, 1.0]

    result = mrr_wrap_multiple_slates([y_pred_1, y_pred_05], [y_true, y_true], ats=[1, 2])

    assert result[0][0] == 1.0
    assert result[0][1] == 1.0
    assert result[1][0] == 0.0
    assert result[1][1] == 0.5


def test_mrr_zero_when_no_relevant():
    y_pred = [0.5, 0.2]
    y_true = [0.0, 0.0]

    result = mrr_wrap_single_slate(y_pred, y_true)

    assert single_slate_and_ats(result) == 0.0


def test_mrr_with_padded_input():
    y_pred = [0.5, 0.2, 1.0]
    y_true = [1.0, 0.0, PADDED_Y_VALUE]

    result = mrr_wrap_single_slate(y_pred, y_true)

    assert single_slate_and_ats(result) == 1.0


def test_mrr_with_padded_input_2():
    y_pred = [0.5, 0.2, 1.0]
    y_true = [0.0, 1.0, PADDED_Y_VALUE]

    result = mrr_wrap_single_slate(y_pred, y_true)

    assert single_slate_and_ats(result) == 0.5

def test_baseline_mrr():
    predictions = [[0.9, 0.85, 0.71, 0.63],
                            [0.87, 0.76, 0.64, 0.26],
                            [0.7, 0.65, 0.32, 0.1]]

    ground_truth = [[2.0, 0.0, 2.0, 1.0],
                             [0.0, 2.0, 1.0, 0.0],
                             [1.0, 1.0, 0.0, 0.0]]

    result = mrr_wrap_multiple_slates(predictions, ground_truth, min_relevance=2)

    assert result.mean()  == 0.5