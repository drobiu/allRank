import torch
import numpy as np

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.metrics import avgrank


def avgrank_wrap_single_slate(y_pred, y_true, ats=[3]):
    return avgrank_wrap_multiple_slates([y_pred], [y_true], ats)


def avgrank_wrap_multiple_slates(y_pred, y_true, ats=[3]):
    return torch.mean(avgrank(torch.tensor(y_pred), torch.tensor(y_true), ats=ats)).numpy()


def compute_mean(result):
    return np.mean(result)


def single_slate_and_ats(result):
    return result[0][0]


def test_avgrank_simple_1():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = avgrank_wrap_single_slate(y_pred, y_true)
    assert compute_mean(result) == 1.5, (compute_mean(result), result)


def test_avgrank_simple_no_ats():
    y_pred = [0.5, 0.2]
    y_true = [1.0, 0.0]

    result = avgrank_wrap_single_slate(y_pred, y_true, ats=None)

    assert compute_mean(result) == 1.5


def test_avgrank_simple_2():
    y_pred = [0.5, 0.2, 0.7, 0.3]
    y_true = [0.0, 1.0, 0.0, 1.0]

    result = avgrank_wrap_single_slate(y_pred, y_true)

    assert compute_mean(result) == 4.0/3.0

if __name__ == "__main__":
    print('here')
    test_avgrank_simple_1()
    test_avgrank_simple_no_ats()
    test_avgrank_simple_2()

# def test_avgrank_multiple_slates():
#     y_pred_1 = [0.2, 0.5]
#     y_pred_05 = [0.5, 0.2]
#     y_true = [0.0, 1.0]

#     result = avgrank_wrap_multiple_slates([y_pred_1, y_pred_05], [y_true, y_true])

#     assert result[0][0] == 1.0
#     assert result[1][0] == 0.5


# def test_avgrank_multiple_ats():
#     y_pred = [0.5, 0.2]
#     y_true = [0.0, 1.0]

#     result = avgrank_wrap_single_slate(y_pred, y_true, ats=[1, 2])

#     assert result[0][0] == 0.0
#     assert result[0][1] == 0.5


# def test_avgrank_multiple_slates_multiple_ats():
#     y_pred_1 = [0.2, 0.5]
#     y_pred_05 = [0.5, 0.2]
#     y_true = [0.0, 1.0]

#     result = avgrank_wrap_multiple_slates([y_pred_1, y_pred_05], [y_true, y_true], ats=[1, 2])

#     assert result[0][0] == 1.0
#     assert result[0][1] == 1.0
#     assert result[1][0] == 0.0
#     assert result[1][1] == 0.5


# def test_avgrank_zero_when_no_relevant():
#     y_pred = [0.5, 0.2]
#     y_true = [0.0, 0.0]

#     result = avgrank_wrap_single_slate(y_pred, y_true)

#     assert single_slate_and_ats(result) == 0.0


# def test_avgrank_with_padded_input():
#     y_pred = [0.5, 0.2, 1.0]
#     y_true = [1.0, 0.0, PADDED_Y_VALUE]

#     result = avgrank_wrap_single_slate(y_pred, y_true)

#     assert single_slate_and_ats(result) == 1.0


# def test_avgrank_with_padded_input_2():
#     y_pred = [0.5, 0.2, 1.0]
#     y_true = [0.0, 1.0, PADDED_Y_VALUE]

#     result = avgrank_wrap_single_slate(y_pred, y_true)

#     assert single_slate_and_ats(result) == 0.5
