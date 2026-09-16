import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype = int)
    y_pred = np.asarray(y_pred, dtype = float)

    N = y_true.shape[0]

    indices = np.arange(N)

    correct_preds = y_pred[indices, y_true]

    return float(-np.mean(np.log(correct_preds)))

    