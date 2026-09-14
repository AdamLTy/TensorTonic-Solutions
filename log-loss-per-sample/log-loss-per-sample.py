import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here
    losses = []
    for y, p in zip(y_true, y_pred):
        p = max(eps, min(p, 1 - eps))
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))

    return losses