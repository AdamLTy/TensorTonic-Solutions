import numpy as np

def nesterov_momentum(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    """
    Returns classical and Nesterov MSE loss curves in a dictionary.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape
    w = np.zeros(D)
    v_t = 0
    res = []
    for _ in range(n_epochs):
        y_pred = X @ w
        loss = np.mean((y_pred - y) ** 2)
        grad = (2 / N) * X.T @ (y_pred - y)
        v_t = beta * v_t + grad
        w -= lr * v_t
        res.append(loss.item())

    
    w = np.zeros(D)
    v_t = 0
    res2 = []
    for _ in range(n_epochs):
        y_pred_ = X @ w
        y_pred = X @ (w - lr * beta * v_t)
        loss = np.mean((y_pred_ - y) ** 2)
        grad = (2 / N) * X.T @ (y_pred - y)
        v_t = beta * v_t + grad
        w -= lr * v_t
        res2.append(loss.item())
    return {
        "classical_losses":res,
        "nesterov_losses":res2
    }
        
        