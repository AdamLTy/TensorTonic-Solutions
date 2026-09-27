import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    w = np.asarray(w)
    G = np.asarray(G)
    g = np.asarray(g)
    N = w.shape[0]
    lr = np.full((N), lr)
    G = G + g ** 2
    w = w - lr / (np.sqrt(G + eps)) * g

    return {
        "new_w":w,
        "new_G":G
    }
    