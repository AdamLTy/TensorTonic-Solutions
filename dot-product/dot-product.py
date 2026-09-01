import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # Write code here
    res = 0.0
    for i in range(len(x)):
        res += x[i] * y[i]

    return res
            