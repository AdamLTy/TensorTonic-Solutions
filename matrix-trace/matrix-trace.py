import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    # Write code here
    n = len(A)
    res = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                res += A[i][j]

    return res