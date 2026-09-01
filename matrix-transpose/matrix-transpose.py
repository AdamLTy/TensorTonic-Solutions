import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    m = len(A)
    n = len(A[0])

    output = [[0] * m for _ in range(n)]

    for i in range(m):
        for j in range(n):
            output[j][i] = A[i][j]

    return np.array(output)
