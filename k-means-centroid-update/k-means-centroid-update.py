import numpy as np

def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    # Write code here
    d = len(points[0])
    points = np.asarray(points, dtype = float)
    assignments = np.asarray(assignments, dtype = float)
    output = [d * [0] for _ in range(k)]
    for i in range(k):
        mask = (assignments == i)
        if mask.any():
            selected_points = points[mask]
            output[i] = np.mean(selected_points, axis = 0).tolist()

    return output