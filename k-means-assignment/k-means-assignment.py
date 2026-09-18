import numpy as np

def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    # Write code here
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    output = np.argmin(np.sum((points[:, None, :] - centroids) ** 2, axis=-1), axis=-1)
    return output.tolist()