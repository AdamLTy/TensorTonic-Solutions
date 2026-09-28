import numpy as np

def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores[:k])
    if relevance_scores.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, 2 + relevance_scores.size))
    return float(np.sum((2 ** relevance_scores - 1) / discounts))

def ndcg(relevance_scores: list, k: int) -> float:
    """
    Returns NDCG as a float.
    """
    # Write code here
    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k(np.sort(relevance_scores)[::-1], k)

    return dcg / idcg if idcg > 0 else 0.0