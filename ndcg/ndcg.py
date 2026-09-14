import math

def gain(x):
    return 2 ** x - 1

def ndcg(relevance_scores: list, k: int) -> float:
    """
    Returns NDCG as a float.
    """
    # 1. 计算实际截断到 k 的 DCG
    k_scores = relevance_scores[:k]
    dcg = 0
    for index, score in enumerate(k_scores):
        dcg += gain(score) / math.log(index + 2, 2)

    # 2. 计算 IDCG：对完整列表降序排序，再取前 k 个
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = 0
    for index, score in enumerate(ideal_scores):
        idcg += gain(score) / math.log(index + 2, 2)

    # 3. 避免全 0 时分母为 0
    if idcg == 0:
        return 0.0

    return float(dcg / idcg)