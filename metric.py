import numpy as np

def evaluate(pair):
    MR = 0; MRR = 0; hit_1 = 0; hit_3 = 0; hit_10 = 0
    n = len(pair)
    for p in pair:
        id_true = p[0]
        id_pred = p[1]
        MR += np.argwhere(id_pred == id_true)[0][0]
        MRR += 1/MR
        hit_1 += (id_true == id_pred[0])
        hit_3 += id_true in id_pred[0:3]
        hit_10 += id_true in id_pred[0:10]
    MR /= n; MRR /= n; hit_1 /= n; hit_3 /= n; hit_10 /= n
    return {'MR': MR, 'MRR': MRR, 'hit@1': hit_1, 'hit@3': hit_3, 'hit@10': hit_10}