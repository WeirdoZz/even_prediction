import math
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np


def bleu_each(hyps, refs):
    """
    bleu
    """
    bleu_4 = []
    hyps=hyps.cpu().numpy()
    refs=refs.cpu().numpy()
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_4.append(score)
    return bleu_4


def dcg_k(actual, predicted, topk):
    """
    获取预测值的dcg评分

    :param actual: 真实的事件链
    :param predicted: 预测的事件链
    :param topk: 最高需要判断多长的连续的子序列
    :return: 返回每个用户的dcg得分
    """

    k = min(topk, len(actual))

    dcgs = []
    # 将预测序列和真实序列转成numpy格式
    predicted = predicted.cpu().numpy()

    for chain_id in range(len(actual)):
        # 分别计算每个用户的dcg_k值
        value = []

        for i in predicted[chain_id]:
            try:
                value += [topk - int(np.argwhere(actual[chain_id] == i))]
            except:
                value += [0]

        dcg_k = sum([value[j] / math.log(j + 2, 2) for j in range(k)])

        if dcg_k == 0:
            dcg_k = 1e-5
        dcgs.append(dcg_k)

    return sum(dcgs)

def precision_at_k(actual, predicted,item_i):
    sum_precision = 0.0
    chain = 0
    num_chains = len(predicted)
    for i in range(num_chains):
        if actual[i][item_i]>0:
            chain +=1
            act_set = actual[i][item_i]
            pred_set = predicted[i]
            if act_set in pred_set:
                sum_precision += 1
        else:
            continue
    #print(user)
    return sum_precision / chain

def ndcg_k(actual, predicted, topk,item_i):
    k = min(topk, len(actual))
    res = 0
    chain = 0
    for chain_id in range(len(actual)):
        if actual[chain_id][item_i] > 0:
            chain +=1
            dcg_k = sum([int(predicted[chain_id][j] in [actual[chain_id][item_i]]) / math.log(j + 2, 2) for j in range(k)])
            res += dcg_k
        else:
            continue
    return res/chain

def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res