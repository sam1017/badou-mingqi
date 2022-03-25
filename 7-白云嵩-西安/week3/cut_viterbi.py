# -*- coding: utf-8 -*-
# @Time    : 2022/3/15
# @Author  : baiyunsong
# @File    : cut_viterbi
# @Software: IDEA


def generate_vocab(sentence,dict,max_len):
    vocab = {}
    if sentence == '':
        return
    else:
        for i in range(len(sentence) - 1,-1,-1):
            word = sentence[i]
            vocab[i] = []
            lens = min(max_len,len(sentence[:i+1]))
            for j in range(lens):
                target = sentence[i-j:i+1]
                if target in dict.keys():
                    vocab[i].append(i-j)
    return vocab


def cut_viterbi(sentence,dict,max_len):
    f_scores = [0 for _ in range(len(sentence)+1)]
    p_idxs = [0 for _ in range(len(sentence)+1)]
    max_val = 0
    income = generate_vocab(sentence,dict,max_len)
    print(income)
    idx = 0
    if sentence == '':
        return
    else:
        for i in range(1,len(sentence)+1):
            #word = sentence[i]
            for j in income[i-1]:
                target = sentence[j:i]
                tmp = f_scores[j] + dict[target]
                #f_scores[i] = f_scores[j] + dict[target]
                if max_val < tmp:
                    max_val = f_scores[i]
                    f_scores[i] = tmp
                    idx = j
            p_idxs[i] = idx
    a = p_idxs[-1]
    b = len(sentence)
    res = []
    while a != 0:
        word = sentence[a:b]
        res.append(word)
        b = a
        a = p_idxs[a]
    word = sentence[a:b]
    res.append(word)
    #print(res[::-1])
    return f_scores,p_idxs,res[::-1]




if __name__ == '__main__':
    #词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
    Dict = {"经常":0.1,
            "经":0.05,
            "有":0.1,
            "常":0.001,
            "有意见":0.1,
            "歧":0.001,
            "意见":0.2,
            "分歧":0.2,
            "见":0.05,
            "意":0.05,
            "见分歧":0.05,
            "分":0.1}

    #待切分文本
    sentence = "经常有意见分歧"

    f_scores,p_idxs,result = cut_viterbi(sentence,Dict,3)
    print(f_scores)
    print(p_idxs)
    print(result)
    #print(len(target))