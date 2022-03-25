# -*- coding: utf-8 -*-
"""====================================================
@Author  : Song Jiazhen
@Date    : 2022/3/17
@Describe:
===================================================="""
def cut_method(sentence, word_dict, max_len, words ,result):
    if sentence == '':
        result.append(words)
        return
    else:
        lens = min(max_len,len(sentence))
        for _ in range(lens):
            word = sentence[:lens]
            new_words = words.copy()
            if word in word_dict:
                new_words.append(word)
                new_sentence = sentence[lens:]
                cut_method(new_sentence, word_dict, max_len, new_words, result)
            lens = lens - 1
    return

def all_cut(sentence, Dict):
    max_len = 0
    words = []
    result = []
    for k in Dict.keys():
        tmp_l = len(k)
        if tmp_l > max_len:
            max_len = tmp_l
    cut_method(sentence, Dict, max_len, words, result)
    return result


if __name__ == '__main__':
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
    result = all_cut(sentence, Dict)
    for data in result:
        print(data)