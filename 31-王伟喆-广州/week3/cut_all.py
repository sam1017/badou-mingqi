#coding:utf8

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""
# 本次代码参考于别人，感谢大佬哈
def max_word_len(dict):
    """
    获取词典中最大词的长度
    """
    max_word_len = 0
    for word in dict.keys():
        if len(word) > max_word_len:
            max_word_len = len(word)
    return max_word_len

target = []
def cut_method(string, word_dict, max_len, words):
    if string == '':
        target.append(words)
        # pass
    else:
        lens = min(max_len, len(string))
        while lens != 0:
            word = string[:lens]
            my_words = words.copy()
            # print(my_words)
            if word in word_dict:
                my_words.append(word)
                str_new = string[lens:]
                cut_method(str_new, word_dict, max_len, my_words)
            lens = lens - 1
    return target


if __name__ == '__main__':
    Dict = {"经常": 0.1,
            "经": 0.05,
            "有": 0.1,
            "常": 0.001,
            "有意见": 0.1,
            "歧": 0.001,
            "意见": 0.2,
            "分歧": 0.2,
            "见": 0.05,
            "意": 0.05,
            "见分歧": 0.05,
            "分": 0.1}

    sentence = "经常有意见分歧"
    max_len = max_word_len(Dict)
    cut_traget = []
    cut_method(sentence, Dict, max_len, cut_traget)

    # 打印切分结果
    for i in target:
        print(i)

