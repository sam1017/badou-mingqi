def cut_method(string, Dict, max_len, words, target):
    while string == '':
        target.append(words)
        return
    else:
        lens = min(max_len,len(string))
        for _ in range(lens):
            word = string[:lens]
            my_words = words.copy()
            if word in Dict:
                my_words.append(word)
                str_new = string[lens:]
                cut_method(str_new, Dict, max_len, my_words, target)
            lens = lens - 1
    return target

def all_cut(sentence, Dict):
    max_len = 0
    words = []
    target = []
    for k in Dict.keys():
        tmp_l = len(k)
        if tmp_l > max_len:
            max_len = tmp_l
    cut_method(sentence, Dict, max_len, words, target)
    return target



#待切分文本
sentence = "经常有意见分歧"
result_str = all_cut(sentence,Dict={
        "经常":0.1,
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
        "分":0.1})
print(result_str)