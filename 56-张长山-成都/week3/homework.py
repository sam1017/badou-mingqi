# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {
    "经常": 0.1,
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
    "分": 0.1,
}

# 待切分文本
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # TODO
    targets = []
    cuts = []

    def dfs(word_start, word_end, cuts):
        word = sentence[word_start:word_end]
        if not word:
            targets.append(cuts)
            return True

        if word in Dict:
            new_word_start = word_start + len(word)
            new_word_end = new_word_start + 1
            if dfs(new_word_start, new_word_end, cuts):
                cuts.append(word)
                return True
            else:
                return dfs(new_word_start, new_word_end + 1, cuts)
        else:
            return False

    dfs(0, 1, cuts)
    return target


# 目标输出;顺序不重要
target = [
    ["经常", "有意见", "分歧"],
    ["经常", "有意见", "分", "歧"],
    ["经常", "有", "意见", "分歧"],
    ["经常", "有", "意见", "分", "歧"],
    ["经常", "有", "意", "见分歧"],
    ["经常", "有", "意", "见", "分歧"],
    ["经常", "有", "意", "见", "分", "歧"],
    ["经", "常", "有意见", "分歧"],
    ["经", "常", "有意见", "分", "歧"],
    ["经", "常", "有", "意见", "分歧"],
    ["经", "常", "有", "意见", "分", "歧"],
    ["经", "常", "有", "意", "见分歧"],
    ["经", "常", "有", "意", "见", "分歧"],
    ["经", "常", "有", "意", "见", "分", "歧"],
]

res = all_cut(sentence, Dict)
for cut in res:
    print(cut)

assert res == target
