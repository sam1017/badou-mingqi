
from Tree import Tree

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


def cut_all(father, sentence, dict, max_length = 3):
    start_index, end_index = 0, 1
    end_value = min((start_index + max_length), len(sentence))
    end_index = start_index + 1
    while end_index <= end_value:
        word = sentence[start_index:end_index]
        if word in dict:
            if father.left is None:
                left = Tree(word)
                left.father = father
                father.left = left
                father.count += 1
                if end_index < len(sentence):
                    cut_all(left, sentence[end_index:], dict)
            elif father.right is None:
                right = Tree(word)
                right.father = father
                father.right = right
                father.count += 1
                if end_index < len(sentence):
                    cut_all(right, sentence[end_index:], dict)
        end_index += 1


string = "经常有意见分歧"
start = Tree()
cut_all(start, string, Dict)
start.getAllChild()

