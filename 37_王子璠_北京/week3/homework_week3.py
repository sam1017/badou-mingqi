def all_cut(sentence, Dict):
    target = []
    path = []
    process(sentence, Dict, 0, target, path)
    return target

def process(sentence, Dict, index, target, path):
    if index == len(sentence):
        target.append(list(path))
    for i in range(index, len(sentence)):
        s = sentence[index:i + 1]
        if s in Dict:
            path.append(s)
            process(sentence, Dict, i + 1, target, path)
            path.remove(s)

if __name__ == "__main__":
    Dict = {
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
        "分":0.1
    }
    sentence = "经常有意见分歧"

    target = all_cut(sentence, Dict)
    for t in target:
        print(t)