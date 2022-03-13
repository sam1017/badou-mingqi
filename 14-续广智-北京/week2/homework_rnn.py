#coding:utf8
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
"""


class TextDataset(Dataset):
    def __init__(self, data_list, vocab, sentence_length, target_chars=None):
        Dataset.__init__(self)

        self.data = data_list
        self.vocab = vocab
        self.sentence_length = sentence_length
        self.target_chars = target_chars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # clip to sentence_length
        x_word = self.data[idx][:self.sentence_length]
        # fill with the index of 'unk'
        x = np.ones(self.sentence_length) * self.vocab['unk']
        # fill indices of chars
        x[:len(x_word)] = np.array(
                [self.vocab.get(char, self.vocab['unk']) for\
                char in x_word])   #将字转换成序号，为了做embedding
        x = torch.tensor(x).long()
        # if target_chars is given, also return y
        sample = [x,]
        if self.target_chars is not None:
            if set(self.target_chars) & set(x_word):
                y = 1
            else:
                y = 0
            sample.append(y)

        return sample

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, n_vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(n_vocab, vector_dim)  #embedding层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify = nn.Linear(vector_dim, 2)     #线性层
        self.loss = nn.CrossEntropyLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)     #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.rnn(x)[0]
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        y_pred = self.classify(x)   #(batch_size, vector_dim) -> (batch_size, 2)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

def load_data(abspath):
    '''Load a txt file'''
    with open(abspath, 'r', encoding='utf8') as fin:
        text_all = fin.read()
    return text_all.strip().split(' ')

def build_vocab():
    chars = '''abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:'"()[]{}@$&*%+-=<>'''
    vocab = {'unk': 0}  # default unk to 0
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    return vocab

def predict(model_path, vocab_path, input_strings, device='cpu'):

    device = torch.device(device)

    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    target_chars = 'auv'  # 检测出现的字符

    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModel(char_dim, sentence_length, len(vocab)).to(device)     #建立模型
    model.load_state_dict(torch.load(model_path, map_location=device))       #加载训练好的权重

    dataset = TextDataset(input_strings, vocab, sentence_length, target_chars=None)
    dataloader = DataLoader(dataset, batch_size=len(input_strings), shuffle=False)

    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        for x in dataloader:
            x = x[0]
            y = model(x)  #模型预测
            y = torch.softmax(y, 1)
            y = y.detach().numpy()
            y_pred = np.argmax(y, axis=1)
            y_prob = np.max(y, axis=1)

    print("检测是否包含字符:", target_chars)
    for i, input_string in enumerate(input_strings):
        print("输入：{:<20}\t预测类别：{:>4}\t概率值：{:>.4f}".format(
                input_string, y_pred[i], y_prob[i]))

    return

def train():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 8000    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.005 #学习率
    train_split = 0.8     # train/val split
    target_chars = 'auv'  # 检测出现的字符
    device = 'cpu'
    save_model_path = 'model.pth'

    # 建立字表
    vocab = build_vocab()
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))

    # 建立模型
    model = TorchModel(char_dim, sentence_length, len(vocab))     #建立模型
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 读取数据
    dataset = load_data('a_tale_of_two_cities.txt')[:train_sample]
    # split train-validation data
    n_train = int(len(dataset) * train_split)
    idx = np.random.permutation(len(dataset))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_dataset = TextDataset([dataset[ii] for ii in train_idx],
            vocab, sentence_length, target_chars)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TextDataset([dataset[ii] for ii in val_idx],
            vocab, sentence_length, target_chars)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 训练过程
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in train_dataloader:
            x, y = batch
            #ratio = (y.sum()/len(y)).item()
            #print('ratio', ratio)
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())

        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        model.eval()
        correct, wrong = 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                #ratio = (y.sum()/len(y)).item()
                #print('ratio', ratio)
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)      #模型预测
                y_pred = torch.softmax(y_pred, 1)
                y_pred = y_pred.detach().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                y = y.detach().numpy()
                c = (y_pred == y).sum()
                correct += c
                wrong += len(y) - c

        acc = correct/(correct+wrong)
        print("正确预测个数：%d, 正确率：%f"%(correct, acc))

        if epoch > 0 and acc > log[-1][0]:
            #保存模型
            print('Save model to:', save_model_path)
            torch.save(model.state_dict(), save_model_path)
        log.append([acc, np.mean(watch_loss)])

    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show(block=False)

    return


if __name__ == "__main__":

    arg = sys.argv[1]
    if arg == 'train':
        train()
    elif arg == 'test':
        test_strings = ['References', 'to', 'web-pages', 'containing',
                'multi-process', 'loading', 'configurations']
        predict("model.pth", "vocab.json", test_strings)
    else:
        sys.exit()

