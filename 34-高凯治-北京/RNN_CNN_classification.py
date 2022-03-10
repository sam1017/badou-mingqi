import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt


#判断cuda
if torch.cuda.is_available():
    device=torch.device("cuda:0")
else:
    device=torch.device("cpu")
#设定随机种子
def setup_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)   #为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   #如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(0)

#建立语料
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {}
    vocab.update({"unk": 0, "pad": 1})
    for index, char in enumerate(chars):
        vocab[char] = index+2         #每个字对应一个序号
    return vocab

def build_sample(vocab, min_sentence_length, max_sentence_length):
    l = random.randint(min_sentence_length, max_sentence_length)
    x = []
    for _ in range(l):
        char = random.choice(list(vocab.keys()))
        while vocab[char]==1 or vocab[char]==0:
            char = random.choice(list(vocab.keys()))
        x.append(char)
    assert len(x) == l, "文本长度不等于l"
    if set("gkz") & set(x):
        y = 1
    #指定字都未出现，则为负样本
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字符转换成序号，为了做embedding
    return x, y

def build_dataset(sample_length, vocab, min_sentence_length, max_sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, min_sentence_length, max_sentence_length)
        dataset_x.append(x)    #二维列表
        dataset_y.append([y])  #二维列表
    return dataset_x, torch.Tensor(dataset_y)


def padding_dataset(vocab, dataset):
    l=[len(k) for k in dataset]
    max_len = max(l)
    result = []
    for text in dataset:
        if len(text)<max_len:
            for i in range(max_len-len(text)):
                text.append(vocab["pad"])
        result.append(text)
    return torch.LongTensor(result)



class my_model(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size):
        super(my_model, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.pool = nn.AvgPool1d(sentence_length)     #池化层

        self.conv1 = nn.Conv2d(1, 1, (3, vector_dim))
        self.conv2 = nn.Conv2d(1, 1, (4, vector_dim))
        self.conv3 = nn.Conv2d(1, 1, (5, vector_dim))
        self.Max1_pool = nn.MaxPool2d((sentence_length-3+1, 1))
        self.Max2_pool = nn.MaxPool2d((sentence_length-4+1, 1))
        self.Max3_pool = nn.MaxPool2d((sentence_length-5+1, 1))

        self.classify = nn.Linear(vector_dim+3, 1)     #线性层
        self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.mse_loss  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)(16,15,64)
        x4, _ = self.rnn(x)                            #(batch_size, sen_len, vector_dim) ->(batch_size,seq_len,D*hidden_size)(16,15,2*32)
        x4 = self.pool(x4.transpose(1, 2)).squeeze()  # (batch_size, sen_len, vector_dim) -> (16, 64)


        batch = x.shape[0]
        x = x.unsqueeze(1)                         #(batch_size, in_channel, seq_len,D*hidden_size)(16,1,15,64)

        # Convolution
        x1 = F.relu(self.conv1(x))                #(batch_size,Cout, sen_len-3+1, 1) (16,1，15,1)
        x2 = F.relu(self.conv2(x))                #(batch_size,Cout, sen_len-4+1, 1) (16,1，15,1)
        x3 = F.relu(self.conv3(x))                #(batch_size,Cout, sen_len-5+1, 1) (16,1，15,1)
        # Pooling
        x1 = self.Max1_pool(x1)                    #(batch_size,1，1,1）
        x2 = self.Max2_pool(x2)                    #(batch_size,1，1,1）
        x3 = self.Max3_pool(x3)                    #(batch_size,1，1,1）
        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)            #(batch_size,1，1,3）
        x = x.view(batch, 1, -1)                   #(batch_size,1,3）
        # project the features to the labels
        x = x.squeeze(1)                           #(batch_size,3）

        #把CNN与RNN的向量表征拼接起来
        x = torch.cat((x, x4), -1)                 # (batch_size,67）
        x = self.classify(x)                       #(batch_size, 67) -> (batch_size, 1)


        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#建立模型
def build_model(char_dim, sentence_length, vocab, hidden_size):
    model = my_model(char_dim, sentence_length, vocab, hidden_size)
    return model

#测试代码,用来测试每轮模型的准确率
def evaluate(model, vocab, min_sentence_length, max_sentence_length, batch_size):
    model.eval()
    x, y = build_dataset(208, vocab, min_sentence_length, max_sentence_length)  #建立200个用于测试的样本
    x = padding_dataset(vocab, x)
    correct, wrong, p_sample= 0, 0, 0
    for i in range(int(208 / batch_size)):
        batch = x[i * batch_size:i * batch_size + batch_size]  # 一个batch的数据
        label = y[i * batch_size:i * batch_size + batch_size]  # 一个batch的标签
        batch = batch.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_pred = model(batch)      #模型预测
            for y_p, y_t in zip(y_pred, label):  #与真实标签进行对比
                if float(y_p) < 0.5 and int(y_t) == 0:
                    correct += 1   #负样本判断正确
                elif float(y_p) >= 0.5 and int(y_t) == 1:
                    correct += 1   #正样本判断正确
                else:
                    wrong += 1
        p_sample = p_sample+sum(label)
    print("本次预测集中共有%d个正样本，%d个负样本" % (p_sample, 208 - p_sample))
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)
def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 16       #每次训练样本个数
    hidden_size = 32        #隐藏层维度
    train_sample = 640    #每轮训练总共训练的样本总数
    char_dim = 64         #每个字的维度
    min_sentence_length = 10   #最小文本长度
    max_sentence_length = 20   #最大文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    #建立训练数据
    x, y = build_dataset(train_sample, vocab, min_sentence_length, max_sentence_length)
    #填充训练数据
    x = padding_dataset(vocab, x)
    # 建立模型
    sentence_length = len(x[0])
    model = build_model(char_dim, sentence_length, vocab, hidden_size)
    model = model.to(device)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for i in range(int(train_sample / batch_size)):
            batch = x[i * batch_size:i * batch_size + batch_size]    #一个batch的数据
            label = y[i * batch_size:i * batch_size + batch_size]    #一个batch的标签
            batch =batch.to(device)
            label =label.to(device)
            optim.zero_grad()    #梯度归零
            loss = model(batch, label)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, min_sentence_length, max_sentence_length, batch_size)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings, hidden_size):
    char_dim = 64  # 每个字的维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    padding_input_strings = padding_dataset(vocab, x)
    sentence_length = len(padding_input_strings[0])
    model = build_model(char_dim, sentence_length, vocab, hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果



if __name__ == "__main__":
    #main()
    test_strings = ["ab", "casdfg", "rb", "nlkdwwjdhfjbakjfsjkagfsab"]
    predict("model.pth", "vocab.json", test_strings, 32)
