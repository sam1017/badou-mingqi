import numpy as np
import torch.nn as nn
import random
import torch
def build_vocab():

    chars='abcdefghijklmnopqrstuvwxyz'
    vocab={}
    for idx,word in enumerate(chars):
        vocab[word]=idx

    vocab['unk']=len(vocab)
    return vocab

def build_sample(sample_lenght,vocab):
    x=[random.choice(list(vocab.keys())) for _ in range(sample_lenght)]
    if set('abc') & set(x):
        y=1
    else:
        y=0
    x=[vocab.get(word,vocab['unk']) for word in x]

    return x,y

def build_dataset(sample_lenght,sample_num,vocab):
    sample_x=[]
    sample_y=[]
    for i in range(sample_num):
        x,y = build_sample(sample_lenght,vocab)
        sample_x.append(x)
        sample_y.append([y])
    return torch.LongTensor(sample_x),torch.FloatTensor(sample_y)

class model(nn.Module):
    def __init__(self,vector_dim,sentence_lenght,vocab):
        super(model,self).__init__()
        self.embedding=nn.Embedding(len(vocab),vector_dim)
        self.rnn=nn.RNN(vector_dim,vector_dim)
        self.pool=nn.AvgPool1d(sentence_lenght,1)
        self.classify=nn.Linear(vector_dim,1)
        self.activation=torch.sigmoid
        self.loss=nn.functional.mse_loss

    def forward(self,x,y=None):
        x=self.embedding(x)
        _,x=self.rnn(x)
        x=self.pool(x.transpose(1,2)).squeeze()
        y_pred=self.classify(x)
        if y is not None:
            return self.loss(y_pred,y.squeeze())
        else:
            return y_pred

def main():
    char_dim=20
    batch_size=20
    epoch=10
    train_sample=500
    sample_lenght=6
    learning_ratio=0.005
    vocab=build_vocab()
    _model=model(char_dim,sample_lenght,vocab)
    optim=torch.optim.Adam(_model.parameters(),lr=learning_ratio)
    for i in range(epoch):
        _model.train()
        print_loss=[]
        for batch in range(int(train_sample/batch_size)):
            sample_x, sample_y = build_dataset(sample_lenght, batch_size, vocab)
            optim.zero_grad()
            loss=_model(sample_x,sample_y)
            loss.backward()
            optim.step()
            print_loss.append(loss.item())
        print('现在是第%d轮训练,损失为%f'%(i+1,np.mean(print_loss)))
    torch.save(_model.state_dict(),'model.pth')

def predict(test_string,model_path):
    # assert len(test_string.item()) == 6
    char_dim=20
    char_lenght=6
    vocab=build_vocab()
    _model=model(char_dim,char_lenght,vocab)
    _model.load_state_dict(torch.load(model_path))
    _model.eval()

    for i in range(len(test_string)):
        x = []
        for input_char in test_string[i]:
             x.append(vocab[input_char])
        x=torch.tensor(x).unsqueeze(0)
        pred_result=_model(x)
        if pred_result >= 0.5:
            print(1)
        elif pred_result<0.5:
            print(0)

if __name__== '__main__':
    # main()
    test_string=['abcdes','hjkuih','jklope']
    predict(test_string,'model.pth')