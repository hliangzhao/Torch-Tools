"""
This module implements sentiment classification for text.
Two models: (1) Bi-directional RNN, (2) textCNN.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import collections
import random
from tqdm import tqdm
import torch
import os
from torch import nn
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data
import metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = '../../../D2L-PyTorch/data'           # this dir is out of this project


def get_pretrained_vec(name='6B', dim=100):
    """
    Get pretrained embedding vectors.
    """
    # this dir is out of this project
    return Vocab.GloVe(name, dim, cache='../../../D2L-PyTorch/data/pretrained_glove')


def read_imdb(folder='train', path=os.path.join(DATA_ROOT, 'aclImdb')):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(path, folder, label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                # a sample is consists of one review text and pos/neg
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


def tokenize(data):
    """
    Input all tokens into a list from the input data, and return it.
    :param data: a list of samples, i.e. [review, 1 or 0]
    :return: a list of size (sample_num, review_tokens_num)
    """
    def tokenizer(text):
        return [tk.lower() for tk in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


def creat_vocab(data):
    """
    Create vocabulary from the given dataset.
    """
    tokenized_data = tokenize(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # use Vocab to create vocabulary directly!
    return Vocab.Vocab(counter, min_freq=5)


def preprocess(data, vocab,  max_len=500):
    """
    Tokenize each review, change into indices, and pending with 0 or truncating to fix length as 500.
    """
    def pad(x):
        return x[:max_len] if len(x) > max_len else x + [0] * (max_len - len(x))

    tokenized_data = tokenize(data)
    features = torch.tensor([pad([vocab.stoi[tk] for tk in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


def get_data_iter(train_data, test_data, vocab, batch_size=64):
    train_set = Data.TensorDataset(*preprocess(train_data, vocab))
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_set = Data.TensorDataset(*preprocess(test_data, vocab))
    test_iter = Data.DataLoader(test_set, batch_size, shuffle=True)
    return train_iter, test_iter


class BiRNN(nn.Module):
    """
    Create a deep bidirectional LSTM.
    Firstly, the input words are transferred into embedding vectors.
    Secondly, use BiRNN to encode the embedding vectors.
    Thirdly, take the hidden states encoded by BiRNN as the representative feature for classification (decoding).
    """
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim=embed_size)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, bidirectional=True)
        # in this model, we concatenate the hidden states (2 because it is bidirectional) of the first time step
        # and the last time step, and view this as the feature to decode
        # thus, the input feature size is 4 * num_hiddens
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        """
        :param inputs: of size (batch_size, seq_len), here max_len is the feature length (also num_steps)
        :return: of size (batch_size, 2)
        """
        # embedding is used to replace one_hot
        # after embedding, (seq_len, batch_size) ---> (seq_len, batch_size, embed_size)
        embeddings = self.embedding(inputs.permute(1, 0))
        # (h_0, c_0) is not provided, which is initialized as zeros by default
        # we don't need (h_n, c_n), only want the outputs, which is (seq_len, batch_size, 2 * hidden_size)
        # this outputs is not the final output, but the input of dense layer (softmax layer)
        outputs, _ = self.encoder(embeddings)
        # concatenate the last dim, we get (batch_size, 4 * hidden_size)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        # return is of size (batch_size, 2)
        return self.decoder(encoding)


class TextCNN(nn.Module):
    """
    Given the word sequence, go through several conv layers, each then go through a global max pooling layer.
    THe output of each pooling layer is concatenated, as the input of dense layer.
    """
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim=embed_size)
        self.constant_embedding = nn.Embedding(len(vocab), embedding_dim=embed_size)
        self.dropout = nn.Dropout(0.5)

        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = metrics.GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2 * embed_size, out_channels=c, kernel_size=k))

    def forward(self, inputs):
        """
        :param inputs: of size (batch_size, seq_len)
        :return: of size (batch_size, 2)
        """
        # (batch_size, seq_len, 2 * embed_size)
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # (batch_size, 2 * embed_size, seq_len)
        embeddings = embeddings.permute(0, 2, 1)
        # conv(embeddings) is of size (batch_size, out_channels, out_seq_len)
        # pool(relu(conv(embeddings))) is of size (batch_size, out_channels, 1)
        # after squeeze(-1), the last dim disappeared ---> (batch_size, out_channels)
        # after cat on dim=1 ---> (batch_size, sum_out_channels), as the input for dense layer
        encoding = torch.cat(
            [self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs],
            dim=1
        )
        return self.decoder(self.dropout(encoding))


def load_pretrained_embedding(tokens, pretrained_vocab):
    """
    Find the pretrained embedding vectors for tokens appeared in our dataset and replace the original embedding vectors.
    We don't need to train embedding parameters.
    There may exist tokens which do not appear in the pretrained_vocab.
    In this case, we just set their embedding vector as zero and do nothing.
    """
    embeds = torch.zeros(len(tokens), pretrained_vocab.vectors[0].shape[0])
    oov_count = 0    # out of vocab num
    for i, tk in enumerate(tokens):
        try:
            idx = pretrained_vocab.stoi[tk]
            embeds[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print('There are %d out of vocabulary tokens' % oov_count)
    return embeds


def predict_sentiment(net, vocab, sentence):
    this_device = list(net.parameters())[0].device
    sentence = torch.tensor([vocab.stoi[tk] for tk in sentence], device=this_device)
    label = torch.argmax(net(sentence.view(1, -1)), dim=1)
    return 'positive' if label.item() == 1 else 'negative'


if __name__ == '__main__':
    # 1. load dataset and get vocab
    train_data, test_data = read_imdb('train'), read_imdb('test')
    vocab = creat_vocab(train_data)
    print(len(vocab), type(vocab))

    # 2. creat data iter for training
    batch_size = 64
    train_iter, test_iter = get_data_iter(train_data, test_data, vocab, batch_size)
    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        break
    print('#batches:', len(train_iter))

    # load pretrained embedding vectors
    glove = get_pretrained_vec()

    # 3. test BiRNN
    embed_size, num_hiddens, num_layers = 100, 100, 2
    net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
    net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove))
    net.embedding.weight.requires_grad = False

    lr, num_epochs_1 = 0.01, 1
    optimizer_1 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss_1 = nn.CrossEntropyLoss()
    metrics.universal_train(net, train_iter, test_iter, loss_1, num_epochs_1, batch_size, lr, optimizer_1)

    print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))

    # 4. test TextCNN
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
    lr, num_epochs_2 = 0.001, 5
    optimizer_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    loss_2 = nn.CrossEntropyLoss()
    metrics.universal_train(net, train_iter, test_iter, loss_2, num_epochs_2, batch_size, lr, optimizer_2)

    print(predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))
