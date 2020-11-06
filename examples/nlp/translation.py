"""
This module implements the seq2seq model, which includes a Decoder and an Encoder.
We also implements the Attention model and the example of machine translation.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import collections
import os
import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchtext.vocab as Vocab
import torch.utils.data as Data
import metrics


PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_seq(seq_tokens, all_tokens, all_seqs, max_len):
    """
    Record all tokens appeared in the given sequence into all_tokens, and add PAD at the end utils it reaches max_len.
    :param seq_tokens:
    :param all_tokens: all tokens recorded (permit repeat)
    :param all_seqs:
    :param max_len:
    :return:
    """
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


def build_data(all_tokens, all_seqs):
    """
    Create the vocabulary and tensor of tokens.
    """
    vocab = Vocab.Vocab(collections.Counter(all_tokens), specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[tk] for tk in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


def read_data(max_len):
    """
    Read translation data from a small-scale dataset fr-en-small.txt.
    Used for training.
    """
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with open('../../data/fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        # seq which longer than max_len is excluded
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_len - 1:
            continue
        process_seq(in_seq_tokens, in_tokens, in_seqs, max_len)
        process_seq(out_seq_tokens, out_tokens, out_seqs, max_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, Data.TensorDataset(in_data, out_data)


class Encoder(nn.Module):
    """
    Here the encoder is implemented as an RNN.
    Firstly, it translate the input token into its embedding vector;
    This embedding vec. is input into a GRU.
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0, **kwargs):
        """
        :param vocab_size:
        :param embed_size:
        :param num_hiddens: number of hidden units of each hidden layer
        :param num_layers: number of hidden layers (depth)
        :param drop_prob:
        :param kwargs:
        """
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, hidden_state):
        """
        :param inputs: of size (batch_size, num_steps)
        :param hidden_state:
        :return:
        """
        # (batch_size, num_steps, embed_size) ---> (num_steps, batch_size, embed_size)
        embeddings = self.embedding(inputs.long()).permute(1, 0, 2)
        # return:
        # outputs: (num_steps, batch_size, num_hiddens)
        # hidden_states: (num_layers, batch_size, num_hiddens)
        # (note that when the rnn is deep, the returned hidden states is the hidden units of each layer of the last time step)
        return self.rnn(embeddings, hidden_state)

    def begin_state(self):
        return None


def attention_model(input_size, attention_size):
    """
    In this implementation, input is (num_steps, batch_size, 2 * num_hiddens). See the func below.
    :param input_size: 2 * num_hiddens
    :param attention_size: num of hidden units
    :return:
    """
    return nn.Sequential(
        nn.Linear(input_size, attention_size, bias=False),
        nn.Tanh(),
        nn.Linear(attention_size, 1, bias=False)
    )


def attention_forward(a_model, enc_s, dec_s):
    """

    :param a_model:
    :param enc_s: the output of encoder, of size (num_steps, batch_size, num_hiddens)
    :param dec_s: hidden state of current time step of decoder, of size (batch_size, num_hiddens)
    :return: of size (batch_size, num_hiddens)
    """
    # (num_steps, batch_size, num_hiddens)
    dec_states = dec_s.unsqueeze(dim=0).expand_as(enc_s)       # broadcast
    enc_dec_states = torch.cat((enc_s, dec_states), dim=2)
    e = a_model(enc_dec_states)                                # (num_steps, batch_size, 1)
    alpha = F.softmax(e, dim=0)                                # (num_steps, batch_size, 1)
    # alpha is viewed as weights of enc_s on each time step
    # (num_steps, batch_size, 1) * (num_steps, batch_size, num_hiddens)
    # --> (num_steps, batch_size, num_hiddens) (broadcast automatically)
    # background vec for current dec_s is the weighted average, of size (batch_size, num_hiddens)
    return (alpha * enc_s).sum(dim=0)


class Decoder(nn.Module):
    """
    The initial hidden states of decoder is directly the final hidden states of encoder.
    Firstly, it translate the input token into its embedding vector;
    This embedding vec. is concatenated with the background vec. returned from attention model, as the input of GRU.
    The output is sent to a fc to get the probability distribution of the next token,
    while the hidden states is transferred to the next time step.
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)
        # what attention model input is the concatenation of broadcast of current dec_states and enc_s
        self.attention = attention_model(2 * num_hiddens, attention_size)
        # what attention model input is the concatenation of current embedding vec. and background vec.
        self.rnn = nn.GRU(num_hiddens + embed_size, num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, dec_states, enc_outputs):
        """
        :param cur_input: of size (batch_size)
        :param dec_states: (num_layers, batch_size, num_hiddens), initialized as the final hidden_states of encoder
        :param enc_outputs: (num_steps, batch_size, num_hiddens)
        :return: output (batch_size, vocab_size) and dec_states (num_layers, batch_size, num_hiddens)
        """
        c = attention_forward(self.attention, enc_outputs, dec_states[-1])
        # embedding(cur_input): (batch_size, embed_size)
        # c: (batch_size, num_hiddens)
        # input_and_c: (batch_size, embed_size + num_hiddens)
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)
        # num_steps is set as 1
        outputs, dec_states = self.rnn(input_and_c.unsqueeze(0), dec_states)
        # (1, batch_size, vocab_size) ---> (batch_size, vocab_size)
        return self.out(outputs).squeeze(dim=0), dec_states

    def begin_state(self, enc_states):
        return enc_states


def batch_loss(encoder, decoder, X, Y, loss, out_vocab):
    """

    :param encoder:
    :param decoder:
    :param X: of size (batch_size)
    :param Y: of size (batch_size, seq_len)
    :param loss:
    :param out_vocab:
    :return:
    """
    batch_size = X.shape[0]
    enc_states = encoder.begin_state()
    enc_outputs, enc_states = encoder(X, enc_states)
    dec_states = decoder.begin_state(enc_states)
    # input of the first time step is BOS
    dec_input = torch.tensor([out_vocab.stoi[BOS]] * batch_size)
    mask, num_not_pad_tks = torch.ones(batch_size,), 0
    ls = torch.tensor([0.])
    for y in Y.permute(1, 0):
        dec_outputs, dec_states = decoder(dec_input, dec_states, enc_outputs)
        # here loss is CrossEntropyLoss, set reduction as 'none' becasue we only want the loss for each token, not their average
        ls = ls + (mask * loss(dec_outputs, y)).sum()
        # use the corresponding y as the next input (compulsory teaching)
        dec_input = y
        num_not_pad_tks += mask.sum().item()
        mask = mask * (y != out_vocab.stoi[EOS]).float()
    return ls / num_not_pad_tks


def train(encoder, decoder, dataset, lr, batch_size, num_epochs, out_vocab):
    enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        ls_sum = 0.
        for X, Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            ls = batch_loss(encoder, decoder, X, Y, loss, out_vocab)
            ls.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            ls_sum += ls.item()
        if (epoch + 1) % 10 == 0:
            print('epoch %d, loss %.3f' % (epoch + 1, ls_sum / len(data_iter)))


def translate(encoder, decoder, input_seq, max_len):
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_len - len(in_tokens) - 1)
    # enc_input: (1, max_len=num_steps)
    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]])
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.tensor([out_vocab.stoi[EOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[pred.item()]
        if pred_token == EOS:
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens


def BLEU(pred_tokens, label_tokens, k):
    """
    The BLEU for evaluation of machine translation.
    :param pred_tokens:
    :param label_tokens:
    :param k:
    :return:
    """
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def score(input_seq, label_seq, k, max_len):
    pred_tokens = translate(encoder, decoder, input_seq, max_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s' % (BLEU(pred_tokens, label_tokens, k), ' '.join(pred_tokens)))


if __name__ == '__main__':
    # 1. load dataset
    max_len = 7
    in_vocab, out_vocab, dataset = read_data(max_len)
    print(dataset[0])

    # 2. initialize the encoder, decoder and train
    embed_size, num_hiddens, num_layers = 64, 64, 2
    attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
    encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers, drop_prob)
    decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers, attention_size, drop_prob)
    train(encoder, decoder, dataset, lr, batch_size, num_epochs, out_vocab)

    # 3. translate
    input_seq = 'ils regardent .'
    print(translate(encoder, decoder, input_seq, max_len))

    # 4. test BLEU
    score('ils regardent .', 'they are watching .', 2, max_len)
    score('ils sont canadienne .', 'they are canadian .', 2, max_len)
