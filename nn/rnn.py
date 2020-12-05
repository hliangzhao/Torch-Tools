"""
This module implements RNN models.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
import tools
import math
import time
import metrics
import numpy as np
from torch import nn, optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
In the following, implement RNN, GRU, and LSTM without using nn.RNN, nn.GRU, and nn.LSTM.
"""


def get_rnn_params(num_inputs, num_hiddens, num_outputs):
    """
    Initialize the params for the basic RNN.
    The basic RNN is a one-hidden-layer MLP with hidden states.
    """
    def _one(shape):
        return nn.Parameter(
            torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32),
            requires_grad=True
        )

    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nn.Parameter(torch.zeros(num_hiddens, device=device), requires_grad=True)
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device), requires_grad=True)

    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def get_gru_params(num_inputs, num_hiddens, num_outputs):
    """
    Initialize the params for GRU.
    """
    def _one(shape):
        return nn.Parameter(
            torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32),
            requires_grad=True
        )

    def _three():
        return (
            _one((num_inputs, num_hiddens)),
            _one((num_hiddens, num_hiddens)),
            nn.Parameter(torch.zeros(num_hiddens, device=device), requires_grad=True)
        )

    W_xz, W_hz, b_z = _three()
    W_xr, W_hr, b_r = _three()
    W_xh, W_hh, b_h = _three()

    W_hq = _one((num_hiddens, num_outputs))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device), requires_grad=True)
    return nn.ParameterList([W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q])


def get_lstm_params(num_inputs, num_hiddens, num_outputs):
    """
    Initialize the params for LSTM.
    """
    def _one(shape):
        return nn.Parameter(
            torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32),
            requires_grad=True
        )

    def _three():
        return (
            _one((num_inputs, num_hiddens)),
            _one((num_hiddens, num_hiddens)),
            nn.Parameter(torch.zeros(num_hiddens, device=device), requires_grad=True)
        )

    W_xi, W_hi, b_i = _three()
    W_xf, W_hf, b_f = _three()
    W_xo, W_ho, b_o = _three()
    W_xc, W_hc, b_c = _three()

    W_hq = _one((num_hiddens, num_outputs))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),   # for hidden state
            torch.zeros((batch_size, num_hiddens), device=device))   # for cell


def rnn(inputs, hidden_state, params):
    """
    Inputs and outputs are num_steps tensors, each of size (batch_size, vocab_size), rather than one tensor of size
    (batch_size, num_steps). In other words, the input is after one-hot encoding.
    :param inputs:
    :param hidden_state:
    :param params:
    :return: outputs and (H,). Here we return (H,) because we can use state[0] to represent hidden state.
    We write it this way to keep the same with LSTM (its outputs state is (hidden state, cell state).
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = hidden_state
    outputs = []
    for X in inputs:         # iteration over num_steps
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def gru(inputs, hidden_state, params):
    """
    Inputs and outputs are num_steps tensors, each of size (batch_size, vocab_size).
    """
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = hidden_state
    outputs = []
    for X in inputs:         # iteration over num_steps
        Z = torch.sigmoid(torch.matmul(X, W_xz) + torch.matmul(H, W_hz) + b_z)
        R = torch.sigmoid(torch.matmul(X, W_xr) + torch.matmul(H, W_hr) + b_r)
        H_tilde = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilde
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def lstm(inputs, hidden_state, params):
    """
    Inputs and outputs are num_steps tensors, each of size (batch_size, vocab_size).
    """
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = hidden_state
    outputs = []
    for X in inputs:         # iteration over num_steps
        I = torch.sigmoid(torch.matmul(X, W_xi) + torch.matmul(H, W_hi) + b_i)
        F = torch.sigmoid(torch.matmul(X, W_xf) + torch.matmul(H, W_hf) + b_f)
        O = torch.sigmoid(torch.matmul(X, W_xo) + torch.matmul(H, W_ho) + b_o)
        C_tilde = torch.tanh(torch.matmul(X, W_xc) + torch.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilde
        H = O * C.tanh()
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)


"""
In the following, implement RNN, GRU, and LSTM by using nn.RNN, nn.GRU, and nn.LSTM directly.
"""


def rnn_layer(input_size, hidden_size):
    """
    Define RNN with torch directly.
    Input X should be a tensor, which of shape (num_steps, batch_size, vocab_size).
    """
    return nn.RNN(input_size=input_size, hidden_size=hidden_size)


def gru_layer(input_size, hidden_size):
    """
    Define GRU with torch directly.
    Input X should be a tensor, which of shape (num_steps, batch_size, vocab_size).
    """
    return nn.GRU(input_size=input_size, hidden_size=hidden_size)


def lstm_layer(input_size, hidden_size):
    """
    Define LSTM with torch directly.
    Input X should be a tensor, which of shape (num_steps, batch_size, vocab_size).
    """
    return nn.LSTM(input_size=input_size, hidden_size=hidden_size)


class RNNModel(nn.Module):
    """
    Wrap the RNN, GRU or LSTM to get a complete net.
    """
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)
        self.hidden_state = None

    def forward(self, inputs, hidden_state):
        # inputs is of shape (batch_size, num_steps)  ---> X is of size num_steps * (batch_size, vocab_size)
        X = metrics.to_onehot(inputs, self.vocab_size)
        # use torch.stack(X) to change the list X into a tensor (num_steps, batch_size, vocab_size)
        # Y is of size (num_steps, batch_size, hidden_size)
        Y, self.hidden_state = self.rnn(torch.stack(X), hidden_state)
        # change Y's shape into (num_steps * batch_size, hidden_size), and get (num_steps * batch_size, vocab_size) as the output
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.hidden_state


def rnn_train_and_predict(model, params, init_hidden_state, num_hiddens, vocab_size, idx_to_char, char_to_idx, device,
                          corpus_indices, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    """
    RNN train and prediction.
    :param model: the fn to get a rnn model
    :param params:
    :param init_hidden_state:
    :param num_hiddens:
    :param vocab_size:
    :param idx_to_char:
    :param char_to_idx:
    :param device:
    :param corpus_indices:
    :param is_random_iter: the data batch is obtained by random sampling or consecutive sampling
    :param num_epochs:
    :param num_steps:
    :param lr:
    :param clipping_theta:
    :param batch_size:
    :param pred_period: how often do we make predictions
    :param pred_len: the next pred_len chars to predict
    :param prefixes: the list of prefixes
    :return:
    """
    if is_random_iter:
        data_iter_fn = tools.get_timeseries_data_batch_random
    else:
        data_iter_fn = tools.get_timeseries_data_batch_consecutive
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            # if use consecutive sampling, the hidden state should be initialized only at the beginning of each epoch
            # initialization is not required at the beginning of each minibatch
            hidden_state = init_hidden_state(batch_size, num_hiddens, device)
        ls_sum, n, start = 0., 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:
                # if use random sampling, the hidden state should be initialized at the beginning of each minibatch
                hidden_state = init_hidden_state(batch_size, num_hiddens, device)
            else:
                # notice that hidden_state is of size (batch_size, num_hiddens), which are belated from the last epoch
                # the hidden states of these samples should be detached from the computation graph
                for s in hidden_state:
                    s.detach_()

            inputs = metrics.to_onehot(X, num_classes=vocab_size)
            (outputs, hidden_state) = model(inputs, hidden_state, params)
            # notice that the outputs is a list of num_steps tensors, each of size (batch_size, vocab_size)
            # after torch.cat on dim=0, outputs is a tensor of size (num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # y = [sample1_step1, sample2_step1, ..., samplen_step1,
            #      sample1_step2, sample2_step2, ..., samplen_step2,
            #      ...        ...     ...     ...
            #      sample1_stepm, sample2_stepm, ..., samplen_stepm]
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            ls = loss(outputs, y.long())    # turn y into long type as label because it is used as index for torch.gather

            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            ls.backward()
            metrics.grad_clipping(params, clipping_theta, device)
            metrics.sgd(params, lr, batch_size=1)         # ls has been averaged, here batch_size is set as 1
            ls_sum += ls.item() * y.shape[0]
            n += y.shape[0]                       # y.shape[0] is num_steps * batch_size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(ls_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', rnn_predict(prefix, pred_len, model, params, init_hidden_state, num_hiddens,
                                        vocab_size, idx_to_char, char_to_idx, device))


def rnn_train_and_predict_torch(model, idx_to_char, char_to_idx, device,
                                corpus_indices, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    hidden_state = None
    for epoch in range(num_epochs):
        ls_sum, n, start = 0., 0, time.time()
        data_iter = tools.get_timeseries_data_batch_consecutive(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if hidden_state is not None:
                if isinstance(hidden_state, tuple):
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
                else:
                    hidden_state = hidden_state.detach()
            (output, hidden_state) = model(X, hidden_state)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            ls = loss(output, y.long())                 # output and y are of shape (num_steps * batch_size, vocab_size)
            optimizer.zero_grad()
            ls.backward()
            metrics.grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            ls_sum += ls.item() * y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(ls_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', rnn_predict_torch(prefix, pred_len, model, idx_to_char, char_to_idx, device))


def rnn_predict(prefix, num_chars, rnn, params, init_hidden_state, num_hiddens,
                vocab_size, idx_to_char, char_to_idx, device):
    """
    For given prefix of chars, predict the next num_chars chars.
    """
    hidden_state = init_hidden_state(batch_size=1, num_hiddens=num_hiddens, device=device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # batch_size, num_steps = 1, 1
        X = metrics.to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        (Y, hidden_state) = rnn(X, hidden_state, params)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            # notice that here Y is a list of 1 tensor, which is of size (1, vocab_size)
            # thus we need Y[0] to get this tensor and compare the value on dim 1
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[o] for o in output])


def rnn_predict_torch(prefix, num_chars, model, idx_to_char, char_to_idx, device):
    """
    For given prefix of chars, predict the next num_chars chars.
    """
    hidden_state = None
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if hidden_state is not None:
            if isinstance(hidden_state, tuple):
                hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
            else:
                hidden_state = hidden_state.to(device)
        (Y, hidden_state) = model(X, hidden_state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            # notice that here Y is of size (1, vocab_size) because both num_steps and batch_size are 1
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = tools.load_jay_lyrics(path='../data/jaychou_lyrics.txt.zip')
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

    # 1 test RNN predict
    params = get_rnn_params(num_inputs, num_hiddens, num_outputs)
    print(rnn_predict('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, idx_to_char, char_to_idx, device))

    # 1 test RNN predict (torch)
    model = RNNModel(rnn_layer(input_size=vocab_size, hidden_size=256), vocab_size=vocab_size).to(device)
    print(rnn_predict_torch('分开', 10, model, idx_to_char, char_to_idx, device))

    # 2 test RNN
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2  # attention to the lr!
    params = get_rnn_params(num_inputs, num_hiddens, num_outputs)
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    rnn_train_and_predict(rnn, params, init_rnn_state, num_hiddens, vocab_size, idx_to_char, char_to_idx,
                          device, corpus_indices, True, num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)
    # if no new params initialized, then what is trained is the same net
    params = get_rnn_params(num_inputs, num_hiddens, num_outputs)
    rnn_train_and_predict(rnn, params, init_rnn_state, num_hiddens, vocab_size, idx_to_char, char_to_idx,
                          device, corpus_indices, False, num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)

    # 2 test RNN (torch)
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e-3, 1e-2
    model = RNNModel(rnn_layer(input_size=vocab_size, hidden_size=num_hiddens), vocab_size=vocab_size).to(device)
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    rnn_train_and_predict_torch(model, idx_to_char, char_to_idx, device,
                                corpus_indices, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)

    # 3 test GRU
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    params = get_gru_params(num_inputs, num_hiddens, num_outputs)
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    rnn_train_and_predict(gru, params, init_gru_state, num_hiddens, vocab_size, idx_to_char, char_to_idx,
                          device, corpus_indices, False, num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)

    # 3 test GRU (torch)
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e-2, 1e-2    # a larger lr compared with rnn
    model = RNNModel(gru_layer(input_size=vocab_size, hidden_size=num_hiddens), vocab_size=vocab_size).to(device)
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    rnn_train_and_predict_torch(model, idx_to_char, char_to_idx, device,
                                corpus_indices, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)

    # 4 test LSTM
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    params = get_lstm_params(num_inputs, num_hiddens, num_outputs)
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    rnn_train_and_predict(lstm, params, init_lstm_state, num_hiddens, vocab_size, idx_to_char, char_to_idx,
                          device, corpus_indices, False, num_epochs, num_steps, lr, clipping_theta,
                          batch_size, pred_period, pred_len, prefixes)

    # 4 test LSTM (torch)
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e-2, 1e-2    # a larger lr compared with rnn
    model = RNNModel(lstm_layer(input_size=vocab_size, hidden_size=num_hiddens), vocab_size=vocab_size).to(device)
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    rnn_train_and_predict_torch(model, idx_to_char, char_to_idx, device,
                                corpus_indices, num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)

