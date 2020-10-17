"""
This module implements RNN models.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
import tools
import metrics
import numpy as np
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_params(num_inputs, num_hiddens, num_outputs):
    """
    Initialize the params for the basic RNN.
    The basic RNN is a one-hidden-layer MLP with hidden states.
    :param num_inputs: should be vocab_size
    :param num_hiddens:
    :param num_outputs: should be vocab_size
    :return:
    """
    def _normal(shape):
        return nn.Parameter(
            torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32),
            requires_grad=True
        )

    W_xh = _normal((num_inputs, num_hiddens))
    W_hh = _normal((num_hiddens, num_hiddens))
    b_h = nn.Parameter(torch.zeros(num_hiddens, device=device), requires_grad=True)
    W_hq = _normal((num_hiddens, num_outputs))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device), requires_grad=True)

    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])


def init_hidden_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, hidden_state, params):
    """
    inputs and outputs are num_steps tensors, each of size (batch_size, vocab_size), rather than one tensor of size
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
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


def rnn_layer(input_size, hidden_size):
    """
    Input X should be of shape (num_steps, batch_size, vocab_size).
    """
    return nn.RNN(input_size=input_size, hidden_size=hidden_size)


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)
        self.hidden_state = None

    def forward(self, inputs, hidden_state):
        # X is of shape (batch_size, num_steps)  ---> num_steps * (batch_size, vocab_size)
        X = metrics.to_onehot(inputs, self.vocab_size)
        Y, self.hidden_state = self.rnn(torch.stack(X), hidden_state)
        # change shape into (num_steps * batch_size, hidden_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.hidden_state


if __name__ == '__main__':
    corpus_indices, char_to_idx, idx_to_char, vocab_size = tools.load_jay_lyrics(path='../data/jaychou_lyrics.txt.zip')
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    params = get_params(num_inputs, num_hiddens, num_outputs)

    # test predict
    print(metrics.rnn_predict('分开', 10, rnn, params, init_hidden_state, num_hiddens, vocab_size, idx_to_char, char_to_idx, device))

    # test predict (torch)
    model = RNNModel(rnn_layer(input_size=vocab_size, hidden_size=256), vocab_size=vocab_size)
    print(metrics.rnn_predict_torch('分开', 10, model, idx_to_char, char_to_idx, device))

    # test rnn train and predict
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    metrics.rnn_train_and_predict(rnn, params, init_hidden_state, num_hiddens, vocab_size, idx_to_char, char_to_idx,
                                  device, corpus_indices, True, num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)
    # if no new params initialized, then what is trained is the same net
    metrics.rnn_train_and_predict(rnn, params, init_hidden_state, num_hiddens, vocab_size, idx_to_char, char_to_idx,
                                  device, corpus_indices, False, num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes)

    # test rnn train and predict (torch)
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    metrics.rnn_train_and_predict_torch(model, vocab_size, idx_to_char, char_to_idx, device,
                                        corpus_indices, num_epochs, num_steps, lr, clipping_theta,
                                        batch_size, pred_period, pred_len, prefixes)
