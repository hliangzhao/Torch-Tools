"""
This module gives a test case on using linear regression to predict the house price.
"""
import torch
from torch import nn, optim
import pandas as pd
import tools


def read_data(train_data_path, test_data_path):
    # train_data is of size (1460, 81) and test_data is of size (1459, 80)
    return pd.read_csv(train_data_path), pd.read_csv(test_data_path)


def preprocess(train_data, test_data):
    """
    Here data is of size (sample_num, feature_num).
    """
    # the first column is id, which should not be used
    # the last column of train_data is house price, which should be viewed as label
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # string of dtypes 'object'
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # standardization
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    # NaN is set as zero (mean value)
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # convert categorical variable into dummy/indicator variables
    all_features = pd.get_dummies(all_features, dummy_na=False)

    # change features and labels into np array
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
    train_labels = torch.tensor(train_data[train_data.columns[-1]].values, dtype=torch.float).view(-1, 1)
    print(train_features.shape, test_features.shape, '\n', train_labels)
    return train_features, test_features, train_labels


def log_rmse(net, features, labels):
    """
    Use logarithmic mean square error to evaluate the model.
    Gradient should not be tracked!
    """
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, epoch_num, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epoch_num):
        for X, y in tools.get_data_batch_torch(batch_size, train_features, train_labels):
            ls = loss(net(X.float()).float(), y.float())
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, test_features))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold(k, i, X, y):
    """
    Get k-fold training and evaluation dataset.
    The i-th fold is for evaluation, the others are for training.
    """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx, :]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train, y_train = torch.cat((X_train, X_part), dim=0), torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold_train(k, X_train, y_train, epoch_num, lr, weight_decay, batch_size):
    pass


if __name__ == '__main__':
    train_data, test_data = read_data('../data/kaggle_house/train.csv', '../data/kaggle_house/test.csv')
    preprocess(train_data, test_data)
    loss = nn.MSELoss()
    k, num_epochs, lr, weight_decay, batch_size = 5, 250, 5, 0, 64

