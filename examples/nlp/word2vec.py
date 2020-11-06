"""
This module implements the word2vec model. It implements the training and using of embedding vectors.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import collections
import math
import random
import time
import torch
from torch import nn, optim
import torch.utils.data as Data
import tools
import metrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def import_dataset(path='../../data/ptb/ptb.train.txt'):
    """
    In this dataset of sentences:
        - the last token in one sentence: 'eos'
        - number appeared: 'N'
        - unusual token: 'unk'
    """
    with open(path) as f:
        lines = f.readlines()
        raw_dataset = [sentence.split() for sentence in lines]
        # raw_dataset is a list of list, where the inner list's elements are the tokens of one sentence
        print('#sentences: %d' % len(raw_dataset))
        print('#tokens in the first sentence: %d' % len(raw_dataset[0]))
    return raw_dataset


def set_token_idx(raw_dataset):
    counter = collections.Counter([tk for st in raw_dataset for tk in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

    idx2tk = [tk for tk, _ in counter.items()]                # a list with element being token (use list idx to get it)
    tk2idx = {tk: idx for idx, tk in enumerate(idx2tk)}       # a dict with element {tk: idx}
    # same shape with raw_dataset, but element is replaced by idx
    dataset = [[tk2idx[tk] for tk in st if tk in tk2idx] for st in raw_dataset]
    num_tokens = sum([len(st) for st in dataset])
    print('#tokens in the preprocessed dataset: %d' % num_tokens)
    return counter, idx2tk, tk2idx, dataset, num_tokens


def discard(idx, counter, idx2tk, num_tokens):
    """
    Discard a token or not, decided by the frequency of this token.
    """
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx2tk[idx]] * num_tokens)


def subsampling(dataset, counter, idx2tk, num_tokens):
    """
    Call discard() to sub-sampling the dataset.
    Subsampling is to remove very highly frequent tokens, such as 'the', 'in', 'a', etc.
    """
    return [[tk for tk in st if not discard(tk, counter, idx2tk, num_tokens)] for st in dataset]


def compare_cts(tk, tk2idx, dataset, subsampled_dataset):
    idx = tk2idx[tk]
    return '#\'%s\': before = %d, after = %d' % \
           (tk, sum([st.count(idx) for st in dataset]), sum([st.count(idx) for st in subsampled_dataset]))


def get_centers_contexts(input_dataset, max_window_size):
    """
    Get all center tokens and corresponding context tokens from the subsampled dataset.
    """
    centers, contexts = [], []
    for st in input_dataset:
        if len(st) < 2:
            # at least 2 tokens to formulate the 'contexts - center - contexts'
            continue
        centers += st     # equals to extend (a list of tokens)
        for center_idx in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_idx - window_size), min(len(st), center_idx + 1 + window_size)))
            indices.remove(center_idx)
            contexts.append([st[idx] for idx in indices])      # a list of lists, the inner list's elements are context tokens
    return centers, contexts


def neg_sampling(all_contexts, idx2tk, counter, K):
    """
    Negative sampling.
    """
    # the negative sampling distribution P(w) is set as the frequency^0.75.
    sampling_weights = [counter[tk] ** 0.75 for tk in idx2tk]

    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        # sample K negative tokens for each 'center-contexts' pair
        while len(negatives) < len(contexts) * K:
            # the if is True if current sampling results have all been processed but still not enough
            if i == len(neg_candidates):
                i = 0
                # randomly sample 1e5 tokens from population according to their weights
                # set k larger to avoid repeating sampling
                neg_candidates = random.choices(population, sampling_weights, k=int(1e5))
            neg = neg_candidates[i]
            i = i + 1
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


class FinalDataset(Data.Dataset):
    """
    This class is the final dataset ready-for-use.
    To get this dataset, we need import raw dataset ---> preprocess ---> get centers, contexts, and negatives.
    """
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx], self.negatives[idx]

    def __len__(self):
        return len(self.centers)


def batchify(f_data):
    """
    This is the collate function to get data batch of given final_dataset.
    This function regulates how one sample is defined.
    """
    # max_len is the max length of 'each context + each negative'
    max_len = max(len(c) + len(n) for _, c, n in f_data)
    # label is 1 if its context token, otherwise 0
    # masks is used to distinguish the pending part
    centers, contexts_and_negatives, masks, labels = [], [], [], []
    for center, context, negative in f_data:
        cur_len = len(context) + len(negative)
        centers += [center]
        # pad out to max_len with 0
        contexts_and_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return torch.tensor(centers).view(-1, 1), torch.tensor(contexts_and_negatives), torch.tensor(masks), torch.tensor(labels)


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """
    Define the forward of skip gram.
    :param center: of size (batch_size, 1)
    :param contexts_and_negatives: of size (batch_size, max_len)
    :param embed_v: nn.Embedding(), of size (num_tokens, embedding_dim)
    :param embed_u: nn.Embedding(), of size (num_tokens, embedding_dim)
    :return: the inner product of center vector and each context vector
    """
    v = embed_v(center)                           # (batch_size, 1, embedding_dim)
    u = embed_u(contexts_and_negatives)           # (batch_size, max_len, embedding_dim)
    return torch.bmm(v, u.permute(0, 2, 1))       # (batch_size, 1, max_len)


class SigmoidBinCEL(nn.Module):
    """
    Implement the Sigmoid Binary Cross Entropy Loss.
    """
    def __init__(self):
        super(SigmoidBinCEL, self).__init__()

    def forward(self, preds, targets, mask=None):
        """
        How it works:

        pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
        label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])

        def sigmoid(x):
            return - math.log(1 / (1 + math.exp(-x)))

        loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1)
        is equal to
        [(sigmoid(1.5) + sigmoid(-0.3) + sigmoid(1) + sigmoid(-2)) / 4, (sigmoid(1.1) + sigmoid(-0.6) + sigmoid(-2.2)) / 3]
        """
        preds, targets, mask = preds.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction='none', weight=mask)
        return res.sum(dim=1)


def embed(idx2tk, embed_size):
    """
    The net to implement word2vec.
    """
    return nn.Sequential(
        nn.Embedding(num_embeddings=len(idx2tk), embedding_dim=embed_size),    # center vector
        nn.Embedding(num_embeddings=len(idx2tk), embedding_dim=embed_size)     # context vector
    )


def train(net, lr, num_epochs, loss, data_iter):
    print('train on', device)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, ls_sum, batch_idx = time.time(), 0., 0
        for batch in data_iter:
            # (batch_size, 1), (batch_size, max_len), (batch_size, max_len), (batch_size, max_len)
            center, contexts_and_negatives, mask, label = [d.to(device) for d in batch]

            # (batch_size, 1, max_len)
            pred = skip_gram(center, contexts_and_negatives, net[0], net[1])

            ls = (loss(pred.view(label.shape), label, mask) / mask.float().sum(dim=1)).mean()
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            ls_sum += ls.cpu().item()
            batch_idx += 1
        print('epoch %d, loss %.2f, time %.2fs' % (epoch + 1, ls_sum / batch_idx, time.time() - start))


def get_similar_tokens(query_tk, k, tk2idx, idx2tk, embed_v):
    # (num_tokens, embedding_dim)
    W = embed_v.weight.data
    # (1, embedding_dim)
    x = W[tk2idx[query_tk]]
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, top_k = torch.topk(cos, k + 1)
    top_k = top_k.cpu().numpy()
    for i in top_k[1:]:
        print('cosine similarity = %.3f: %s' % (cos[i].item(), (idx2tk[i])))


if __name__ == '__main__':
    # 1. import dataset and preprocess
    raw_dataset = import_dataset()
    for sentence in raw_dataset[:3]:
        print('#tokens:', len(sentence), '-- the first 5 tokens:', sentence[:5])
    counter, idx2tk, tk2idx, dataset, num_tokens = set_token_idx(raw_dataset)
    subsampled_dataset = subsampling(dataset, counter, idx2tk, num_tokens)
    print(compare_cts('the', tk2idx, dataset, subsampled_dataset))

    # 2. get the center word and context word
    toy_dataset = [list(range(7)), list(range(7, 10))]
    for center, context in zip(*get_centers_contexts(toy_dataset, 2)):
        print('center:', center, 'has contexts:', context)
    all_centers, all_contexts = get_centers_contexts(subsampled_dataset, max_window_size=5)
    all_negatives = neg_sampling(all_contexts, idx2tk, counter, K=5)

    # 3. get batchified data iter
    final_dataset = FinalDataset(all_centers, all_contexts, all_negatives)
    data_iter = Data.DataLoader(final_dataset, batch_size=512, shuffle=True, collate_fn=batchify, num_workers=4)
    for batch in data_iter:
        for name, data in zip(['centers', 'contexts and negatives', 'masks', 'labels'], batch):
            print(name, 'shape:', data.shape)
        break

    # test SigmoidBinCEL
    pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
    label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    loss = SigmoidBinCEL()
    print(loss(pred, label, mask) / mask.float().sum(dim=1))

    # 4. train the skip gram model and use it to get similar tokens
    embed_size = 100
    net = embed(idx2tk, embed_size)
    loss = SigmoidBinCEL()
    train(net, 0.01, 2, loss, data_iter)
    get_similar_tokens('chip', 3, tk2idx, idx2tk, net[0])
