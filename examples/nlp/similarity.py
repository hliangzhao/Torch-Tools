"""
This module implements the methods to get similar tokens and analogous tokens.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import torch
import torchtext.vocab as Vocab


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_pretrained_vec(name='6B', dim=50):
    """
    Get pretrained embedding vectors.
    """
    # this dir is out of this project
    return Vocab.GloVe(name, dim, cache='../../data/pretrained_glove')


def knn(W, x, k):
    """
    Get k-highest similarities.
    :param W: (num_tokens, embedding_dim)
    :param x: (1, embedding_dim)
    :param k:
    :return:
    """
    cos = torch.matmul(W, x.view((-1,))) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, top_k = torch.topk(cos, k + 1)
    top_k = top_k.cpu().numpy()
    return top_k, [cos[i].item() for i in top_k]


def get_similar_tokens(query_tk, k, embed):
    top_k, cos = knn(embed.vectors, embed.vectors[embed.stoi[query_tk]], k)
    for i, c in zip(top_k[1:], cos[1:]):
        print('cosine sim = %.3f: %s' % (c, (embed.itos[i])))


def get_analogous_tokens(tk_a, tk_b, tk_c, embed):
    vecs = [embed.vectors[embed.stoi[t]] for t in [tk_a, tk_b, tk_c]]
    x = vecs[2] + vecs[1] - vecs[0]
    top_k, cos = knn(embed.vectors, x, 1)
    print(embed.itos[top_k[0]])


if __name__ == '__main__':
    # 1. import pretrained glove
    print(Vocab.pretrained_aliases.keys())
    glove = get_pretrained_vec()
    print(len(glove.stoi))
    print(glove.stoi['beautiful'], glove.itos[3366], glove.vectors[3366])

    # 2. test similarity
    get_similar_tokens('chip', 3, glove)
    get_similar_tokens('beautiful', 3, glove)
    get_similar_tokens('study', 3, glove)

    # 3. test analogy
    get_analogous_tokens('man', 'woman', 'son', glove)
    get_analogous_tokens('beijing', 'china', 'tokyo', glove)
    get_analogous_tokens('bad', 'worst', 'big', glove)
