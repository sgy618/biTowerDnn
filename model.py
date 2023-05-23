import os
import sys
import common

import numpy as np

import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable

def cosine_similarity(x1, x2, eps=1e-12):
    w12 = torch.sum(x1 * x2, 1)
    w1 = torch.norm(x1, 2, 1)
    w2 = torch.norm(x2, 2, 1)
    return (w12 / (w1 * w2 + eps)).unsqueeze(1)

def rank_loss(left, right, label):
    pred_diff = left - right
    loss = torch.log1p(torch.exp(pred_diff)) - label * pred_diff
    return loss.sum()

#def triplet_loss(left, right, margin):
    #return torch.clamp(left - right + margin, min=0)

def triplet_loss(dist_pos, dist_neg, margin=1.0):
    """  triplet loss """
    hinge = torch.clamp(margin + dist_pos - dist_neg, min=0.0)
    loss = torch.mean(hinge)
    return loss

def embedding_sum(embeds, weights=None):
    if weights is None or isinstance(weights, ConstantList):
        embeds_sum = embeds.sum(1).squeeze(1)
    else:
        embeds_sum = (
                embeds * weights.unsqueeze(2).expand_as(embeds)).sum(1).squeeze(1)
    return embeds_sum

class CosineQtDiscriminator(nn.Module):
    def __init__(self, dct, word_embed_dim, h1_size, dropout=0.0,
            init_word_embedding='', is_predict=False):
        super(CosineQtDiscriminator, self).__init__()

        self.word_embed_dim = word_embed_dim

        vocab_size = len(dct)
        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=common.PAD_IDX)

        if not is_predict and init_word_embedding:
            self.init_word_embedding(init_word_embedding, dct)

        self.drop_layer = nn.Dropout(dropout)

        self.query_hidden = nn.Sequential(
            nn.Linear(word_embed_dim, h1_size),
            nn.Tanh(),
        )
        self.title_hidden = nn.Sequential(
            nn.Linear(word_embed_dim, h1_size),
            nn.Tanh(),
        )

    def init_word_embedding(self, file, dct):
        count = 0
        for line in open(file):
            fs = line.strip().split()
            word = fs[0]
            vector = fs[1:]
            if len(vector) != self.word_embed_dim:
                log_str = 'init vector dim %d not equals word_embed_dim %d' % (len(vector), self.word_embed_dim)
                print(log_str)
                continue
            if word in dct:
                self.embedding.weight.data[dct[word]].copy_(torch.from_numpy(np.array(vector, dtype='float32')))
                count += 1

        log_str = 'init {} words from the embedding file'.format(count)
        print(log_str)

    def query_vector(self, tokens):
        embeds = self.embedding(tokens)
        embeds = self.drop_layer(embeds)
        embeds = embedding_sum(embeds)
        #embeds = nn.functional.tanh(embeds)
        hidden = self.query_hidden(embeds)
        return hidden

    def title_vector(self, tokens):
        embeds = self.embedding(tokens)
        embeds = self.drop_layer(embeds)
        embeds = embedding_sum(embeds)
        #embeds = nn.functional.tanh(embeds)
        hidden = self.title_hidden(embeds)
        return hidden

    def forward(self, query_tokens, title_tokens):
        query_vec = self.query_vector(query_tokens)
        title_vec = self.title_vector(title_tokens)
        qt_cosine = cosine_similarity(query_vec, title_vec)

        pred_score = (qt_cosine + 1) / 2
        return pred_score


class CosineQtTrain(object):
    def __init__(self, dct, word_embed_dim, h1_size, num_epochs, learning_rate=1e-3,
            dropout=0.0, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
            callback=None, init_word_embedding='', warm_start_file=None, is_cuda=False):

        self.word_embed_dim = word_embed_dim
        self.h1_size = h1_size
        self.num_epochs = num_epochs

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.weight_decay = weight_decay
        self.callback = callback

        self.dm = CosineQtDiscriminator(dct, word_embed_dim, h1_size, dropout=dropout,
                init_word_embedding=init_word_embedding)

        if is_cuda:
            self.dm.cuda()

        if warm_start_file is not None:
            self.dm.load_state_dict(torch.load(warm_start_file)['model'])

    @property
    def params():
        pass

    def train(self, dataloader, is_cuda=False):
        dm_optimizer = torch.optim.Adam(filter(lambda x:x.requires_grad, self.dm.parameters()),
                lr=self.learning_rate, betas=(self.beta1, self.beta2),
                eps=self.eps, weight_decay=self.weight_decay)

        criterion = nn.BCELoss()

        for epoch in range(1, self.num_epochs + 1):
            print("epoch: ", epoch)
            for (batch_id, data) in enumerate(dataloader.batch(), 1):
                # data
                query, title, label = data
                if is_cuda:
                    query, title, label = map(lambda x: x.cuda(), [query, title, label])

                data_num = query.size()[0]

                # query, title, label = map(Variable, [query, title, label])

                # initialize
                self.dm.zero_grad()

                # forward
                pred_score = self.dm(query, title).squeeze()

                # backward
                loss = criterion(pred_score, label)
                loss.backward()
                dm_optimizer.step()

                # test
                #loss_val = loss.data[0]
                loss_val = loss.data
                if self.callback:
                    info_dict = {
                            'd_loss': loss_val,
                            'epoch': epoch,
                            'batch_id': batch_id
                            }
                    self.callback(self, info_dict)
        return


