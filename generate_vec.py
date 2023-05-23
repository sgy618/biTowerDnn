import os
import sys
import argparse
import common
import numpy as np

from os.path import join as join_path
from sklearn.metrics import roc_auc_score

import torch
from torch.autograd import Variable
from model import CosineQtDiscriminator
from data import DataLoader
from data import DnnQtDataProcessor


def load_dict(dct_file, sort_by_freq=True, is_lower=False):
    words = []
    for line in open(dct_file):
        flds = line.rstrip('\n').split('\t')
        if len(flds) == 2:
            word, count = flds
        else:
            word, count = flds[0], 1
        if is_lower:
            word = word.lower()
        words.append((word, int(count)))
    if sort_by_freq:
        words = sorted(words, key=itemgetter(1), reverse=True)
    dct = dict()
    dct['<unk>'] = common.UNK_IDX
    dct['<pad>'] = common.PAD_IDX
    count = common.PAD_IDX + 1
    for w, _ in words:
        if w in ('<unk>', '<pad>'):
            continue
        if w in dct:
            continue
        dct[w] = count
        count += 1

    dct_i2w = {}
    for k, v in dct.items():
        dct_i2w[v] = k.strip()

    return dct, dct_i2w


if __name__ == '__main__':

    DATA_DIR = './data'
    dct, dct_i2w = load_dict(join_path(DATA_DIR, 'vocab.txt'), sort_by_freq=False)

    test_data_processor1 = DnnQtDataProcessor(dct)
    test_data = [
        'train.txt'
        ]

    test_loader = []
    for x in test_data:
        test_loader.append(DataLoader(join_path(DATA_DIR, x), test_data_processor1, batch_size=64))

    vocab_size = len(dct)
    print("vocab_size: ", vocab_size)

    model = CosineQtDiscriminator(dct, word_embed_dim=128, h1_size=64)
    model.eval()

    CKPT_MODEL_PATH = './output/two_tower_dnn_model_epoch9_auc0.9603.chkpt'
    model.load_state_dict(torch.load(CKPT_MODEL_PATH)['model'])

    for loader in test_loader:
        all_labels = []
        all_preds  = []
        for data in loader.batch():
            query, title, labels = data
            text = query

            with torch.no_grad():
                query = Variable(query)
                title = Variable(title)

            qvec = model.query_vector(query)
            vecs = qvec.data.cpu().numpy().tolist()

            for i in range(len(vecs)):
                vec = ' '.join(map(str, vecs[i]))
                word = ''
                for w in text[i].numpy():
                    if w in dct_i2w and w !=0 and w != 1:
                        word += dct_i2w[w]
                print(word, vec)

