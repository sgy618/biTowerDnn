import os
import sys
import argparse
import common
import numpy as np

from os.path import join as join_path
from sklearn.metrics import roc_auc_score

import torch
from torch.autograd import Variable
from model import CosineQtTrain
from data import DataLoader
from data import DnnQtDataProcessor


def load_dict(dct_file, sort_by_freq=True, is_lower=False):
    words = []
    for line in open(dct_file, 'r', encoding='gbk', errors='ignore'):
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
    return dct


class DnnQtCallback(object):
    def __init__(self, model_name=None, save_dir=None,
            print_every=100, loaders=[], target_test_data_idx=0, auc_thres=0.5):
        if model_name is None:
            self.model_name = 'model'
        else:
            self.model_name = model_name

        self.save_dir      = save_dir
        self.print_every   = print_every
        self.loaders       = loaders
        self.target_test_data_idx = target_test_data_idx
        self.auc_thres            = auc_thres

        self.d_loss        = []

        self.batch_num  = 0
        self.last_epoch = 0
        self.best_auc   = 0.0

    def _calc_auc(self, dm):
        dm.eval()
        res = []
        for loader in self.loaders:
            all_labels = []
            all_preds  = []
            for data in loader.batch():
                query, title, labels = data

                with torch.no_grad():
                    query = Variable(query)
                    title = Variable(title)

                all_labels += labels.numpy().tolist()
                preds = dm(query, title)
                #print("label: ", labels)
                #print("preds: ", preds)
                #sys.exit()
                preds = preds[:, -1]
                preds = preds.data.cpu().numpy().tolist()
                all_preds += preds
            auc = roc_auc_score(all_labels, all_preds)
            res.append(auc)
        dm.train()
        return res

    def __call__(self, model, info_dict):
        epoch = info_dict['epoch']

        self.d_loss.append(info_dict['d_loss'])

        self.batch_num += 1
        if self.batch_num % self.print_every == 0:
            log_str = '%s batch: %d, avg d_loss: %f' % \
                    (self.model_name, self.batch_num, np.mean(self.d_loss))
            print(log_str)

            if self.batch_num % (self.print_every * 5) == 0:
                auc = self._calc_auc(model.dm)
                log_str = '%s batch: %d, test AUC: %s' % \
                        (self.model_name, self.batch_num, '\t'.join(map(str, auc)))
                print(log_str)

                if auc[self.target_test_data_idx] > self.auc_thres and \
                        auc[self.target_test_data_idx] > self.best_auc:
                    checkpoint = {
                        'model': model.dm.state_dict(),
                        'epoch': epoch,
                        'batch': self.batch_num,
                    }
                    model_name = '%s_epoch%d_auc%.4f.chkpt' % \
                            (self.model_name, epoch, auc[self.target_test_data_idx])
                    model_path = join_path(self.save_dir, model_name)
                    print("save model %s", model_path)
                    torch.save(checkpoint, model_path)
                    self.best_auc = auc[self.target_test_data_idx]


if __name__ == '__main__':

    batch_size = 100

    random_seed = 53113
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        torch.cuda.manual_seed(random_seed)

    DATA_DIR = './data'

    dct = load_dict(join_path(DATA_DIR, 'vocab.txt'), sort_by_freq=False)

    train_data_processor = DnnQtDataProcessor(dct)
    train_loader = DataLoader(join_path(DATA_DIR, 'train.txt'), train_data_processor, batch_size=batch_size, shuffle=True)

    test_data_processor1 = DnnQtDataProcessor(dct)
    #test_data_processor2 = DnnQtDataProcessor(dct)

    test_data = [
        'test.txt',
        'test.forTrain.txt'
        ]

    test_loader = []
    for x in test_data:
        test_loader.append(DataLoader(join_path(DATA_DIR, x), test_data_processor1, batch_size=batch_size))
        #test_loader.append(DataLoader(join_path(DATA_DIR, x), test_data_processor2, batch_size=batch_size))

    SAVE_DIR = './output'
    init_word_embedding = ''
    #init_word_embedding = join_path(DATA_DIR, config['data']['init_word_embedding'])


    # CKPT_MODEL_PATH = './output/two_tower_dnn_model_epoch1_auc0.7777.chkpt'
    CKPT_MODEL_PATH = './output/two_tower_dnn_model_epoch1_auc0.6696.chkpt'
    model = CosineQtTrain(
            dct,
            word_embed_dim = 128,
            h1_size = 64,
            num_epochs = 10,
            callback=DnnQtCallback(
                model_name="two_tower_dnn_model",
                save_dir=SAVE_DIR,
                print_every=100,
                loaders=test_loader
                ),
            init_word_embedding=init_word_embedding,
            warm_start_file=CKPT_MODEL_PATH,
            is_cuda=USE_CUDA
            )

    model.train(train_loader, is_cuda=USE_CUDA)

