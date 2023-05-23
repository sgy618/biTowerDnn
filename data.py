from __future__ import print_function
import os
import sys
import jieba
import random
from os.path import join as join_path
from abc import ABCMeta, abstractmethod
import gzip
import torch
import common

def read_and_shuffle(filepath, shuffle=False):
    lines = []
    if filepath.endswith('gz') or filepath.endswith('gzip'): f = gzip.open(filepath, 'rb')
    else:
        f = open(filepath, 'r', encoding='gbk', errors='ignore')
    for line in f:
        lines.append(line.rstrip())

    if shuffle:
        random.shuffle(lines)
    return lines


def _batch_data_to_tensor(
        batch_data,
        batch_max_lens,
        id_weight_seq_fields,
        label_fields = [],
        shuffle_batch=False):

    for any_len in batch_max_lens:
        if any_len == 0:
            log_str = "invalid batch_max_len: %d" % any_len
            print(log_str)

    batch_data = list(batch_data)
    batch_size = len(batch_data)

    output = [None] * (len(id_weight_seq_fields) * 1 + len(label_fields))

    offset = 0

    for i in range(len(id_weight_seq_fields)):
        batch_max_len = batch_max_lens[i + offset]

        ids = torch.LongTensor(batch_size, batch_max_len)
        ids.fill_(common.PAD_IDX)
        output[i * 1 + offset] = ids

    offset += len(id_weight_seq_fields) * 1

    for i in range(len(label_fields)):
        labels = torch.zeros(batch_size)
        output[i + offset] = labels

    offset += len(label_fields)

    if shuffle_batch:
        random.shuffle(batch_data)

    for i, sample in enumerate(batch_data):
        offset = 0

        for field_i, field in enumerate(id_weight_seq_fields):
            ids = sample[field]
            seq_len = len(ids)
            output[field_i * 1 + offset][i, 0:seq_len] = torch.LongTensor(ids)

        offset += len(id_weight_seq_fields) * 1

        for field_i, field in enumerate(label_fields):
            label = sample[field]
            output[field_i + offset][i] = label

        offset += len(label_fields)

    return tuple(output)


def elementwise_max(left, right):
    new = []
    for i in range(len(left)):
        new.append(max(left[i], right[i]))
    return new


class DataProcessor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def parse(self, line_iterator):
        pass

    @abstractmethod
    def to_tensor(self, batch_data):
        pass

    @abstractmethod
    def batch_finish(self):
        pass


class DnnQtDataProcessor(DataProcessor):
    def __init__(self, dct):
        self.dct = dct
        self.batch_max_len = [0, 0]

    def sample_len(self, sample):
        return len(sample)
        #return len(sample[0])

    def parse_text(self, text):
        ids = []
        for w in text.split(' '):
        #for w in [x for x in jieba.cut(text)]:
            ids.append(self.dct.get(w.strip(), common.UNK_IDX))
        return ids

    def parse(self, line_iterator):
        for line in line_iterator:
            output = []
            columns = line.rstrip().split('\t')
            if len(columns) < 3:
                continue

            query_info = self.parse_text(columns[0])
            title_info = self.parse_text(columns[1])

            output.append(query_info)
            output.append(title_info)
            output.append(int(columns[2]) > 0)

            current_max_len = []
            for ll in map(self.sample_len, output[:2]):
                current_max_len.append(ll)

            self.batch_max_len = elementwise_max(current_max_len, self.batch_max_len)
            yield output

    def batch_finish(self):
        self.batch_max_len = [0, 0]

    def to_tensor(self, batch_data):
        return _batch_data_to_tensor(batch_data, self.batch_max_len, [0, 1], label_fields=[2])


class DataLoader(object):
    def __init__(self,
                 data_path,
                 data_processor,
                 batch_size=200,
                 shuffle=False):

        if os.path.isdir(data_path):
            self.files = [
                join_path(data_path, i) for i in os.listdir(data_path)
            ]
        else:
            self.files = [data_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_processor = data_processor

    def batch(self):
        return self.sync_batch()

    def sync_batch(self):
        count = 0
        batch_data = []
        for f in self.files:
            for parse_res in self.data_processor.parse(
                    read_and_shuffle(f, self.shuffle)):
                count += 1
                batch_data.append(parse_res)
                if count == self.batch_size:
                    batch_tensor = self.data_processor.to_tensor(batch_data)
                    yield batch_tensor
                    count = 0
                    batch_data = []
                    self.data_processor.batch_finish()
            if batch_data:
                batch_tensor = self.data_processor.to_tensor(batch_data)
                yield batch_tensor


