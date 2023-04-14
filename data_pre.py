import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


# def build_vocab(entities_path, relations_path, binary=True):
#     """建立实体-ID表"""
#     # entities_path = '/mnt/home/linjie/projects/KG/CAFE/get_path_code/data/entities_df.csv'
#     # relations_path = '/mnt/home/linjie/projects/KG/CAFE/data/hetionet_CmC/kg_relations.txt'
#     entities_df = pd.read_csv(entities_path)
#     relations_df = pd.read_table(relations_path, names=['id', 'name'])
#     word = list(entities_df['name']) + list(relations_df['name'])
#     word = ['<pad>'] + word
#     # word = ['<pad>', '<bos>', '<eos>'] + word
#     word_ID_dict = dict(zip(word, range(len(word))))
#     ID_word_dict = dict(zip(range(len(word)), word))
#
#     if binary:
#         class_dict = {'0': 0, '1': 1}
#     else:
#         class_list = list(entities_df.loc[entities_df['metanode'] == 'Compound', 'name'])
#         class_dict = dict(zip(class_list, range(len(class_list))))
#
#     return word_ID_dict, ID_word_dict, class_dict


def build_vocab(entities_path, relations_path, binary=True):
    """建立实体-ID表"""
    # entities_path = '/mnt/home/linjie/projects/KG/CAFE/get_path_code/data/entities_df.csv'
    # relations_path = '/mnt/home/linjie/projects/KG/CAFE/data/hetionet_CmC/kg_relations.txt'
    entities_df = pd.read_csv(entities_path)
    relations_df = pd.read_table(relations_path, names=['id', 'name'])
    word = list(entities_df['name']) + list(relations_df['name'])
    word = ['<pad>'] + word
    # word = ['<pad>', '<bos>', '<eos>'] + word
    entities_ID = list(entities_df['entities_global_id']+1)
    relations_ID = list(relations_df['id']+len(entities_df)+1)
    ID = [0]+entities_ID+relations_ID
    word_ID_dict = dict(zip(word, ID))
    ID_word_dict = dict(zip(ID, word))

    if binary:
        class_dict = {'0': 0, '1': 1}
    else:
        class_list = list(entities_df.loc[entities_df['metanode'] == 'Compound', 'name'])
        class_dict = dict(zip(class_list, range(len(class_list))))

    return word_ID_dict, ID_word_dict, class_dict


class ClassificationDataset():
    def __init__(self, entities_path, relations_path, batch_size, sep='|', binary=True):
        self.word_ID_dict, self.ID_word_dict, self.class_dict = build_vocab(entities_path, relations_path, binary)
        self.pad = self.word_ID_dict['<pad>']
        self.batch_size = batch_size
        self.sep = sep


    def word2ID(self, filepath):
        """转换为Token序列"""
        # filepath = '/mnt/home/linjie/projects/KG/CAFE/transformer_torch/data/CmC_path_list.txt'
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        data = []
        max_len = 0
        for line in lines:
            line = line.split(self.sep)
            src, tgt = line[:-1], line[-1]
            src_tensor = torch.tensor([self.word_ID_dict[token] for token in src], dtype=torch.long)
            tgt_tensor = torch.tensor(self.class_dict[tgt], dtype=torch.long)
            data.append((src_tensor, tgt_tensor))
            max_len = max(max_len, src_tensor.size(0))

        return data, max_len

    def pad_batch_sequence(self, sequences):
        """padding处理"""
        # sequences = [x[0] for x in data[:5]]+[x[0] for x in data[-5:]]
        sequence_batch = pad_sequence(sequences, batch_first=True, padding_value=self.pad)
        sequences_mask = sequence_batch == 0

        return sequence_batch, sequences_mask


    def generate_batch(self, data_batch):
        """对每个batch中的数据集进行padding处理"""
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        # batch_sentence# [batch_size,max_len]
        batch_sentence, batch_pad_mask = self.pad_batch_sequence(batch_sentence)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


    def load_dataset(self, dataset_path):
        # dataset_path = '/mnt/home/linjie/projects/KG/CAFE/transformer_torch/data/CmC_path_list.txt'
        data, max_len = self.word2ID(dataset_path)
        # 构造DataLoader
        data_iter = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=self.generate_batch)
        return data_iter



if __name__ == '__main__':
    dataset_path = '/mnt/home/linjie/projects/KG/CAFE/transformer_torch/data/CmC_path_list.txt'
    entities_path = '/mnt/home/linjie/projects/KG/CAFE/get_path_code/data/entities_df.csv'
    relations_path = '/mnt/home/linjie/projects/KG/CAFE/data/hetionet_CmC/kg_relations.txt'
    batch_size = 64
    data_loader_class = ClassificationDataset(entities_path, relations_path, batch_size, sep='|')
    train_iter = data_loader_class.load_dataset(dataset_path)
    for sample, label in train_iter:
        print(sample.shape)# [seq_len,batch_size]