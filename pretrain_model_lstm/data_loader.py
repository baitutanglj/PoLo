import csv
import argparse
import random
import numpy as np
from collections import defaultdict
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class RelationEntityGrapher(object):
    def __init__(self, triple_store, entity_vocab, relation_vocab, max_num_actions=None, store=None):
        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.triple_store = triple_store
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.store = defaultdict(list) if store is None else store
        self.max_num_actions = max_num_actions
        self.masked_array_store = None
        random.seed(123)
        self.rev_entity_vocab = dict([(v, k) for k, v in entity_vocab.items()])
        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.create_graph()
        print("KG constructed.")

    def create_graph(self):
        with open(self.triple_store) as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                e1 = self.entity_vocab[line[0]]
                r = self.relation_vocab[line[1]]
                e2 = self.entity_vocab[line[2]]
                self.store[e1].append((r, e2))


        self.max_num_actions = max([len(s) for s in self.store.values()]) if self.max_num_actions is None else self.max_num_actions
        print(f"max_num_actions:{self.max_num_actions}")
        self.array_store = torch.ones((len(self.entity_vocab), self.max_num_actions, 2), dtype=torch.int32)
        self.array_store[:, :, 0] *= self.ePAD
        self.array_store[:, :, 1] *= self.rPAD

        for k, value in self.store.items():
            value = value[:self.max_num_actions]
            random.shuffle(value)
            self.store[k] = value

        for e1 in tqdm(self.store, total=len(self.store)):
            self.array_store[e1, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[e1, 0, 0] = e1
            num_actions = 1
            for r, e2 in self.store[e1]:
                if num_actions == self.array_store.shape[1]:
                    break
                self.array_store[e1, num_actions, 0] = e2
                self.array_store[e1, num_actions, 1] = r
                num_actions += 1
        # del self.store
        # self.store = None


class RelationEntityBatcher(object):
    def __init__(self, input_dir, entity_vocab, relation_vocab, grapher, mode="train"):
        self.ePAD = entity_vocab['PAD']
        self.rPAD = relation_vocab['PAD']
        self.input_dir = input_dir
        self.input_file = input_dir+'{}.txt'.format(mode)
        print('Reading vocab...')
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.mode = mode
        self.grapher = grapher
        self.e1_set = set()
        self.create_triple_store(self.input_file)
        print("Batcher loaded.")


    def create_triple_store(self, input_file):
        self.store_all_correct = defaultdict(set)
        self.store = []
        path = []
        fact_files = ['train', 'dev'] if self.mode=='train' else ['test']
        for f in fact_files:
            with open(self.input_dir + f + '.txt') as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter='\t')
                for line in csv_file:
                    p = [self.entity_vocab[l] if idx%2==0 else self.relation_vocab[l] for idx, l in enumerate(line[3].split('|'))]
                    p = [p[i:i + 3] for i in range(0, len(p)-2, 2)]
                    path.append(torch.tensor(p))
                    self.e1_set.add(p[0][0])
                self.e1_set = torch.tensor(list(self.e1_set))
                self.map_graph_action(path)
                if self.mode != 'train':
                    for line in csv_file:
                        e1 = line[0]
                        r = line[1]
                        e2 = line[2]
                        if (e1 in self.entity_vocab) and (e2 in self.entity_vocab):
                            e1 = self.entity_vocab[e1]
                            r = self.relation_vocab[r]
                            e2 = self.entity_vocab[e2]
                            self.store.append([e1, r, e2])
                    self.store = torch.tensor(self.store)

                # fact_files = ['train', 'test', 'dev']
                # for f in fact_files:
                #     with open(self.input_dir + f + '.txt') as raw_input:
                #         csv_file = csv.reader(raw_input, delimiter='\t')
                #         for line in csv_file:
                #             e1 = line[0]
                #             r = line[1]
                #             e2 = line[2]
                #             if (e1 in self.entity_vocab) and (e2 in self.entity_vocab):
                #                 e1 = self.entity_vocab[e1]
                #                 r = self.relation_vocab[r]
                #                 e2 = self.entity_vocab[e2]
                #                 self.store_all_correct[e1].add(e2)



    def map_graph_action(self, path):
        self.action = []
        self.path = []
        for idx, p in enumerate(path):
            path_action_tmp = []
            for triple in p:
                e1_action = self.grapher.array_store[triple[0]]
                action_idx = torch.where(torch.all(e1_action==torch.tensor([triple[2], triple[1]]), axis=1))[0]
                if action_idx.shape!=torch.Size([0]):
                    path_action_tmp.append(action_idx)
                else:
                    path_action_tmp = []
                    break
            if len(path_action_tmp)>0:
                try:
                    self.path.append(p)
                    self.action.append(torch.stack(path_action_tmp, axis=1).squeeze())
                except:
                    print(path_action_tmp)

class MyDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.max_num_actions = None if args.max_num_actions is None else args.max_num_actions
        args.relation_vocab = json.load(open(args.vocab_dir + 'relation_vocab.json'))
        args.entity_vocab = json.load(open(args.vocab_dir + 'entity_vocab.json'))
        self.grapher_small = RelationEntityGrapher(triple_store=args.input_dir + 'graph_small.txt',
                                                   entity_vocab=args.entity_vocab,
                                                   relation_vocab=args.relation_vocab,
                                                   max_num_actions=self.max_num_actions)
        self.grapher = RelationEntityGrapher(triple_store=args.input_dir + 'graph_big.txt',
                                             entity_vocab=args.entity_vocab,
                                             relation_vocab=args.relation_vocab,
                                             max_num_actions=self.grapher_small.max_num_actions,
                                             store=self.grapher_small.store)
        # self.grapher = self.grapher_small
        args.max_num_actions = self.grapher.max_num_actions
        self.data = RelationEntityBatcher(input_dir=args.input_dir,
                                          entity_vocab=args.entity_vocab,
                                          relation_vocab=args.relation_vocab,
                                          grapher=self.grapher,
                                          mode=mode)

    def __getitem__(self, index):
        x = self.data.path[index]
        y = self.data.action[index]
        return x, y

    def __len__(self):
        return len(self.data.path)


    # def __getitem__(self, index):
    #     if self.mode == "train":
    #         x = self.data.path[index]
    #         y = self.data.action[index]
    #         return x, y
    #     else:
    #         x = self.data.e1_set[index]
    #         el = int(x.data.numpy().item())
    #         y = list(self.data.store_all_correct[el])
    #         return x

    # def __len__(self):
    #     if self.mode == "train":
    #         return len(self.data.path)
    #     else:
    #         return len(self.data.e1_set)


def mycollate_fn(data):
    #这里的data是getittem返回的（x，y）的二元组，总共有batch_size个
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(x[0]) for x in data]
    x_batch = [i[0] for i in data]
    y_batch = [i[1] for i in data]
    x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0)
    y_batch = pad_sequence(y_batch, batch_first=True, padding_value=0)
    return x_batch, y_batch, data_length

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input_dir",
                        default="/home/linjie/projects/KG/PoLo/pretrain_model1/data/")
    parser.add_argument("--save_dir", type=str, help="result save dir",
                        default="/home/linjie/projects/KG/PoLo/pretrain_model1/result")
    parser.add_argument("--vocab_dir", type=str, help="entity_vocab",
                        default="/home/linjie/projects/KG/PoLo/datasets/Hetionet_CmC/vocab/")
    parser.add_argument("--device", type=int, help="gpu index", default=0)
    parser.add_argument("--batch_size", type=int, help="model train batch_size", default=1024)
    parser.add_argument("--path_length", type=int, help="path_len", default=4)
    parser.add_argument("--max_num_actions", type=int, help="max_num_actions", default=400)
    parser.add_argument("--query_relations", type=str, help="query relations",default="CmC")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = add_args()
    train_data = MyDataset(args, mode="train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=mycollate_fn)
    for x, y, data_length in train_loader:
        print('x:', x.shape)
        print('y:', y.shape)

