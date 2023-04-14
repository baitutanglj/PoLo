import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

def get_polo_data(path, CmC_pair_df, data):
    polo_df = pd.read_csv(path, header=None, sep='\t', names=['head', 'relation', 'tail'])
    polo_df = pd.merge(CmC_pair_df, polo_df, how='inner')
    polo_path_df = pd.merge(polo_df, data, on=['head', 'tail'], how='inner')
    return polo_df, polo_path_df


def get_data(data_path, save_dir):
    entities_path = "/home/linjie/projects/KG/CAFE/get_path_code/data/entities_df.csv"
    polo_train = "/home/linjie/projects/KG/PoLo/datasets/Hetionet_CmC/train.txt"
    polo_test = "/home/linjie/projects/KG/PoLo/datasets/Hetionet_CmC/test.txt"
    polo_dev = "/home/linjie/projects/KG/PoLo/datasets/Hetionet_CmC/dev.txt"
    data = pd.read_csv(data_path, header=None, names=['path'], sep="\t")
    # data['head'] = data['path'].apply(lambda x: x.split('|')[0])
    # data['tail'] = data['path'].apply(lambda x: x.split('|')[-1])
    # data['CmC_pair'] = data['head'].str.cat(data['tail'], sep='-')

    entities_df = pd.read_csv(entities_path)
    entities_df['entities_value'] = entities_df['entities_value'].apply(lambda x: x.replace(' ', '_'))
    f_graph = open(save_dir+'/graph.txt', 'w')
    CmC_pair = []
    path_list = []
    triple_set = []

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        entity_value_list = [list(entities_df.loc[entities_df['name']==name, 'entities_value'])[0] if i%2==0 else name.replace('rev_', '_') if 'rev_' in name else name for i, name in enumerate(row['path'].split('|'))]
        CmC_pair.append([entity_value_list[0], 'CmC', entity_value_list[-1]])
        path_list.append('|'.join(entity_value_list))
        for idx in np.arange(0, len(entity_value_list)-1, 2):
            triple = entity_value_list[idx:idx+3]
            triple_set.append(triple)
            rev_rel = '_'+triple[1] if '_' not in triple[1] else triple[1].replace('_', '')
            rev_triple = [triple[-1], rev_rel, triple[0]]
            triple_set.append(rev_triple)


    triple_df = pd.DataFrame(triple_set, columns=['head', 'relation', 'tail'])
    triple_df.drop_duplicates(inplace=True, ignore_index=True)
    graph_big = pd.read_csv('/home/linjie/projects/KG/PoLo/datasets/Hetionet_CmC/graph.txt', header=None, sep='\t', names=['head', 'relation', 'tail'])
    graph_big = pd.concat([graph_big, triple_df, triple_df], axis=0)
    graph_big.drop_duplicates(keep=False, inplace=True, ignore_index=True)
    triple_df.to_csv(save_dir + '/graph_small.txt', index=False, header=False, sep='\t')
    graph_big.to_csv(save_dir + '/graph_big.txt', index=False, header=False, sep='\t')


    CmC_pair_df = pd.DataFrame(CmC_pair, columns=['head', 'relation', 'tail'])
    data[['head', 'tail']] = CmC_pair_df[['head', 'tail']]
    data['path'] = path_list
    CmC_pair_df.drop_duplicates(inplace=True, ignore_index=True)

    train, train_path_df = get_polo_data(polo_train, CmC_pair_df, data)
    test, test_path_df = get_polo_data(polo_test, CmC_pair_df, data)
    dev, dev_path_df = get_polo_data(polo_dev, CmC_pair_df, data)

    train.to_csv(save_dir + '/train_CmC.txt', index=False, header=False, sep='\t')
    test.to_csv(save_dir+'/test_CmC.txt', index=False, header=False, sep='\t')
    dev.to_csv(save_dir + '/dev_CmC.txt', index=False, header=False, sep='\t')
    train_path_df.to_csv(save_dir + '/train.txt', index=False, header=False, sep='\t')
    test_path_df.to_csv(save_dir+'/test.txt', index=False, header=False, sep='\t')
    dev_path_df.to_csv(save_dir+'/dev.txt', index=False, header=False, sep='\t')



if __name__ == "__main__":
    data_path = "/home/linjie/projects/KG/CAFE/transformer_torch/data/CmC_path_list.txt"
    save_dir = "/home/linjie/projects/KG/PoLo/pretrain_model1/data"
    get_data(data_path, save_dir)
