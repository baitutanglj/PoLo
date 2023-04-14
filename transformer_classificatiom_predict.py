import argparse
import json
import os
from glob import glob

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from transformer_classification import MyTransformerClassification, PositionalEncoding
# from transformer_model import MyTransformerClassification, PositionalEncoding
from data_pre import ClassificationDataset

def evaluate(model, test_loader, ID_to_word, args):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, total_acc = 0, 0
    y_true, y_pred, y_score = [], [], []
    src_word, head_entity, tail_entity, CmC_pair, group_hit = [], [], [], [], []
    with torch.no_grad():
        for idx, (src, label) in enumerate(test_loader):
            src, label = src.to(args.device), label.to(args.device)
            src_pad_mask = src == model.pad
            output = model(src, src_pad_mask, args.concat_type)
            output_logits = F.softmax(output, dim=1)
            pre_scores = output_logits[:, 1]
            pred_label = torch.max(output_logits, 1).indices
            loss = criterion(output, label)
            total_loss += loss.item()
            total_acc += (output.argmax(1) == label).float().mean()
            y_pred.extend(list(pred_label.data.cpu().numpy()))
            y_true.extend(list(label.data.cpu().numpy()))
            y_score.extend(list(pre_scores.data.cpu().numpy()))

            for senc in src.data.cpu().numpy():
                tmp_word = '|'.join([ID_to_word[i] for i in senc if i != 0])
                src_word.append(tmp_word)
                head_entity.append(tmp_word.split('|', 1)[0])
                tail_entity.append(tmp_word.rsplit('|', 1)[1])
                CmC_pair.append('|'.join([tmp_word.split('|', 1)[0], tmp_word.rsplit('|', 1)[1]]))

    df = pd.DataFrame({'head_entity': head_entity, 'tail_entity': tail_entity,
                       'CmC_pair': CmC_pair, 'input': src_word, 'true_label': y_true,
                       'pred_label': y_pred, 'pred_score': y_score})
    df['flag'] = df['true_label'] + df['pred_label']
    # df.to_csv(os.path.join(args.save_dir, 'predict.csv'), index=False, sep='\t\t\t')
    print("drop_duplicates")
    metric_df_drop_duplicates = get_hit(df, y_true, y_pred, total_loss, total_acc, drop_duplicates=True)
    print("**no drop_duplicates**")
    metric_df = get_hit(df, y_true, y_pred, total_loss, total_acc, drop_duplicates=False)
    return metric_df_drop_duplicates


def get_hit(df, y_true, y_pred, total_loss, total_acc, drop_duplicates=True):
    old_model_hit_num = 0
    hit1_num, hit3_num, hit10_num, mrr = 0, 0, 0, 0
    df_group = df.groupby(by='head_entity')
    # df_group = df.groupby(by='CmC_pair')
    data_sort_df = pd.DataFrame(columns=['input', 'true_label', 'pred_label', 'pred_score'])
    for head_entity, group in df_group:
        group.sort_values(by=['pred_score', 'true_label'], ascending=[False, False], inplace=True, ignore_index=True)
        if drop_duplicates:
            group.drop_duplicates(subset=['CmC_pair'], keep='first', inplace=True, ignore_index=True)
        old_model_hit = 1 if sum(group['true_label']) > 0 else 0
        # if old_model_hit == 1:
        data_sort_df = data_sort_df.append(group[['input', 'true_label', 'pred_label', 'pred_score']])
        old_model_hit_num = old_model_hit_num + 1
        hit1_num = hit1_num + 1 if group.loc[0, 'flag'] == 2 else hit1_num
        hit3_num = hit3_num + 1 if sum(group.loc[:2, 'flag'] == 2) > 0 else hit3_num
        hit10_num = hit10_num + 1 if sum(group.loc[:9, 'flag'] == 2) > 0 else hit10_num
        mrr = mrr + 1 / (group.loc[group['true_label'] == 1].index.values[0] + 1) if sum(
            group.loc[:, 'true_label'] == 1) > 0 else mrr

    hit1 = hit1_num / len(df_group)
    hit3 = hit3_num / len(df_group)
    hit10 = hit10_num / len(df_group)
    mrr = mrr / len(df_group)

    avg_type = 'binary' if args.binary_type else 'macro'
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=avg_type)
    f1 = f1_score(y_true, y_pred, average=avg_type)
    auc = roc_auc_score(y_true, y_pred)
    test_loss = total_loss / len(test_loader)
    test_acc = total_acc / len(test_loader)
    metric_df_acc = pd.DataFrame({'loss': test_loss, 'acc': acc, 'recall': recall, 'f1': f1}, index=[0]).round(3)
    metric_df = pd.DataFrame({'hit1': hit1, 'hit3': hit3, 'hit10': hit10,
                              'hit1_num': hit1_num, 'hit3_num': hit3_num, 'hit10_num': hit10_num,
                              'mrr': mrr, 'old_model_hit_num': old_model_hit_num}, index=[0]).round(3)
    if drop_duplicates:
        metric_df_acc.to_csv(args.save_dir + '/transformer_metric_acc_drop_duplicates.csv', index=False)
        metric_df.to_csv(args.save_dir + '/transformer_metric_drop_duplicates.csv', index=False)
        data_sort_df.to_csv(args.save_dir + '/group_sort_drop_duplicates.csv', sep='\t')
    else:
        metric_df_acc.to_csv(args.save_dir + '/transformer_metric_acc.csv', index=False)
        metric_df.to_csv(args.save_dir + '/transformer_metric.csv', index=False)
        data_sort_df.to_csv(args.save_dir + '/group_sort.csv', sep='\t')
    print(metric_df)
    print(metric_df_acc)
    return metric_df


def add_args():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_path", type=str, help="test data path",
                        default="data/test_binary.csv")
    parser.add_argument("--hetiones_path", type=str, help="hetiones graph path",
                        default="./data/df_triples.csv")
    parser.add_argument("--config_file", type=str, help="train argparse config file",
                        default="output/args.json")
    parser.add_argument("--device", type=int, help="gpu index", default=0)
    parser.add_argument("--batch_size", type=int, help="model train batch_size", default=1024)
    parser.add_argument("--parent_model", type=str, help="parent model name", default="polo_model")
    parser.add_argument("--save_dir", type=str, help="save_dir", default=None)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    for k, v in config.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.model_path = glob(os.path.dirname(args.config_file) + "/model/*.pt")[0]
    args.save_dir = os.path.join(args.save_dir, args.parent_model + "_predict") if args.save_dir is None else args.save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args


if __name__ == "__main__":
    args = add_args()
    data_loader_class = ClassificationDataset(args.entities_path, args.relations_path, args.batch_size, sep='|',
                                              binary=args.binary_type)
    pad = data_loader_class.pad
    ID_to_word = data_loader_class.ID_word_dict
    test_loader = data_loader_class.load_dataset(args.test_path)
    model = torch.load(args.model_path)
    print(model)
    model = model.to(args.device)
    metric_dict = evaluate(model, test_loader, ID_to_word, args)
