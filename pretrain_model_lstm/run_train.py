import argparse
import math
import os
import shutil
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_loader import MyDataset, mycollate_fn
from pretrain_model import lstm_model
from environment import Env



def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input_dir",
                        default="/home/linjie/projects/KG/PoLo/pretrain_model1/data/")
    parser.add_argument("--save_dir", type=str, help="result save dir",
                        default="/home/linjie/projects/KG/PoLo/pretrain_model1/result")
    parser.add_argument("--vocab_dir", type=str, help="entity_vocab",
                        default="/home/linjie/projects/KG/PoLo/datasets/Hetionet_CmC/vocab/")
    parser.add_argument("--device", type=int, help="gpu index", default=0)
    parser.add_argument("--batch_size", type=int, help="model train batch_size", default=2048)
    parser.add_argument("--epochs", type=int, help="model train epochs", default=50)
    parser.add_argument("--hidden_size", type=int, help="model hidden_size", default=32)
    parser.add_argument("--embedding_size", type=int, help="model embedding_size", default=32)
    parser.add_argument("--LSTM_layers", type=int, help="model LSTM_layers", default=2)
    parser.add_argument("--path_length", type=int, help="path_len", default=4)
    parser.add_argument("--num_rollouts", type=int, help="num_rollouts", default=30)
    parser.add_argument("--test_rollouts", type=int, help="test_rollouts", default=100)
    parser.add_argument("--sample_topk_list", type=int, help="test beam search sample_topk", default=[20,1,1,1])
    parser.add_argument("--max_num_actions", type=int, help="max_num_actions", default=400)
    parser.add_argument('--use_entity_embeddings', default=1, type=int)
    parser.add_argument("--lr", type=float, help="learning_rate", default=0.01)
    parser.add_argument("--query_relations", type=str, help="query relations", default="CmC")
    parser.add_argument('--eval_every', default=10, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
    with open(args.save_dir+'/args.json', 'w') as f:
        f.write(json.dumps(args.__dict__))

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_data = MyDataset(args, mode="train")
        self.train_loader = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True, collate_fn=mycollate_fn)
        self.test_data = MyDataset(args, mode="test")
        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size, shuffle=False, collate_fn=mycollate_fn)
        self.model = lstm_model(self.args)
        self.model = self.model.to(self.args.device)
        print(self.model)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.99)#学习率learning_rate应用指数衰减
        self.early_stopping = False
        self.create_dir()

    def create_dir(self):
        self.log_dir = self.args.save_dir + "/log"
        self.model_dir = self.args.save_dir + "/model"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        else:
            shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else:
            shutil.rmtree(self.model_dir)
            os.makedirs(self.model_dir)
        self.writer = SummaryWriter(self.log_dir)

    def run_training(self):
        best_model = None
        best_epoch = 0
        max_acc = float("-inf")
        for epoch in range(self.args.epochs):
            train_loss, train_acc = self.train_epoch()
            print(f"epoch:{epoch} train loss={train_loss}, train acc={train_acc}")
            if epoch % self.args.eval_every == 0:
                test_loss, test_acc = self.test()
                print(f"epoch:{epoch} test loss={test_loss}, test acc={test_acc}")
                self.writer.add_scalars("loss", {"train": train_loss, "test": test_loss}, epoch)

            if train_acc > max_acc:
                best_model = self.model
                best_epoch = epoch
                max_acc = train_acc

        best_model_path = f"{self.model_dir}/best_model_{str(best_epoch)}.pt"
        torch.save(best_model, best_model_path)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        acc = 0
        data_num = 0
        for x_batch, y_batch, data_length in self.train_loader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            y_batch = torch.reshape(y_batch, [-1])
            mask = torch.unsqueeze((y_batch != 0).int(), dim=1)  # torch.Size([4096, 1])
            output, predict_action = self.model(x_batch, mask, self.train_data.grapher)#torch.Size([4096, 501])
            self.optimizer.zero_grad()  # 清空梯度
            loss = self.criterion(output, y_batch)
            loss.backward()  # 计算梯度
            self.optimizer.step()  # 更新参数
            total_loss += loss.item()
            acc += sum(torch.eq(predict_action, y_batch)*torch.squeeze(mask))
            # pre_action_list.extend(list((torch.eq(predict_action, y_batch)*torch.squeeze(mask)).data.cpu().numpy()))
            data_num += (sum(mask)).item()#43784# 加上dev：48900
        # print('pre_action_list', len(pre_action_list))

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = acc / data_num

        return epoch_loss, epoch_acc

    def test(self):
        self.model.eval()
        total_loss = 0
        acc = 0
        data_num = 0
        pre_action_list = []
        with torch.no_grad():
            for x_batch, y_batch, data_length in self.test_loader:
                x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
                y_batch = torch.reshape(y_batch, [-1])
                mask = torch.unsqueeze((y_batch != 0).int(), dim=1)  # torch.Size([4096, 1])
                output, predict_action = self.model(x_batch, mask, self.train_data.grapher)  # torch.Size([4096, 501])
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()
                acc += sum(torch.eq(predict_action, y_batch) * torch.squeeze(mask))
                data_num += (sum(mask)).item()#11720
                # pre_action_list.extend(list((torch.eq(predict_action, y_batch) * torch.squeeze(mask)).data.cpu().numpy()))
            # print('pre_action_list', len(pre_action_list))
            test_loss = total_loss / len(self.test_loader)
            test_acc = acc / data_num
        return test_loss, test_acc




if __name__ == "__main__":
    args = add_args()
    trainer = Trainer(args)
    trainer.run_training()