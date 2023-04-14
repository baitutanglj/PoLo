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

from data_pre import ClassificationDataset


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=8):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class MyTransformerClassification(nn.Module):

    def __init__(self, d_model=128, num_emb=45211, output_dim=1540,
                 nhead=8, num_encoder_layers=2, dim_feedforward=512,
                 dropout=0.1, pad=0, max_len=8, use_pretrain_embeds=True,
                 embed_requires_grad=True, entities_emb_path=None, relations_emb_path=None):
        super(MyTransformerClassification, self).__init__()
        self.pad = pad
        self.max_len = max_len
        # self.entities_emb_path = entities_emb_path
        # self.relations_emb_path = relations_emb_path
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0, max_len=self.max_len)
        # 定义transformer_encoder。
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model//2),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropout),
                                     nn.Linear(d_model//2, output_dim))

        # 初始化权重
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        ## 初始化embedding
        # 定义词向量，词典数为45213。
        self.embedding = nn.Embedding(num_embeddings=num_emb, embedding_dim=d_model, padding_idx=self.pad)
        if use_pretrain_embeds:
            self.pretrain_embeds = self.map_embedding(entities_emb_path, relations_emb_path)
            self.embedding.weight.data.copy_(self.pretrain_embeds)
            self.embedding.weight.requires_grad = embed_requires_grad
    #     self.init_weights()
    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def map_embedding(self, entities_emb_path, relations_emb_path):
        # entities_emb_path = "/home/linjie/projects/KG/hetionet_data/ckpts/TransR_hetionet_0/hetionet_TransR_entity.npy"
        # relations_emb_path = "/home/linjie/projects/KG/hetionet_data/ckpts/TransR_hetionet_0/hetionet_TransR_relation.npy"
        entities_emb = np.load(entities_emb_path)
        relations_emb = np.load(relations_emb_path)
        pad_emb = np.zeros((1, entities_emb.shape[1]))
        emb = np.concatenate((pad_emb, entities_emb, relations_emb, relations_emb), axis=0)
        emb_tensor = torch.from_numpy(emb)

        return emb_tensor

    def forward(self, src, src_key_padding_mask, concat_type='last', test_mode=False):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len,  ntoken]
        """

        # 对src和tgt进行编码
        src = self.embedding(src)  # [src_len, batch_size, embed_dim]
        # 给src的token增加位置信息
        src = self.positional_encoding(src)  # [src_len, batch_size, embed_dim]

        # 将准备好的数据送给transformer
        memory = self.transformer_encoder(src=src,
                                          src_key_padding_mask=src_key_padding_mask)  # [src_len,batch_size,embed_dim]
        if concat_type == 'sum':
            memory = torch.sum(memory, dim=0)
        elif concat_type == 'avg':
            memory = torch.sum(memory, dim=0) / memory.size(0)
        elif concat_type == 'last':
            memory = memory[:, -1, :]  # 取最后一个时刻
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        output = self.decoder(memory)  # 输出logits # [batch_size, num_class]
        # if test_mode:
        #     output = nn.Softmax(output)

        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def evaluate(model, test_loader, criterion, device=0, concat_type='last', binary_type=True):
    model.eval()
    total_loss, total_acc = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx, (src, label) in enumerate(test_loader):
            src, label = src.to(device), label.to(device)
            src_pad_mask = src == model.pad
            output = model(src, src_pad_mask, concat_type)
            loss = criterion(output, label)
            total_loss += loss.item()
            pred = output.argmax(1)
            total_acc += (output.argmax(1) == label).float().mean()
            y_pred.extend(list(pred.data.cpu().numpy()))
            y_true.extend(list(label.data.cpu().numpy()))
    avg_type = 'binary' if binary_type else 'macro'

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=avg_type)
    f1 = f1_score(y_true, y_pred, average=avg_type)
    test_loss = total_loss / len(test_loader)
    test_acc = total_acc / len(test_loader)
    return test_loss, test_acc, acc, recall, f1


def run_train(epochs, model, save_dir, train_loader, valid_loader, test_loader=None, device=0, concat_type='last', binary_type=True):
    log_dir = save_dir + "/log"
    model_dir = save_dir + "/model"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        shutil.rmtree(log_dir)
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
    writer = SummaryWriter(save_dir + '/log')
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 3e-4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_model = None
    # min_loss = float("inf")
    max_acc = float("-inf")
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        acc, recall, f1 = 0, 0, 0
        for idx, (src, label) in enumerate(train_loader):
            src, label = src.to(device), label.to(device)
            src_pad_mask = src == model.pad
            # label = torch.unsqueeze(label, dim=1)
            # 进行transformer的计算
            output = model(src, src_pad_mask, concat_type)
            pred = output.argmax(1)
            if np.isnan(output[0, 0].data.cpu().numpy()):
                print(idx)
            """
            计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                    我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                    (batch_size*词数, 词典大小)。
                    而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                    除以n_tokens。
            """
            optimizer.zero_grad()  # 清空梯度
            loss = criterion(output, label)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()
            acc += (output.argmax(1) == label).float().mean()
            y_pred = pred.data.cpu().numpy()
            y_true = label.data.cpu().numpy()
            recall += recall_score(y_true, y_pred)
            f1 += f1_score(y_true, y_pred)

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = acc / len(train_loader)
        epoch_recall = recall / len(train_loader)
        epoch_f1 = f1 / len(train_loader)

        # test_loss, _, test_acc, test_recall, test_f1 = evaluate(model, valid_loader, criterion, device, concat_type,
        #                                                         binary_type)
        test_loss, _, test_acc, test_recall, test_f1 = evaluate(model, valid_loader, criterion, device, concat_type,
                                                                binary_type)
        writer.add_scalars("loss", {"train": epoch_loss, "valid": test_loss}, epoch)
        writer.add_scalars("accuracy", {"train": epoch_acc, "valid": test_acc}, epoch)
        writer.add_scalars("recall", {"train": epoch_recall, "valid": test_recall}, epoch)
        writer.add_scalars("F1", {"train": epoch_f1, "valid": test_f1}, epoch)
        # if test_loss < min_loss:
        #     min_loss = test_loss
        #     best_model = model
        #     best_epoch = epoch
        # print(f"Epoch: {epoch}, Train loss :{epoch_loss:.3f}, Train acc: {epoch_acc:.3f}"
        #       f" || Test loss :{test_loss:.3f}, Test acc: {test_acc:.3f} "
        #       f"Test recall: {test_recall:.3f} Test F1: {test_f1:.3f}")

        # if test_acc > max_acc and epoch>0:
        #     max_acc = test_acc
        #     best_model = model
        #     best_epoch = epoch
        best_model = model
        best_epoch = epoch
        print(f"Epoch: {epoch}, Train loss :{epoch_loss:.3f}, Train acc: {epoch_acc:.3f}, "
              f"Train recall: {epoch_recall:.3f}, Train f1: {epoch_f1:.3f}")
        print(f"Valid loss :{test_loss:.3f}, Valid acc: {test_acc:.3f} "
              f"Valid recall: {test_recall:.3f} Valid F1: {test_f1:.3f}")
        print('*'*10)

    test_loss, _, test_acc, test_recall, test_f1 = evaluate(model, test_loader, criterion, device, concat_type,
                                                            binary_type)
    print(f"Test loss :{test_loss:.3f}, Test acc: {test_acc:.3f} "
          f"Test recall: {test_recall:.3f} Test F1: {test_f1:.3f}")
    print('--'*10)

    best_model_path = f"{save_dir}/model/best_model_{str(best_epoch)}.pt"
    torch.save(best_model, best_model_path)

    # model_path = f"{save_dir}/model/model_{str(epoch)}.pt"
    # torch.save(model, model_path)


def add_args():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="train data path")
    parser.add_argument("--test_path", type=str, help="test data path")
    parser.add_argument("--valid_path", type=str, help="valid data path")
    parser.add_argument("--entities_path", type=str, help="entities_df data path",
                        default='data/entities_df.csv')
    parser.add_argument("--relations_path", type=str, help="relations_df data path",
                        default='data/kg_relations.txt')
    parser.add_argument("--entities_emb_path", type=str, help="entities_emb.npy file path when use_pretrain_embeds==True",
                        default="data/TransR_hetionet_0/hetionet_TransR_entity.npy")
    parser.add_argument("--relations_emb_path", type=str, help="relations_emb.npy file path when use_pretrain_embeds==True",
                        default="data/TransR_hetionet_0/hetionet_TransR_relation.npy")
    parser.add_argument("--save_dir", type=str, help="result save dir", required=True)
    parser.add_argument("--binary_type", type=boolean, help="classification type", default=True)
    parser.add_argument("--use_pretrain_embeds", type=boolean, help="use pretrain embedding", default=False)
    parser.add_argument("--embed_requires_grad", type=boolean, help="embedding requires_grad", default=True)
    parser.add_argument("--concat_type", type=str, choices=["sum", "avg" "last"], help="use pretrain embedding", default="last")
    parser.add_argument("--device", type=int, help="gpu index", default=0)
    parser.add_argument("--batch_size", type=int, help="model train batch_size", default=1024)
    parser.add_argument("--epochs", type=int, help="model train epochs", default=20)
    parser.add_argument("--max_len", type=int, help="path max length", default=9)
    parser.add_argument("--nhead", type=int, help="attention model num head", default=5)
    parser.add_argument("--d_model", type=int, help="hidden size", default=100)
    parser.add_argument("--output_dim", type=int, help="output size", default=2)

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
    with open(args.save_dir+'/args.json', 'w') as f:
        f.write(json.dumps(args.__dict__))

    return args


if __name__ == "__main__":
    args = add_args()
    data_loader_class = ClassificationDataset(args.entities_path, args.relations_path, args.batch_size, sep='|', binary=args.binary_type)
    pad = data_loader_class.pad
    train_loader = data_loader_class.load_dataset(args.train_path)
    test_loader = data_loader_class.load_dataset(args.test_path)
    # valid_loader = data_loader_class.load_dataset(args.valid_path)
    valid_loader = test_loader
    model = MyTransformerClassification(d_model=args.d_model, num_emb=45211, output_dim=args.output_dim,
                                        nhead=args.nhead, num_encoder_layers=2, dim_feedforward=512,
                                        dropout=0.3, pad=pad, max_len=args.max_len,
                                        use_pretrain_embeds=args.use_pretrain_embeds,embed_requires_grad = args.embed_requires_grad,
                                        entities_emb_path=args.entities_emb_path, relations_emb_path=args.relations_emb_path)
    print(model)
    model = model.to(args.device)
    run_train(epochs=args.epochs, model=model, save_dir=args.save_dir,
              train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
              device=args.device, concat_type=args.concat_type, binary_type=args.binary_type)


