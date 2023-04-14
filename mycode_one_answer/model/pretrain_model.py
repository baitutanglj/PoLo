import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs


class lstm_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.entity_vocab_size = args.entity_vocab_size
        self.action_vocab_size = args.action_vocab_size
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.LSTM_Layers = args.LSTM_layers
        self.batch_size = args.batch_size
        self.use_entity_embeddings = args.use_entity_embeddings
        self.m = 4 if self.use_entity_embeddings else 2
        self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)
        self.relation_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)

        nn.init.xavier_uniform_(self.entity_embedding.weight)

        self.policy_step = nn.Sequential()
        for i in range(self.LSTM_Layers):
            self.policy_step.add_module(f"LSTMCell_{i}", nn.LSTMCell(input_size=self.m * self.embedding_size, hidden_size=self.m * self.hidden_size))

        self.policy_MLP = nn.Sequential(nn.Linear(self.m * self.hidden_size, 4 * self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(4 * self.hidden_size, 2),
                                        nn.ReLU())

        # self.policy_MLP = nn.Sequential(nn.Linear(self.m * self.hidden_size * 2, 4 * self.hidden_size),
        #                                 nn.ReLU(),
        #                                 nn.Linear(4 * self.hidden_size, 2),
        #                                 nn.ReLU())

        # self.sigmoid = nn.Sigmoid()

    def init_state(self):
        # 初始状态s0
        init_states = []
        for layer in range(self.LSTM_Layers):
            states_c = torch.zeros((self.batch_size, self.m * self.embedding_size),
                                   dtype=torch.float32, device=self.device)
            states_h = torch.zeros((self.batch_size, self.m * self.embedding_size),
                                   dtype=torch.float32, device=self.device)
            init_states.append((states_c, states_h))
        return init_states

    def action_encoder(self, next_relations, next_entities):
        relation_embedding = self.relation_embedding(next_relations)
        entity_embedding = self.entity_embedding(next_entities)
        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], axis=-1)
        else:
            action_embedding = relation_embedding
        return action_embedding

    def step(self, prev_relations, current_entities, prev_states):
        # One step of RNN
        prev_action_embeddings = self.action_encoder(prev_relations, current_entities) # al-1=[rl-1, el]#[2560,128]
        new_states = []
        h_tmp = prev_action_embeddings
        for i, policy_layer in enumerate(self.policy_step):# hl=LSTM(al-1, hl-1)#output:(2560, 128)
            h_tmp, c_tmp = policy_layer(prev_action_embeddings, prev_states[i])
            new_states.append((h_tmp, c_tmp))
        output = h_tmp# torch.Size([2560, 128])

        return new_states, output

    def forward(self, entity_batch, relation_batch, query_relations, data_length):
        #entity_batch:torch.Size([32, 5])
        path_length = entity_batch.shape[1]
        query_embeddings = self.relation_embedding(query_relations)#torch.Size([32, 64])
        #LSTM
        prev_states = self.init_state()#[(torch.Size([3840, 128]), torch.Size([3840, 128])),  (torch.Size([3840, 128]), torch.Size([3840, 128]))]
        output, current_entities = torch.zeros(entity_batch.shape[0]), torch.zeros(entity_batch.shape[0])
        output_list = []
        for entity_idx in range(path_length):
            current_entities = entity_batch[:, entity_idx]#torch.Size([2560])
            prev_relations = relation_batch[:, entity_idx]#torch.Size([2560])
            new_states, output = self.step(prev_relations, current_entities, prev_states)
            prev_states = new_states

            output_list.append(output)
        output_list = torch.stack(output_list, dim=1)
        output_tmp = []
        for idx, i in enumerate(data_length):
            output_tmp.append(output_list[idx, i-1, :])
        output = torch.stack(output_tmp)

        # Get state vector ## hl = [hl;el;rq]#shape=(2560, 256)
        # prev_entities = self.entity_embedding(current_entities)  # el#(2560,64)
        # if self.use_entity_embeddings:
        #     states = torch.cat([output, prev_entities], axis=-1)  # [hl;el]#shape=(2560, 192)
        # else:
        #     states = output
        # state_query_concat = torch.concat([states, query_embeddings], axis=-1)#shape=(2560, 256)

        # MLP for policy
        output = self.policy_MLP(output)
        # output = self.policy_MLP(state_query_concat)#torch.Size([2560, 1])
        output = torch.squeeze(output)##torch.Size([2560])
        # output = self.sigmoid(output)
        return output
