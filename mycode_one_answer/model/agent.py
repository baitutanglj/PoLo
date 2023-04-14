import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

class Agent(nn.Module):
    def __init__(self, params, rev_entity_vocab, rev_relation_vocab):
        super().__init__()
        self.device = params['device']
        self.rev_entity_vocab = rev_entity_vocab
        self.rev_relation_vocab = rev_relation_vocab
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.rPAD = torch.tensor(params['relation_vocab']['PAD'], dtype=torch.int32)
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.LSTM_Layers = params['LSTM_layers']
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_labels = torch.tensor(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'], device=self.device)
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.use_entity_embeddings = params['use_entity_embeddings']


        # self.action_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)
        self.relation_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)
        # self.relation_embedding_init = self.action_embedding

        # self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)
        self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)
        # self.entity_embedding_init = self.entity_embedding
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)

        self.m = 4 if self.use_entity_embeddings else 2

        # self.policy_step = nn.LSTM(input_size=self.m * self.embedding_size, hidden_size=self.m * self.hidden_size,
        #                            num_layers=self.LSTM_Layers, batch_first=True)

        self.policy_step = nn.Sequential()
        for i in range(self.LSTM_Layers):
            self.policy_step.add_module(f"LSTMCell_{i}", nn.LSTMCell(input_size=self.m * self.embedding_size, hidden_size=self.m * self.hidden_size))

        self.policy_MLP = nn.Sequential(nn.Linear(self.m * self.hidden_size*2, 4 * self.hidden_size),
                                        nn.BatchNorm1d(4 * self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(4 * self.hidden_size, self.m * self.embedding_size),
                                        nn.BatchNorm1d(self.m * self.embedding_size),
                                        nn.ReLU())

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # 初始状态s0
        self.init_states = []
        for layer in range(self.LSTM_Layers):
            states_c = torch.zeros((self.batch_size, self.m * self.embedding_size),
                                   dtype=torch.float32, device=self.device)
            states_h = torch.zeros((self.batch_size, self.m * self.embedding_size),
                                   dtype=torch.float32, device=self.device)
            self.init_states.append((states_c, states_h))

    def step(self, next_relations, next_entities, current_entities, prev_states, prev_relations, query_embeddings,
             range_arr):
        prev_action_embeddings = self.action_encoder(prev_relations, current_entities)  # al-1=[rl-1, el]#[2560,128]

        # One step of RNN
        # prev_action_embeddings = torch.unsqueeze(prev_action_embeddings, dim=1)
        # output, new_states = self.policy_step(prev_action_embeddings, prev_states)  # hl=LSTM(al-1, hl-1)#output:(2560, 128)
        new_states = []
        h_tmp = prev_action_embeddings
        for i, policy_layer in enumerate(self.policy_step):
            h_tmp, c_tmp = policy_layer(h_tmp, prev_states[i])
            new_states.append((h_tmp, c_tmp))

        output = h_tmp#torch.Size([3840, 128])
        # output = torch.squeeze(output, dim=1)
        # Get state vector
        prev_entities = self.entity_embedding(current_entities)  # el#(2560,64)
        if self.use_entity_embeddings:
            states = torch.cat([output, prev_entities], axis=-1)  # [hl;el]#shape=(2560, 192)
        else:
            states = output
        state_query_concat = torch.concat([states, query_embeddings], axis=-1)  # hl = [hl;el;rq]#shape=(2560, 256)
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)  # Al=(rl;el+1)#[2560,200,128]

        # MLP for policy
        output = self.policy_MLP(state_query_concat)  # W2(RELU(W1|[hl;el]))#[2560,128]
        output_expanded = torch.unsqueeze(output, axis=1)  # shape=(2560, 1, 128)
        output_ = candidate_action_embeddings * output_expanded#shape=(2560, 200, 128)
        prelim_scores = torch.sum(output_, axis=2)#torch.Size([2560, 200]  # dl=Al(W2(RELU(W1|[hl;el])))#[2560,200] #tf.multiply(candidate_action_embeddings, output_expanded):shape=(2560, 200, 128)

        # Masking PAD actions
        comparison_tensor = torch.ones_like(next_relations, dtype=torch.int32) * self.rPAD  # [2560,200]
        mask = torch.eq(next_relations, comparison_tensor)
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0
        scores = torch.where(mask, dummy_scores, prelim_scores)  # dl#[2560,200]

        # Sample action
        scores_probs = logits_to_probs(scores)
        m = Categorical(scores_probs)
        actions = m.sample()  # Al#[2560,]

        # Loss
        labels_actions = actions  # [2560,]
        loss = self.criterion(input=scores, target=labels_actions)  # [2560,]
        # loss = F.cross_entropy(input=scores, target=labels_actions, reduction='none)

        actions_idx = actions  # (2560,)
        chosen_relations = self.th_gather_nd(x=next_relations, coords=torch.stack([range_arr, actions_idx],dim=1))  # 根据得分选出下一步的relations#(2560,)#tf.transpose(a=tf.stack([range_arr, actions_idx])):shape=(2560, 2)
        # tf.nn.log_softmax(scores)#dl
        # return loss, F.log_softmax(scores, dim=1), new_states, actions_idx, chosen_relations
        return loss, F.log_softmax(scores, dim=1), new_states, actions_idx, chosen_relations

    def th_gather_nd(self, x, coords):
        x = x.contiguous()
        coords = coords.data.cpu()
        inds = coords.mv(torch.LongTensor(x.stride())).to(self.device)
        x_gather = torch.index_select(x.contiguous().view(-1), 0, inds)
        return x_gather

    def get_mem_shape(self):
        return self.LSTM_Layers, 2, None, self.m * self.embedding_size

    def action_encoder(self, next_relations, next_entities):
        relation_embedding = self.relation_embedding(next_relations)
        entity_embedding = self.entity_embedding(next_entities)
        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], axis=-1)
        else:
            action_embedding = relation_embedding
        return action_embedding


    def forward(self, episode, range_arr, path_length):
        episode_states = episode.get_states()
        query_relations = torch.from_numpy(episode.get_query_relations()).to(self.device)
        query_embeddings = self.relation_embedding(query_relations)

        states = self.init_states
        prev_relations = self.dummy_start_labels  # previous_relation#初始r0

        all_loss = []
        all_logits = []
        actions_idx = []
        arguments = []
        for t in range(path_length):
            next_possible_relations = torch.from_numpy(episode_states['next_relations']).to(self.device)#torch.Size([3840, 400])
            next_possible_entities = torch.from_numpy(episode_states['next_entities']).to(self.device)#torch.Size([3840, 400])
            current_entities_t = torch.from_numpy(episode_states['current_entities']).to(self.device)#torch.Size([3840])
            loss, logits, new_states, idx, chosen_relations = self.step(next_relations=next_possible_relations,
                                                                    next_entities=next_possible_entities,
                                                                    current_entities=current_entities_t,
                                                                    prev_states=states,
                                                                    prev_relations=prev_relations,
                                                                    query_embeddings=query_embeddings,
                                                                    range_arr=range_arr)
            states = new_states
            all_loss.append(loss)
            all_logits.append(logits)
            actions_idx.append(idx)
            prev_relations = chosen_relations

            rel = np.copy(episode_states['next_relations'][np.arange(episode_states['next_relations'].shape[0]), idx.data.cpu().numpy()])  # 根据actions_idx选出rel
            ent = np.copy(episode_states['next_entities'][np.arange(episode_states['next_entities'].shape[0]), idx.data.cpu().numpy()])  # 根据actions_idx选出ent
            rel_string = np.array([self.rev_relation_vocab[x] for x in rel])
            ent_string = np.array([self.rev_entity_vocab[x] for x in ent])
            arguments.append(rel_string)
            arguments.append(ent_string)
            episode_states = episode(idx.data.cpu().numpy())

        return all_loss, all_logits, actions_idx, arguments


