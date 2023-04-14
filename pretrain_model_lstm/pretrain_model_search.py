import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs


class lstm_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.entity_vocab_size = len(args.entity_vocab)
        self.action_vocab_size = len(args.relation_vocab)
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.LSTM_Layers = args.LSTM_layers
        self.batch_size = args.batch_size
        self.use_entity_embeddings = args.use_entity_embeddings
        self.max_num_actions = args.max_num_actions
        self.m = 4 if self.use_entity_embeddings else 2
        self.rPAD = torch.tensor(args.relation_vocab['PAD'], dtype=torch.int32)
        self.sample_topk_list = args.sample_topk_list
        self.sample_length = len(self.sample_topk_list)
        self.test_rollouts = args.test_rollouts
        self.entity_embedding = nn.Embedding(self.entity_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)
        self.relation_embedding = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size, padding_idx=0, device=self.device)

        nn.init.xavier_uniform_(self.entity_embedding.weight)

        self.policy_step = nn.Sequential()
        for i in range(self.LSTM_Layers):
            self.policy_step.add_module(f"LSTMCell_{i}", nn.LSTMCell(input_size=self.m * self.embedding_size, hidden_size=self.m * self.hidden_size))

        self.policy_MLP = nn.Sequential(nn.Linear(self.m * self.hidden_size*2, 4 * self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(4 * self.hidden_size, self.m * self.embedding_size),
                                        nn.ReLU())
        # self.policy_MLP = nn.Sequential(nn.Linear(self.m * self.hidden_size, 4 * self.hidden_size),
        #                                 nn.ReLU(),
        #                                 nn.Linear(4 * self.hidden_size, self.max_num_actions),
        #                                 nn.ReLU())

        # self.sigmoid = nn.Sigmoid()

    def init_state(self, batch_num):
        # 初始状态s0
        init_states = []
        for layer in range(self.LSTM_Layers):
            states_c = torch.zeros((batch_num, self.m * self.embedding_size),
                                   dtype=torch.float32, device=self.device)
            states_h = torch.zeros((batch_num, self.m * self.embedding_size),
                                   dtype=torch.float32, device=self.device)
            init_states.append((states_c, states_h))
        return init_states

    def action_encoder(self, next_relations, next_entities):
        relation_embedding = self.relation_embedding(next_relations)#torch.Size([1024, 501, 64])
        entity_embedding = self.entity_embedding(next_entities)#torch.Size([1024, 501, 64])
        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], axis=-1)#torch.Size([1024, 501, 128])
        else:
            action_embedding = relation_embedding
        return action_embedding


    # def step(self, prev_relations, current_entities, next_relations, next_entities, prev_states, query_embeddings):
    #     prev_action_embeddings = self.action_encoder(prev_relations, current_entities)  # al-1=[rl-1, el]#[2560,128]
    #
    #     # One step of RNN
    #     # prev_action_embeddings = torch.unsqueeze(prev_action_embeddings, dim=1)
    #     # output, new_states = self.policy_step(prev_action_embeddings, prev_states)  # hl=LSTM(al-1, hl-1)#output:(2560, 128)
    #     new_states = []
    #     h_tmp = prev_action_embeddings
    #     for i, policy_layer in enumerate(self.policy_step):
    #         h_tmp, c_tmp = policy_layer(h_tmp, prev_states[i])
    #         new_states.append((h_tmp, c_tmp))
    #
    #     output = h_tmp#torch.Size([1024, 128])
    #     # MLP for policy
    #     scores = self.policy_MLP(output) #torch.Size([1024, 501])
    #
    #     # Masking PAD actions
    #     comparison_tensor = torch.ones_like(next_relations, dtype=torch.int32) * self.rPAD  # [2560,200]
    #     mask = torch.eq(next_relations, comparison_tensor)#torch.Size([1024, 501])
    #     dummy_scores = torch.ones_like(scores) * -99999.0#dummy_scores
    #     scores = torch.where(mask, dummy_scores, scores)  # dl#torch.Size([1024, 501])
    #     actions = torch.argmax(F.softmax(scores, dim=-1), dim=1)
    #
    #     return new_states, actions, scores

    def step(self, prev_relations, current_entities, next_relations, next_entities, prev_states, query_embeddings):
        prev_action_embeddings = self.action_encoder(prev_relations, current_entities)  # al-1=[rl-1, el]#[2560,128]

        # One step of RNN
        # prev_action_embeddings = torch.unsqueeze(prev_action_embeddings, dim=1)
        # output, new_states = self.policy_step(prev_action_embeddings, prev_states)  # hl=LSTM(al-1, hl-1)#output:(2560, 128)
        new_states = []
        h_tmp = prev_action_embeddings
        for i, policy_layer in enumerate(self.policy_step):
            h_tmp, c_tmp = policy_layer(h_tmp, prev_states[i])
            new_states.append((h_tmp, c_tmp))

        output = h_tmp#torch.Size([1024, 128])
        # output = torch.squeeze(output, dim=1)
        # Get state vector
        prev_entities = self.entity_embedding(current_entities)  # el#torch.Size([1024, 64])
        if self.use_entity_embeddings:
            states = torch.cat([output, prev_entities], axis=-1)  # [hl;el]#torch.Size([1024, 192])
        else:
            states = output
        state_query_concat = torch.concat([states, query_embeddings], axis=-1)  # hl = [hl;el;rq]# torch.Size([1024, 256])
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)  # Al=(rl;el+1)#torch.Size([1024, 501, 128])

        # MLP for policy
        output = self.policy_MLP(state_query_concat)  # W2(RELU(W1|[hl;el]))#torch.Size([1024, 128])
        output_expanded = torch.unsqueeze(output, axis=1)  # torch.Size([1024, 1, 128])
        output_ = candidate_action_embeddings * output_expanded#torch.Size([1024, 501, 128])
        prelim_scores = torch.sum(output_, axis=2)#torch.Size([1024, 501])  # dl=Al(W2(RELU(W1|[hl;el]))) #tf.multiply(candidate_action_embeddings, output_expanded):shape=(1024, 501, 128)

        # Masking PAD actions
        comparison_tensor = torch.ones_like(next_relations, dtype=torch.int32) * self.rPAD  # [2560,200]
        mask = torch.eq(next_relations, comparison_tensor)#torch.Size([1024, 501])
        dummy_scores = torch.ones_like(prelim_scores) * -99999.0#dummy_scores
        scores = torch.where(mask, dummy_scores, prelim_scores)  # dl#torch.Size([1024, 501])
        actions = torch.argmax(F.softmax(scores, dim=-1), dim=1)

        # Sample action
        # scores_probs = logits_to_probs(scores)
        # m = Categorical(scores_probs)
        # actions = m.sample()  # Al#torch.Size([1024])

        return new_states, actions, scores


    def forward(self, x_batch, mask, grapher):
        path_length = x_batch.shape[1]
        batch_num = x_batch.shape[0]
        query_relations = (torch.tensor(self.args.relation_vocab[self.args.query_relations], device=self.device)).repeat(batch_num)#torch.Size([1024])
        query_embeddings = self.relation_embedding(query_relations)#torch.Size([1024, 64])  #
        prev_states = self.init_state(batch_num)  # [(torch.Size([1024, 128]), torch.Size([1024, 128])),  (torch.Size([1024, 128]), torch.Size([1024, 128]))]
        prev_relations = torch.tensor(
            np.ones(batch_num, dtype='int64') * self.args.relation_vocab['DUMMY_START_RELATION'],
            device=self.device)
        output_list = []
        action_list = []
        for path_idx in range(path_length):
            current_entities = x_batch[:, path_idx, 0]  # torch.Size([1024])
            # next_relations = x_batch[:, path_idx, 1]
            # next_entities = x_batch[:, path_idx, 2]
            next_relations = grapher.array_store[current_entities, :, 1].to(self.device)#torch.Size([1024, 501])
            next_entities = grapher.array_store[current_entities, :, 0].to(self.device)#torch.Size([1024, 501])
            new_states, actions, scores = self.step(prev_relations, current_entities, next_relations, next_entities, prev_states, query_embeddings)

            prev_states = new_states
            prev_relations = x_batch[:, path_idx, 1]  # torch.Size([1024])
            output_list.append(scores)#scores:torch.Size([1024, 501])
            action_list.append(actions)
        output = torch.stack(output_list, dim=1)#torch.Size([1024, 4, 501])
        output = torch.reshape(output, [-1, self.max_num_actions])#torch.Size([4096, 501])
        output = output * mask

        predict_action = torch.stack(action_list, dim=1)  # pre_actions:torch.Size([1024])
        predict_action = torch.reshape(predict_action, [-1])  # torch.Size([4096])
        predict_action = predict_action * torch.squeeze(mask)

        return output, predict_action


    def sample_forward(self, x_batch, grapher):
        x_batch = torch.repeat_interleave(x_batch, self.test_rollouts, dim=0)
        path_length = self.sample_length
        batch_num = x_batch.shape[0]
        beam_probs = torch.zeros((batch_num, 1), device=self.device)
        query_relations = (
            torch.tensor(self.args.relation_vocab[self.args.query_relations], device=self.device)).repeat(
            batch_num)  # torch.Size([1024])
        query_embeddings = self.relation_embedding(query_relations)  # torch.Size([1024, 64])  #
        prev_states = self.init_state(
            batch_num)  # [(torch.Size([1024, 128]), torch.Size([1024, 128])),  (torch.Size([1024, 128]), torch.Size([1024, 128]))]
        prev_relations = torch.tensor(
            np.ones(batch_num, dtype='int64') * self.args.relation_vocab['DUMMY_START_RELATION'],
            device=self.device)

        current_entities = x_batch[:, 0, 0]
        next_relations = grapher.array_store[current_entities, :, 1].to(self.device)  # torch.Size([1024, 501])
        next_entities = grapher.array_store[current_entities, :, 0].to(self.device)  # torch.Size([1024, 501])

        entity_trajectory = []
        relation_trajectory = []
        for i in range(path_length):
            new_states, actions, scores = self.step(prev_relations, current_entities, next_relations, next_entities,
                                                    prev_states, query_embeddings)
            new_states = [torch.stack(list(x)) for x in prev_states]  # [([12100,128],[12100,128]),([12100,128],[12100,128]), ([12100,128],[12100,128])]
            new_states = torch.stack(new_states)

            chosen_relations, chosen_entities, test_actions_idx, new_states, beam_probs, y = \
                self.beam_search(i, scores, beam_probs, batch_num, new_states, current_entities, next_relations, next_entities)
            for j in range(i):
                self.entity_trajectory[j] = self.entity_trajectory[j][y]
                self.relation_trajectory[j] = self.relation_trajectory[j][y]
            new_states = [torch.split(i, 1, dim=0) for i in new_states]
            new_states = [(torch.squeeze(i[0]), torch.squeeze(i[1])) for i in new_states]

            prev_relations = chosen_relations
            entity_trajectory.append(current_entities)
            relation_trajectory.append(chosen_relations)
            current_entities = chosen_entities
            next_entities, next_relations = self.next_episode(current_entities, test_actions_idx, grapher)
            prev_states = new_states


        return beam_probs, entity_trajectory, relation_trajectory



    def beam_search(self, i, scores, beam_probs, batch_num, new_states, current_entities, next_relations, next_entities):
        k = self.test_rollouts
        scores = scores + beam_probs
        if i == 0:
            best_idx = torch.argsort(scores)  # [temp_batch_size max_number_actions]
            best_idx = best_idx[:, -k:]  # [temp_batch_size, k]#(12100, 100)
            ranged_idx = torch.tile(torch.tensor([b for b in range(k)], dtype=torch.long),
                                    (batch_num,))  # [temp_batch_size]#(12100,) #ranged_idx=tensor([0,1,2,...,99, 0,1,2,...99,...])
            best_idx = best_idx[torch.arange(batch_num), ranged_idx]  # [k * B]#(12100,)
        else:
            best_idx = self.top_k(scores, k)#torch.Size([12100])

        y = torch.div(best_idx, self.max_num_actions, rounding_mode='trunc')
        y = y.data.cpu()
        x = best_idx % self.max_num_actions
        y += torch.from_numpy(np.repeat([b * k for b in range(batch_num/k)], k))
        current_entities = current_entities[y]
        next_relations = next_relations[y]
        next_entities = next_entities[y]
        new_states = tuple(mem[:, y, :] for mem in new_states)
        test_actions_idx = x
        chosen_relations = next_relations[torch.arange(batch_num), x]
        chosen_entities = next_entities[torch.arange(batch_num), x]
        beam_probs = scores[y, x].reshape((-1, 1))
        return chosen_relations, chosen_entities, test_actions_idx, new_states, beam_probs, y

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, k * max_num_actions]#(121, 20000)
        best_idx = torch.argsort(scores)#从小到大排序，取索引
        best_idx = best_idx[:, -k:]  # (121,100)#取得分最大的前100个
        return best_idx.reshape(-1)  # (12100,)

    def next_episode(self, current_entities, action, grapher):
        next_relations = grapher[current_entities, action, 1]
        next_entities = grapher[current_entities, action, 0]

        return next_entities, next_relations