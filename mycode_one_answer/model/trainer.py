import codecs
import datetime
import gc
import json
import logging
import os
import random
import resource
import sys
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from scipy.special import logsumexp as lse
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from agent import Agent
from baseline import ReactiveBaseline
from environment import Env
from options import read_options
from rules import prepare_argument, check_rule, modify_rewards


# from pretrain_model import lstm_model

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        self.set_random_seeds(self.seed)
        self.train_environment = Env(params, 'train')
        self.dev_test_environment = Env(params, 'dev')
        self.test_test_environment = Env(params, 'test')
        self.test_environment = self.dev_test_environment
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.agent = Agent(params, self.rev_entity_vocab, self.rev_relation_vocab)
        # self.initialize_policy_step(self.lstm_pretrained_model)
        self.agent = self.agent.to(self.device)
        self.rule_list_dir = self.input_dir + self.rule_file
        with open(self.rule_list_dir, 'r') as file:
            self.rule_list = json.load(file)
        self.baseline = ReactiveBaseline(self.Lambda)
        self.optimizer = torch.optim.Adam(params=self.agent.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.99)#学习率learning_rate应用指数衰减
        self.best_metric = -1
        self.early_stopping = False
        self.current_patience = self.patience

    def set_random_seeds(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

    def calc_reinforce_loss(self):
        loss = torch.stack(self.per_example_loss, dim=1)  # (2560, 3)
        self.tf_baseline = self.baseline.get_baseline_value()#torch.Size([])

        final_rewards = self.cum_discounted_rewards - self.tf_baseline  # torch.Size([2560, 3])
        final_rewards_ = final_rewards.reshape(-1)#torch.Size([11520])
        # 计算均值和方差
        rewards_mean = torch.mean(final_rewards_)
        rewards_var = torch.var(final_rewards_, unbiased=False)
        rewards_std = torch.sqrt(rewards_var) + 1e-6  # Constant added for numerical stability
        final_rewards = torch.div(final_rewards - rewards_mean, rewards_std)  # shape=(2560, 3)

        loss = loss * final_rewards  # (2560, 3)
        self.decaying_beta = self.update_decaying_beta()
        total_loss = torch.mean(loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)  # ()
        return total_loss

    def update_decaying_beta(self):
        decay_steps = 200
        decay_rate = 0.90
        decaying_beta = self.beta * decay_rate ** (self.batch_counter / decay_steps)
        return decaying_beta

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)  # shape=(2560, 200, 3)
        entropy_policy = - torch.mean(torch.mean(torch.exp(all_logits) * all_logits, axis=1))
        return entropy_policy

    def initialize_policy_step(self, lstm_pretrained_model=None):
        if lstm_pretrained_model is not None:
            pretrain_agent = torch.load(lstm_pretrained_model)
            lstm_dict = pretrain_agent.policy_step.state_dict()
            self.agent.policy_step.load_state_dict(lstm_dict)
            print("load pretrain model ok")

    def initialize_agent(self, model_path=None):
        if model_path is None:
            self.agent = self.agent
        else:
            device = f"cuda:{self.device}"
            self.agent = torch.load(model_path, map_location=device)
            # self.agent.load_state_dict(torch.load(model_path, map_location=device))
            self.agent.device = self.device
            print(self.agent)

    def initialize_pretrained_embeddings(self):
        if self.pretrained_embeddings_relation != '':
            logger.info('Using pretrained relation embeddings.')
            pretrained_relations = np.load(self.pretrained_embeddings_relation)
            with open(self.pretrained_relation_to_id, 'r') as f:
                relation_to_id = json.load(f)
            rel_embeddings = self.agent.relation_embedding.weight.data
            for relation, idx in relation_to_id.items():
                rel_embeddings[self.relation_vocab[relation]] = torch.from_numpy(pretrained_relations[idx]).to(self.device)
            self.agent.relation_embedding.weight.data.copy_(rel_embeddings)
            self.agent.relation_embedding.weight.requires_grad = self.train_relation_embeddings

        if self.pretrained_embeddings_entity != '':
            logger.info('Using pretrained entity embeddings.')
            pretrained_entities = np.load(self.pretrained_embeddings_entity)
            with open(self.pretrained_entity_to_id, 'r') as f:
                entity_to_id = json.load(f)
            ent_embeddings = self.agent.entity_embedding.weight.data
            for entity, idx in entity_to_id.items():
                ent_embeddings[self.entity_vocab[entity]] = torch.from_numpy(pretrained_entities[idx]).to(self.device)
            self.agent.entity_embedding.weight.data.copy_(ent_embeddings)
            self.agent.entity_embedding.weight.requires_grad = self.train_entity_embeddings


    def calc_cum_discounted_rewards(self, rewards):
        running_add = torch.zeros([rewards.shape[0]], device=self.device)
        cum_disc_rewards = torch.zeros([rewards.shape[0], self.path_length], device=self.device)
        cum_disc_rewards[:, self.path_length - 1] = rewards
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_rewards[:, t]
            cum_disc_rewards[:, t] = running_add
        return cum_disc_rewards

    def beam_search(self, i, test_scores, beam_probs, temp_batch_size, states, agent_mem):
        k = self.test_rollouts
        new_scores = test_scores + beam_probs  # [k * B, max_number_actions]#(12800, 400)
        if i == 0:
            best_idx = torch.argsort(new_scores)  # [k * B, max_number_actions] #从小到大升序排序
            best_idx = best_idx[:, -k:]  # [k * B, k]#(12800, 100)
            ranged_idx = torch.tile(torch.tensor([b for b in range(k)], dtype=torch.long),
                                    (temp_batch_size,))  # [k *B]#(12800,) #tensor([0,1,2,...,99, 0,1,2,...99,...])
            best_idx = best_idx[torch.arange(k * temp_batch_size), ranged_idx]  # [k * B]
        else:
            best_idx = self.top_k(new_scores, k)#torch.Size([12800])

        y = torch.div(best_idx, self.max_num_actions, rounding_mode='trunc')
        y = y.data.cpu()
        x = (best_idx % self.max_num_actions).data.cpu()
        y += torch.from_numpy(
            np.repeat([b * k for b in range(temp_batch_size)], k))# tensor([0,0,0,...0, 100,100,...100,..., 12700,12700...12700 ])
        states['current_entities'] = states['current_entities'][y]#(12800,)
        states['next_relations'] = states['next_relations'][y]#(12800, 400)
        states['next_entities'] = states['next_entities'][y]#(12800, 400)
        agent_mem = tuple(mem[:, y, :] for mem in agent_mem)
        test_actions_idx = x
        chosen_relations = states['next_relations'][np.arange(k * temp_batch_size), x]
        beam_probs = new_scores[y, x].reshape((-1, 1))#torch.Size([12800, 1])
        return chosen_relations, test_actions_idx, states, agent_mem, beam_probs, y

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, k * max_num_actions]#(128, 20000)
        best_idx = torch.argsort(scores)#从小到大排序，取索引#torch.Size([128, 20000])
        best_idx = best_idx[:, -k:]  # (128,100)#取得分最大的前100个
        return best_idx.reshape(-1)  # (12800,)

    def paths_and_rules_stats(self, b, sorted_idx, qr, ce, end_e, test_rule_count_body, test_rule_count,
                              num_query_with_rules, num_query_with_rules_correct):
        rule_in_path = False
        is_correct = False
        answer_pos_rule = None
        pos_rule = 0
        seen_rule = set()
        for r in sorted_idx[b]:
            argument_temp = self.get_argument(b, r)#采样到的路径,eg:['CbG', 'Gene::11309', '_CbG', 'Compound::DB00950', 'NO_OP', 'Compound::DB00950']
            key_temp = ' '.join(argument_temp[::2])#采样到的关系,eg:'CbG _CbG NO_OP'
            self.paths_stats(argument_temp, key_temp, qr, end_e)
            body, obj = prepare_argument(argument_temp)#body是去掉'NO_OP'后剩下的关系:['CbG', '_CbG']，obj是argument_temp[-1]:'Compound::DB00950'
            rule_in_path, is_correct, answer_pos_rule, pos_rule, seen_rule, test_rule_count_body, test_rule_count = \
                self.rules_stats(b, r, qr, body, obj, ce, end_e, key_temp, test_rule_count_body, test_rule_count,
                                 rule_in_path, is_correct, answer_pos_rule, pos_rule, seen_rule)
        if rule_in_path:
            num_query_with_rules[0] += 1
            if qr[0] == '_':
                num_query_with_rules[1] += 1
            else:
                num_query_with_rules[2] += 1
            if is_correct:
                num_query_with_rules_correct[0] += 1
                if qr[0] == '_':
                    num_query_with_rules_correct[1] += 1
                else:
                    num_query_with_rules_correct[2] += 1
        return num_query_with_rules, num_query_with_rules_correct, answer_pos_rule, test_rule_count_body, \
               test_rule_count


    def get_argument(self, b, r):
        idx = b * self.test_rollouts + r
        argument_temp = [None] * (2 * len(self.relation_trajectory))
        argument_temp[::2] = [str(self.rev_relation_vocab[re[idx]]) for re in self.relation_trajectory]
        argument_temp[1::2] = [str(self.rev_entity_vocab[e[idx]]) for e in self.entity_trajectory][1:]
        return argument_temp

    def paths_stats(self, argument_temp, key_temp, qr, end_e):
        if qr in self.paths_body:
            if key_temp in self.paths_body[qr]:
                self.paths_body[qr][key_temp]['occurrences'] += 1
                if argument_temp[-1] == end_e:
                    self.paths_body[qr][key_temp]['correct_entities'] += 1
            else:
                self.paths_body[qr][key_temp] = {}
                self.paths_body[qr][key_temp]['relation'] = qr
                self.paths_body[qr][key_temp]['occurrences'] = 1
                if argument_temp[-1] == end_e:
                    self.paths_body[qr][key_temp]['correct_entities'] = 1
                else:
                    self.paths_body[qr][key_temp]['correct_entities'] = 0
        else:
            self.paths_body[qr] = {}
            self.paths_body[qr][key_temp] = {}
            self.paths_body[qr][key_temp]['relation'] = qr
            self.paths_body[qr][key_temp]['occurrences'] = 1
            if argument_temp[-1] == end_e:
                self.paths_body[qr][key_temp]['correct_entities'] = 1
            else:
                self.paths_body[qr][key_temp]['correct_entities'] = 0#self.paths_body={'CmC': {'CbG _CbG NO_OP': {'relation': 'CmC', 'occurrences': 1, 'correct_entities': 0}}}

    def rules_stats(self, b, r, qr, body, obj, ce, end_e, key_temp, test_rule_count_body, test_rule_count, rule_in_path,
                    is_correct, answer_pos_rule, pos_rule, seen_rule):
        rule_applied = False
        if qr in self.rule_list:
            rel_rules = self.rule_list[qr]
            for j in range(len(rel_rules)):
                if check_rule(body, obj, end_e, rel_rules[j], only_body=True):#判断预测出来的关系列表是否等于提供的原路径
                    rule_applied = True
                    rule_in_path = True
                    test_rule_count_body[0] += 1
                    if qr[0] == '_':
                        test_rule_count_body[1] += 1
                    else:
                        test_rule_count_body[2] += 1
                    if check_rule(body, obj, end_e, rel_rules[j], only_body=False):#判断预测出来的关系是否等于提供的原路径,且最后的entity等于真实的end_e
                        is_correct = True
                        test_rule_count[0] += 1
                        if qr[0] == '_':
                            test_rule_count[1] += 1
                        else:
                            test_rule_count[2] += 1
                        if answer_pos_rule is None:
                            answer_pos_rule = pos_rule
                    break
        if (ce[b, r] not in seen_rule) and rule_applied:
            seen_rule.add(ce[b, r])
            pos_rule += 1
        if rule_applied:
            self.paths_body[qr][key_temp]['is_rule'] = 'yes'
        else:
            self.paths_body[qr][key_temp]['is_rule'] = 'no'
        return rule_in_path, is_correct, answer_pos_rule, pos_rule, seen_rule, test_rule_count_body, test_rule_count

    def get_answer_pos(self, b, sorted_idx, rewards_reshape, ce):
        answer_pos = None
        pos = 0
        seen = set()
        if self.pool == 'max':
            for r in sorted_idx[b]:
                if rewards_reshape[b, r] == self.positive_reward:
                    answer_pos = pos
                    break
                if ce[b, r] not in seen:
                    seen.add(ce[b, r])
                    pos += 1
        if self.pool == 'sum':
            scores = defaultdict(list)
            answer = ''
            for r in sorted_idx[b]:
                scores[ce[b, r]].append(self.log_probs[b, r].data.cpu().numpy())
                if rewards_reshape[b, r] == self.positive_reward:
                    answer = ce[b, r]
            final_scores = defaultdict(float)
            for e in scores:
                final_scores[e] = lse(scores[e])# #lse解释：一个列表x 取出x最大值b = x.max()，再y = np.exp(x - b)==》y / y.sum()
            sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
            if answer in sorted_answers:
                answer_pos = sorted_answers.index(answer)
            else:
                answer_pos = None
        return answer_pos

    def calculate_query_metrics(self, query_metrics, answer_pos):
        if answer_pos is not None:
            query_metrics[5] += 1.0 / (answer_pos + 1)
            if answer_pos < 20:
                query_metrics[4] += 1
                if answer_pos < 10:
                    query_metrics[3] += 1
                    if answer_pos < 5:
                        query_metrics[2] += 1
                        if answer_pos < 3:
                            query_metrics[1] += 1
                            if answer_pos < 1:
                                query_metrics[0] += 1
        return query_metrics

    def add_paths(self, b, sorted_idx, qr, start_e, se, ce, end_e, answer_pos, answers, rewards):
        tmp_answer_pos = -1 if answer_pos is None else answer_pos
        # all_answers = '\t'.join(all_answers)
        # self.paths[str(qr)].append(str(start_e) + '\t' + all_answers + '\t'+ str(tmp_answer_pos)+ '\n')
        self.paths[str(qr)].append(str(start_e) + '\t' + str(end_e) + '\t' + str(tmp_answer_pos) + '\n')
        self.paths[str(qr)].append('Reward:' + str(1 if (answer_pos is not None) and (answer_pos < 10) else 0) + '\n')
        for r in sorted_idx[b]:
            rev = -1
            idx = b * self.test_rollouts + r
            if rewards[idx] == self.positive_reward:
                rev = 1
            answers.append(self.rev_entity_vocab[se[b, r]] + '\t' + self.rev_entity_vocab[ce[b, r]] + '\t' +
                           str(self.log_probs[b, r]) + '\n')
            self.paths[str(qr)].append('\t'.join([str(self.rev_entity_vocab[e[idx]]) for e in self.entity_trajectory])
                                       + '\n' + '\t'.join([str(self.rev_relation_vocab[re[idx]]) for
                                                           re in self.relation_trajectory]) + '\n' +
                                       str(rev) + '\n' + str(self.log_probs[b, r]) + '\n___' + '\n')
        self.paths[str(qr)].append('#####################\n')
        return answers

    def write_paths_file(self, answers):
        for q in self.paths:
            j = q.replace('/', '-')
            with codecs.open(self.paths_log + '_' + j, 'a', 'utf-8') as pos_file:
                for p in self.paths[q]:
                    pos_file.write(p)
        with open(self.paths_log + 'answers', 'w') as answer_file:
            for a in answers:
                answer_file.write(a)

    def write_paths_summary(self):
        with open(self.output_dir + 'paths_body.json', 'w') as path_file:
            path_file.write('path\toccurrences\tcorrect_entities\tis_rule\trelation\n')
            path_file.write('####################\n')
            for qr in self.paths_body.keys():
                paths_body_sorted = sorted(self.paths_body[qr], key=lambda x: self.paths_body[qr][x]['occurrences'],
                                           reverse=True)
                for p in paths_body_sorted:
                    path_file.write(p + '\t' + str(self.paths_body[qr][p]['occurrences']) + '\t' +
                                    str(self.paths_body[qr][p]['correct_entities']) + '\t' +
                                    self.paths_body[qr][p]['is_rule'] + '\t' +
                                    self.paths_body[qr][p]['relation'] + '\n')
                path_file.write('####################\n')

    def write_scores_file(self, scores_file, final_metrics, final_metrics_rule, final_metrics_head,
                          final_metrics_rule_head, final_metrics_tail, final_metrics_rule_tail, test_rule_count_body,
                          test_rule_count, num_query_with_rules, num_query_with_rules_correct, total_examples):
        metrics = ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'Hits@20', 'MRR']
        metrics_rule = ['Hits@1_rule', 'Hits@3_rule', 'Hits@5_rule', 'Hits@10_rule', 'Hits@20_rule', 'MRR_rule']
        ranking = ['Both:', 'Head:', 'Tail:']
        num_examples = [total_examples, total_examples / 2, total_examples / 2]
        all_results = [[final_metrics, final_metrics_rule],
                       [final_metrics_head, final_metrics_rule_head],
                       [final_metrics_tail, final_metrics_rule_tail]]
        for j in range(len(ranking)):
            scores_file.write(ranking[j])
            scores_file.write('\n')
            for i in range(len(metrics)):
                scores_file.write(metrics[i] + ': {0:7.4f}'.format(all_results[j][0][i]))
                scores_file.write('\n')
            for i in range(len(metrics_rule)):
                scores_file.write(metrics_rule[i] + ': {0:7.4f}'.format(all_results[j][1][i]))
                scores_file.write('\n')
            scores_file.write('Rule count body: {0}/{1} = {2:6.4f}'.format(
                int(test_rule_count_body[j]), int(num_examples[j] * self.test_rollouts),
                test_rule_count_body[j] / (num_examples[j] * self.test_rollouts)))
            scores_file.write('\n')
            scores_file.write('Rule count correct: {0}/{1} = {2:6.4f}'.format(
                int(test_rule_count[j]), int(num_examples[j] * self.test_rollouts),
                test_rule_count[j] / (num_examples[j] * self.test_rollouts)))
            scores_file.write('\n')
            scores_file.write('Number of queries with at least one rule: {0}/{1} = {2:6.4f}'.format(
                int(num_query_with_rules[j]), int(num_examples[j]), num_query_with_rules[j] / num_examples[j]))
            scores_file.write('\n')
            scores_file.write('Number of queries with at least one rule and correct: {0}/{1} = {2:6.4f}'.format(
                int(num_query_with_rules_correct[j]), int(num_examples[j]),
                num_query_with_rules_correct[j] / num_examples[j]))
            scores_file.write('\n')
            scores_file.write('\n')

    def write_scores_file_tail(self, scores_file, final_metrics, final_metrics_rule, final_metrics_head,
                               final_metrics_rule_head, final_metrics_tail, final_metrics_rule_tail,
                               test_rule_count_body,
                               test_rule_count, num_query_with_rules, num_query_with_rules_correct, total_examples):
        metrics = ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'Hits@20', 'MRR']
        metrics_rule = ['Hits@1_rule', 'Hits@3_rule', 'Hits@5_rule', 'Hits@10_rule', 'Hits@20_rule', 'MRR_rule']
        ranking = ['Tail:']
        num_examples = [total_examples]
        all_results = [[final_metrics, final_metrics_rule],
                       [final_metrics_head, final_metrics_rule_head],
                       [final_metrics_tail, final_metrics_rule_tail]]
        for j in range(len(ranking)):
            scores_file.write(ranking[j])
            scores_file.write('\n')
            for i in range(len(metrics)):
                scores_file.write(metrics[i] + ': {0:7.4f}'.format(all_results[j][0][i]))
                scores_file.write('\n')
            for i in range(len(metrics_rule)):
                scores_file.write(metrics_rule[i] + ': {0:7.4f}'.format(all_results[j][1][i]))
                scores_file.write('\n')
            scores_file.write('Rule count body: {0}/{1} = {2:6.4f}'.format(
                int(test_rule_count_body[j]), int(num_examples[j] * self.test_rollouts),
                test_rule_count_body[j] / (num_examples[j] * self.test_rollouts)))
            scores_file.write('\n')
            scores_file.write('Rule count correct: {0}/{1} = {2:6.4f}'.format(
                int(test_rule_count[j]), int(num_examples[j] * self.test_rollouts),
                test_rule_count[j] / (num_examples[j] * self.test_rollouts)))
            scores_file.write('\n')
            scores_file.write('Number of queries with at least one rule: {0}/{1} = {2:6.4f}'.format(
                int(num_query_with_rules[j]), int(num_examples[j]), num_query_with_rules[j] / num_examples[j]))
            scores_file.write('\n')
            scores_file.write('Number of queries with at least one rule and correct: {0}/{1} = {2:6.4f}'.format(
                int(num_query_with_rules_correct[j]), int(num_examples[j]),
                num_query_with_rules_correct[j] / num_examples[j]))
            scores_file.write('\n')
            scores_file.write('\n')

    def train(self):
        self.range_arr = torch.arange(self.batch_size * self.num_rollouts, device=self.device)
        train_loss = 0.0
        self.batch_counter = 0
        self.initialize_policy_step(self.lstm_pretrained_model)
        for episode in self.train_environment.get_episodes():

            self.agent.train()
            self.batch_counter += 1

            self.per_example_loss, self.per_example_logits, self.actions_idx, arguments = self.agent(episode,
                                                                                                     self.range_arr,
                                                                                                     self.path_length)

            query_rel_string = np.array([self.rev_relation_vocab[x] for x in episode.get_query_relations()])
            obj_string = np.array([self.rev_entity_vocab[x] for x in episode.get_query_objects()])

            rewards = episode.get_rewards()#(3840,)
            rewards, rule_count, rule_count_body = modify_rewards(self.rule_list, arguments, query_rel_string,
                                                                  obj_string, self.rule_base_reward, rewards,
                                                                  self.only_body)
            # rewards = torch.from_numpy(rewards)
            self.cum_discounted_rewards = self.calc_cum_discounted_rewards(torch.from_numpy(rewards).to(self.device))

            # Backpropagation
            batch_total_loss = self.calc_reinforce_loss()
            # 更新梯度， self.baseline.update
            self.baseline.update(torch.mean(self.cum_discounted_rewards))
            self.optimizer.zero_grad()  # 清空梯度
            batch_total_loss.backward()
            self.optimizer.step()


            # Print statistics
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            num_hits = np.sum(rewards > 0)#多少条具体路径hit到
            avg_reward = np.mean(rewards)
            rewards_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))#(128,30)
            rewards_reshape = np.sum(rewards_reshape, axis=1)#(128,)
            num_ep_correct = np.sum(rewards_reshape > 0)#以head_entity为组，计算多少个head_entity能hit到
            if np.isnan(train_loss.item()):
                raise ArithmeticError('Error in computing loss.')

            logger.info('batch_counter: {0:4d}, num_hits: {1:7d}, avg_reward: {2:6.4f}, num_ep_correct: {3:4d}, '
                        'avg_ep_correct: {4:6.4f}, train_loss: {5:6.4f}'.
                        format(self.batch_counter, num_hits, avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size), train_loss.item()))
            logger.info('rule_count_body: {0}/{1} = {2:6.4f}'.format(
                rule_count_body, self.batch_size * self.num_rollouts,
                                 rule_count_body / (self.batch_size * self.num_rollouts)))
            logger.info('rule_count_correct: {0}/{1} = {2:6.4f}'.format(
                rule_count, self.batch_size * self.num_rollouts,
                            rule_count / (self.batch_size * self.num_rollouts)))

            if self.batch_counter % self.eval_every == 0:
                with open(self.output_dir + 'scores.txt', 'a') as score_file:
                    score_file.write('Scores for iteration ' + str(self.batch_counter) + '\n')
                paths_log_dir = self.output_dir + str(self.batch_counter) + '/'
                os.makedirs(paths_log_dir)
                self.paths_log = paths_log_dir + 'paths'
                self.test()

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            gc.collect()

            if self.early_stopping:
                break
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, print_paths=False, save_model=True, beam=True):
        self.paths = defaultdict(list)
        self.paths_body = dict()
        batch_counter = 0
        answers = []
        feed_dict = {}
        # For the calculation of the rankings ['Both:', 'Head:', 'Tail:']
        test_rule_count_body = np.zeros(3)
        test_rule_count = np.zeros(3)
        num_query_with_rules = np.zeros(3)
        num_query_with_rules_correct = np.zeros(3)
        # For the calculation of the metrics [h@1, h@3, h@5, h@10, h@20, MRR]
        final_metrics = np.zeros(6)
        final_metrics_rule = np.zeros(6)
        final_metrics_head = np.zeros(6)
        final_metrics_rule_head = np.zeros(6)
        final_metrics_tail = np.zeros(6)
        final_metrics_rule_tail = np.zeros(6)
        total_examples = self.test_environment.total_no_examples#663

        self.agent.eval()
        with torch.no_grad():
            for episode in tqdm(self.test_environment.get_episodes()):
                batch_counter += 1
                temp_batch_size = episode.no_examples  # 128
                beam_probs = torch.zeros((temp_batch_size * self.test_rollouts, 1), device=self.device)  # torch.Size([12800, 1])
                lstm_layers, _, _, hsize = self.agent.get_mem_shape()
                agent_mem = []
                for layer in range(lstm_layers):
                    states_c = torch.zeros((temp_batch_size * self.test_rollouts, hsize),
                                           dtype=torch.float32, device=self.device)
                    states_h = torch.zeros((temp_batch_size * self.test_rollouts, hsize),
                                           dtype=torch.float32, device=self.device)
                    agent_mem.append((states_c, states_h))
                previous_relations = torch.ones((temp_batch_size * self.test_rollouts), dtype=torch.int64,
                                                device=self.device) * self.relation_vocab['DUMMY_START_RELATION']
                self.qrs = episode.get_query_relations()  # query_relations #array([52, 52, 52, ..., 52, 52, 52]) #(12800,)
                query_relations = torch.from_numpy(episode.get_query_relations()).to(self.device)  # torch.Size([12800])
                query_embeddings = self.agent.relation_embedding(query_relations) #torch.Size([12800, 64])
                episode_states = episode.get_states()
                range_arr = torch.arange(temp_batch_size * self.test_rollouts, device=self.device)#torch.Size([12800])
                self.log_probs = torch.zeros((temp_batch_size * self.test_rollouts,), device=self.device) * 1.0 #torch.Size([12800])
                self.entity_trajectory = []
                self.relation_trajectory = []

                for i in range(self.path_length):
                    next_relations = torch.from_numpy(episode_states['next_relations']).to(self.device)
                    next_entities = torch.from_numpy(episode_states['next_entities']).to(self.device)
                    current_entities = torch.from_numpy(episode_states['current_entities']).to(self.device)
                    loss, test_scores, agent_mem, test_actions_idx, chosen_relations = self.agent.step(
                        next_relations=next_relations,
                        next_entities=next_entities,
                        current_entities=current_entities,
                        prev_states=agent_mem,
                        prev_relations=previous_relations,
                        query_embeddings=query_embeddings,
                        range_arr=range_arr)
                    agent_mem = [torch.stack(list(x)) for x in agent_mem]#[([12800,128],[12800,128]),([12800,128],[12800,128])]
                    agent_mem = torch.stack(agent_mem)#torch.Size([2, 2, 12800, 128]) 第一个2是2层layer
                    if beam:
                        chosen_relations, test_actions_idx, episode_states, agent_mem, beam_probs, y = \
                            self.beam_search(i, test_scores, beam_probs, temp_batch_size, episode_states, agent_mem)
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                    # agent_mem = torch.stack(agent_mem)
                    agent_mem = [torch.split(i, 1, dim=0) for i in agent_mem]
                    agent_mem = [(torch.squeeze(i[0]), torch.squeeze(i[1])) for i in agent_mem]

                    previous_relations = torch.from_numpy(chosen_relations).to(self.device)#torch.Size([12800])
                    self.entity_trajectory.append(episode_states['current_entities'])
                    self.relation_trajectory.append(chosen_relations)
                    episode_states = episode(test_actions_idx)
                    self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_actions_idx]

                if beam:
                    self.log_probs = beam_probs#torch.Size([12800, 1])

                self.entity_trajectory.append(episode_states['current_entities'])

                # Ask environment for final reward
                rewards = episode.get_rewards()#(12800,)
                # rewards = torch.from_numpy(rewards)
                rewards_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))#(128, 100)
                self.log_probs = torch.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))#torch.Size([128, 100])
                sorted_idx = torch.argsort(-self.log_probs)#将分数转为正,从小到大升序排序《==》相当于从大到小降序排序取索引
                sorted_idx = sorted_idx.data.cpu().numpy()

                query_metrics = np.zeros(6)
                query_metrics_rule = np.zeros(6)
                query_metrics_head = np.zeros(6)
                query_metrics_rule_head = np.zeros(6)
                query_metrics_tail = np.zeros(6)
                query_metrics_rule_tail = np.zeros(6)

                ce = episode.states['current_entities'].reshape((temp_batch_size, self.test_rollouts))#(128, 100)
                se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))#(128, 100)
                for b in range(temp_batch_size):
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qrs[b * self.test_rollouts]]#'CmC'
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]#'Compound::DB01045'
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]#'Compound::DB00482'
                    all_answers = [self.rev_entity_vocab[i] for i in episode.all_answers[b]]
                    num_query_with_rules, num_query_with_rules_correct, answer_pos_rule, test_rule_count_body, \
                    test_rule_count = self.paths_and_rules_stats(b, sorted_idx, qr, ce, end_e, test_rule_count_body,
                                                                 test_rule_count, num_query_with_rules,
                                                                 num_query_with_rules_correct)
                    answer_pos = self.get_answer_pos(b, sorted_idx, rewards_reshape, ce)#answer的排序位置
                    # if answer_pos is not None and answer_pos < 3:
                    #     print(start_e, end_e)
                    query_metrics = self.calculate_query_metrics(query_metrics, answer_pos)#hit tail_entity rate
                    query_metrics_rule = self.calculate_query_metrics(query_metrics_rule, answer_pos_rule)#hit matepath&tail_entity rate
                    if qr[0] == '_':  # Inverse triple
                        query_metrics_head = self.calculate_query_metrics(query_metrics_head, answer_pos)#hit _CtD tail_entity rate
                        query_metrics_rule_head = self.calculate_query_metrics(query_metrics_rule_head, answer_pos_rule)#hit _CtD matepath&tail_entity rate
                    else:
                        query_metrics_tail = self.calculate_query_metrics(query_metrics_tail, answer_pos)#hit CtD tail_entity rate
                        query_metrics_rule_tail = self.calculate_query_metrics(query_metrics_rule_tail, answer_pos_rule)#hit CtD matepath&tail_entity rate
                    if print_paths:
                        answers = self.add_paths(b, sorted_idx, qr, start_e, se, ce, end_e, answer_pos, answers,
                                                 rewards)

                final_metrics += query_metrics
                final_metrics_rule += query_metrics_rule
                final_metrics_head += query_metrics_head
                final_metrics_rule_head += query_metrics_rule_head
                final_metrics_tail += query_metrics_tail
                final_metrics_rule_tail += query_metrics_rule_tail
        print(final_metrics)
        final_metrics /= total_examples#hit tail_entity rate
        final_metrics_rule /= total_examples#hit matepath&tail_entity rate
        final_metrics_head /= total_examples / 2#hit _CtD tail_entity rate
        final_metrics_rule_head /= total_examples / 2#hit _CtD matepath&tail_entity rate
        final_metrics_tail /= total_examples / 2#hit CtD tail_entity rate
        final_metrics_rule_tail /= total_examples / 2#hit CtD matepath&tail_entity rate

        if save_model:
            if final_metrics[-1] > self.best_metric:
                self.best_metric = final_metrics[-1]
                torch.save(self.agent, self.model_dir + '/model.ckpt')
                # torch.save(self.agent.state_dict(), self.model_dir + '/model.ckpt')
                self.current_patience = self.patience
            elif self.best_metric >= final_metrics[-1]:
                self.current_patience -= 1
                if self.current_patience == 0:
                    self.early_stopping = True

        self.write_paths_summary()
        if print_paths:
            logger.info('Printing paths at {}'.format(self.output_dir + 'test_beam/'))
            self.write_paths_file(answers)

        with open(self.output_dir + 'scores.txt', 'a') as scores_file:
            self.write_scores_file_tail(scores_file, final_metrics, final_metrics_rule, final_metrics_head,
                                        final_metrics_rule_head, final_metrics_tail, final_metrics_rule_tail,
                                        test_rule_count_body, test_rule_count, num_query_with_rules,
                                        num_query_with_rules_correct, total_examples)

        metrics = ['Hits@1', 'Hits@3', 'Hits@5', 'Hits@10', 'Hits@20', 'MRR']
        for i in range(len(metrics)):
            logger.info(metrics[i] + ': {0:7.4f}'.format(final_metrics[i]))
        metrics_rule = ['Hits@1_rule', 'Hits@3_rule', 'Hits@5_rule', 'Hits@10_rule', 'Hits@20_rule', 'MRR_rule']
        for i in range(len(metrics_rule)):
            logger.info(metrics_rule[i] + ': {0:7.4f}'.format(final_metrics_rule[i]))


def create_output_and_model_dir(params, mode):
    current_time = datetime.datetime.now()
    current_time = current_time.strftime('%d%b%y_%H%M%S')
    if mode == 'test':
        params['output_dir'] = params['base_output_dir'] + str(current_time) + '_TEST' + \
                               '_p' + str(params['path_length']) + '_r' + str(params['rule_base_reward']) + \
                               '_e' + str(params['embedding_size']) + '_h' + str(params['hidden_size']) + \
                               '_L' + str(params['LSTM_layers']) + '_l' + str(params['learning_rate']) + \
                               '_o' + str(params['only_body']) + '/'
        os.makedirs(params['output_dir'])
    else:
        params['output_dir'] = params['base_output_dir'] + str(current_time) + \
                               '_p' + str(params['path_length']) + '_r' + str(params['rule_base_reward']) + \
                               '_e' + str(params['embedding_size']) + '_h' + str(params['hidden_size']) + \
                               '_L' + str(params['LSTM_layers']) + '_l' + str(params['learning_rate']) + \
                               '_o' + str(params['only_body']) + '/'
        params['model_dir'] = params['output_dir'] + 'model/'
        os.makedirs(params['output_dir'])
        os.makedirs(params['model_dir'])
    return params


def initialize_setting(params, relation_vocab, entity_vocab, mode=''):
    params = create_output_and_model_dir(params, mode)
    params.pop('relation_vocab', None)
    params.pop('entity_vocab', None)
    with open(params['output_dir'] + 'config.txt', 'w') as out:
        pprint(params, stream=out)
    maxLen = max([len(k) for k in params.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(params.items()):
        print(fmtString % keyPair)
    params['relation_vocab'] = relation_vocab
    params['entity_vocab'] = entity_vocab
    return params

"""
--input_dir /home/linjie/projects/KG/PoLo/datasets/Hetionet/
--base_output_dir /home/linjie/projects/KG/PoLo/output/Hetionet/
--total_iterations 50
--eval_every 2
--patience 3
--rule_base_reward 3
--only_body 1
--embedding_size 32
--hidden_size 32
--LSTM_layers 2
--beta 0.02
--Lambda 0.05
--learning_rate 0.0015
--max_num_actions 200
--train_entity_embeddings 1
--use_entity_embeddings 1
--num_rollouts 20
--load_model 1
--model_load_path /home/linjie/projects/KG/PoLo/output/Hetionet/07Dec22_132121_p3_r3.0_e32_h32_L2_l0.0015_o1/model/model.ckpt
"""
if __name__ == '__main__':
    options = read_options()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %H:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = None
    logger.info('Reading vocab files...')
    vocab_dir = options['input_dir'] + 'vocab/'
    relation_vocab = json.load(open(vocab_dir + 'relation_vocab.json'))
    entity_vocab = json.load(open(vocab_dir + 'entity_vocab.json'))
    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))

    if not options['load_model']:
        for k, v in options.items():
            if not isinstance(v, list):
                options[k] = [v]
        best_permutation = None
        best_metric = -1
        for permutation in ParameterGrid(options):
            permutation = initialize_setting(permutation, relation_vocab, entity_vocab)
            logger.removeHandler(logfile)
            logfile = logging.FileHandler(permutation['output_dir'] + 'log.txt', 'w')
            logfile.setFormatter(fmt)
            logger.addHandler(logfile)

            # Training
            trainer = Trainer(permutation)
            trainer.initialize_pretrained_embeddings()
            trainer.train()

            if (best_permutation is None) or (trainer.best_metric > best_metric):
                best_metric = trainer.best_metric
                best_permutation = permutation

        # Testing on test set with best model
        best_permutation['old_output_dir'] = best_permutation['output_dir']
        best_permutation = initialize_setting(best_permutation, relation_vocab, entity_vocab, mode='test')
        logger.removeHandler(logfile)
        logfile = logging.FileHandler(best_permutation['output_dir'] + 'log.txt', 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        trainer = Trainer(best_permutation)
        model_path = best_permutation['old_output_dir'] + 'model/model.ckpt'
        output_dir = best_permutation['output_dir']

        trainer.initialize_agent(model_path)
        os.makedirs(output_dir + 'test_beam/')
        trainer.paths_log = output_dir + 'test_beam/paths'
        with open(output_dir + 'scores.txt', 'a') as scores_file:
            scores_file.write('Test (beam) scores with best model from ' + model_path + '\n')
        trainer.test_environment = trainer.test_test_environment
        trainer.test(print_paths=True, save_model=False)

        print(f"train save in {best_permutation['old_output_dir']}")
        print(f"test save in {best_permutation['output_dir']}")

    else:
        for k, v in options.items():
            if isinstance(v, list):
                if len(v) == 1:
                    options[k] = v[0]
                else:
                    raise ValueError('Parameter {} has more than one value in the config file.'.format(k))
        logger.info('Skipping training...')
        model_path = options['model_load_path']
        logger.info('Loading model from {}'.format(model_path))
        options = initialize_setting(options, relation_vocab, entity_vocab, mode='test')
        output_dir = options['output_dir']
        logger.removeHandler(logfile)
        logfile = logging.FileHandler(output_dir + 'log.txt', 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        trainer = Trainer(options)

        trainer.initialize_agent(model_path)
        os.makedirs(output_dir + 'test_beam/')
        trainer.paths_log = output_dir + 'test_beam/paths'
        with open(output_dir + 'scores.txt', 'a') as scores_file:
            scores_file.write('Test (beam) scores with best model from ' + model_path + '\n')
        trainer.test_environment = trainer.test_test_environment
        trainer.test(print_paths=True, save_model=False)
