import numpy as np
import sys
# sys.path.append("/home/linjie/projects/KG/PoLo/")
from grapher import RelationEntityGrapher
from feed_data import RelationEntityBatcher


class Episode(object):
    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, mode, batcher = params
        self.mode = mode
        if self.mode == 'train':
            self.rollouts = num_rollouts
        else:
            self.rollouts = test_rollouts
        self.current_hop = 0
        start_entities, query_relation,  end_entities, all_answers = data
        self.no_examples = start_entities.shape[0]
        self.start_entities = np.repeat(start_entities, self.rollouts)
        self.query_relations = np.repeat(query_relation, self.rollouts)
        self.end_entities = np.repeat(end_entities, self.rollouts)
        self.current_entities = np.repeat(start_entities, self.rollouts)
        self.all_answers = all_answers

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities,
                                                        self.query_relations, self.end_entities, self.all_answers,
                                                        self.current_hop == self.path_len - 1, self.rollouts)#(2560, 200, 2)
        self.states = dict()
        self.states['next_relations'] = next_actions[:, :, 1]
        self.states['next_entities'] = next_actions[:, :, 0]
        self.states['current_entities'] = self.current_entities

    def get_states(self):
        return self.states

    def get_query_relations(self):
        return self.query_relations

    def get_query_objects(self):
        return self.end_entities

    def get_rewards(self):
        rewards = (self.current_entities == self.end_entities)
        # Set the True and False values to the values of positive and negative rewards
        condlist = [rewards == True, rewards == False]
        choicelist = [self.positive_reward, self.negative_reward]
        rewards = np.select(condlist, choicelist)   # [B,]
        return rewards

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.states['next_entities'][np.arange(self.no_examples * self.rollouts), action]
        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities,
                                                        self.query_relations, self.end_entities, self.all_answers,
                                                        self.current_hop == self.path_len - 1, self.rollouts)
        self.states['next_relations'] = next_actions[:, :, 1]
        self.states['next_entities'] = next_actions[:, :, 0]
        self.states['current_entities'] = self.current_entities
        return self.states


class Env(object):
    def __init__(self, args, mode='train'):
        self.batch_size = args.batch_size
        self.num_rollouts = args.num_rollouts
        self.test_rollouts = args.test_rollouts
        self.mode = mode
        self.path_len = args.path_length
        input_dir = args.input_dir
        triple_store = input_dir + 'graph.txt'

        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=args.batch_size,
                                                 entity_vocab=args.entity_vocab,
                                                 relation_vocab=args.relation_vocab)
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=args.batch_size,
                                                 entity_vocab=args.entity_vocab,
                                                 relation_vocab=args.relation_vocab,
                                                 mode=mode)
            self.total_no_examples = self.batcher.store.shape[0]

        self.grapher = RelationEntityGrapher(triple_store=triple_store,
                                             entity_vocab=args.entity_vocab,
                                             relation_vocab=args.relation_vocab,
                                             max_num_actions=args.max_num_actions)

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, \
                 self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data is None:
                    return
                yield Episode(self.grapher, data, params)
