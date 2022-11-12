import random

import numpy as np

from policies.AbstractCacheNetwork import AbstractCacheNetwork


class LRUCacheNetwork(AbstractCacheNetwork):
    def __init__(self, network_properties):
        super().__init__(network_properties)
        self.gradient_normsX = 0
        self.gradient_normsXVec = np.zeros(self.graph_size)
        self.gradient_normsDV = 0
        self.gradient_normsDVVec = np.zeros(self.players)
        self.etas = np.zeros(self.graph_size)

        self.average_fractional_gains = np.zeros(self.players)
        self.queues = {}
        for n in self.graph:
            self.queues[n] = list(random.sample(set(range(self.catalog_size)), self.capacities[n]))
        self.refresh_states_from_queues()

    def refresh_states_from_queues(self):
        for n in self.graph:
            self.fractional_caches[n][self.mask_caches[n]] *= 0
            self.fractional_caches[n][np.array(self.queues[n])] = 1

    def adapt_state(self):
        if self.telescope_requests:
            query_nodes = [np.random.choice(self.query_nodes)]
        else:
            query_nodes = self.query_nodes
        for query_node in query_nodes:
            owner = self.cache_owners[query_node]
            batch = self.query_nodes_trace[query_node][self.t - 1]
            requests = []
            item = np.random.choice(np.arange(self.catalog_size), p = np.array(batch)/np.sum(batch))
            candidates = np.array(self.query_node_candidates[(query_node, item)][0]).astype(int)
            for i in range(len(candidates) - 1):
                candidate = candidates[i]
                hit_index = np.where(item == np.array(self.queues[candidate]))[0]
                hit_index = -1 if len(hit_index) == 0 else hit_index[0]
                if hit_index == -1:
                    self.queues[candidate] = [item] + self.queues[candidate][1:]
                elif hit_index>0:
                    self.queues[candidate] = [item] + self.queues[candidate][:hit_index] + self.queues[
                                                                                                         candidate][
                                                                                                     hit_index + 1:]
        self.refresh_states_from_queues()
