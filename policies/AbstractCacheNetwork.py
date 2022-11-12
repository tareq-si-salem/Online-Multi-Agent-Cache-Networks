import logging
from abc import ABC, abstractmethod

import cvxpy as cp
import networkx as nx
import numpy as np

from tools import sample_simplex, inv_dict, refresh_weights


class AbstractCacheNetwork(ABC):
    def __init__(self, network_properties):
        for key, val in network_properties.items():
            setattr(self, key, val)
        self.t = 0
        self.stats_dynamic = {'fractional_gain': [], 'integral_gain': [], 'repo_cost': [], 'opt_fractional_gain': [],
                              'average_fractional_gain': [], 'average_integral_gain': [], 'average_repo_cost': [],
                              'average_opt_fractional_gain': [],
                              'etas': [], 'dvs': [], 'etaDVs': []}
        self.stats_static = {}
        self.integral_caches = np.zeros((self.graph_size, self.catalog_size))
        self.fractional_caches = np.zeros((self.graph_size, self.catalog_size))
        self.mask_caches = np.ones((self.graph_size, self.catalog_size)).astype(bool)
        self.subgradients = {p: np.zeros((self.graph_size, self.catalog_size)) for p in range(self.players)}
        self.weights = refresh_weights(self.graph.edges(), self.min_weight, self.max_weight)
        if self.custom_weights is not None:
            for i, edge in enumerate(self.weights):
                self.weights[edge] = self.custom_weights[i]
        else:
            for edge in self.graph.edges():
                u, v = edge
                if u in self.repo_nodes or v in self.repo_nodes:
                    self.weights[edge] += self.max_weight * self.scale_repo_weight
        for u, v, d in self.graph.edges(data=True):
            d['weight'] = self.weights[(u, v)]
        logging.debug('Weights:', '-'.join(list([str(v) for v in self.weights.values()])))
        _shortest_paths = dict(
            [((s, t), nx.shortest_path(self.graph, source=s, target=t)) for s in self.query_nodes for t in
             range(self.graph_size)])

        self.shortest_paths = np.array([[s, t, nx.path_weight(self.graph, _shortest_paths[(s, t)], 'weight')]
                                        for s in self.query_nodes for t in
                                        range(self.graph_size)])
        query_node_candidates = {}
        for query_node in self.query_nodes:
            for item in range(self.catalog_size):
                item_source = self.item_sources[item]
                mask = np.logical_and(self.shortest_paths[:, 0] == query_node, self.shortest_paths[:, 1] == item_source)
                repo_cost = self.shortest_paths[mask][0][2]
                mask = np.logical_and(self.shortest_paths[:, 0] == query_node, self.shortest_paths[:, 2] < repo_cost)
                costs = self.shortest_paths[mask, 2]
                candidates = self.shortest_paths[mask, 1]
                order = np.argsort(costs)
                query_node_candidates[(query_node, item)] = np.array(
                    [np.hstack((candidates[order], item_source)), np.hstack((costs[order], repo_cost))])
        self.query_node_candidates = query_node_candidates
        self.sources_items = inv_dict(self.item_sources)
        self.diamX = 0
        self.diamXVec = np.zeros(self.graph_size)
        for n in range(self.graph_size):

            self.mask_caches[n] = np.ones(self.catalog_size).astype(bool)
            if n in self.sources_items:
                self.mask_caches[n][self.sources_items[n]] = False
            self.integral_caches[n] = np.zeros(self.catalog_size)
            self.fractional_caches[n] = np.ones(self.catalog_size) * self.capacities[n] / np.max(
                [1, np.sum(self.mask_caches[n])])
            self.fractional_caches[n][~self.mask_caches[n]] = 1
            self.integral_caches[n][~self.mask_caches[n]] = 1
            K, N = self.capacities[n], self.catalog_size - sum(~self.mask_caches[n])
            if N > 0:
                self.diamX += K * (1 - K / N)
                self.diamXVec[n] = K * (1 - K / N)
        self.diamX = np.sqrt(self.diamX)
        self.diamXVec = np.sqrt(self.diamXVec)
        self.rates = {n: np.mean(self.query_nodes_trace[n], axis=0) for n in self.query_nodes}
        self.compute_max_gain()
        self.compute_opt(self.alpha)
        if not self.cached_offline_results:
            self.compute_opt(0.0, construct_pareto_front=self.construct_pareto_front)
        self.stats_static['allocs_mask'] = self.mask_caches
        self.stats_static['allocs_opt'] = self.fractional_caches_opt
        self.stats_static['graph'] = self.graph
        self.stats_static['query_nodes'] = self.query_nodes
        self.stats_static['players'] = self.players
        self.stats_static['repo_nodes'] = self.repo_nodes
        if not self.cached_offline_results and self.construct_utility_point_cloud:
            self.stats_static['utility_point_cloud'] = self.construct_point_cloud()
        super().__init__()

    def compute_opt(self, alpha, construct_pareto_front=False):
        # Create variables
        if construct_pareto_front:
            Ws = np.arange(0, 1 + 1 / self.n_pareto_front, 1 / self.n_pareto_front)
            Ws = np.array([Ws, 1 - Ws]).T
            Ws[Ws == 0] = 0.001
            Ws[Ws == 1] = 1 - 0.001
        else:
            Ws = [np.ones(self.players)]
        for W in Ws:
            X = []
            C = []
            for n in np.arange(self.graph_size):
                xs = []
                for item in np.arange(self.catalog_size):
                    if self.item_sources[item] == n:
                        xs.append(1.0)
                    else:
                        x = cp.Variable(nonneg=True)
                        xs.append(x)
                        C.append(x <= 1)
                X.append(xs)
                C.append(cp.sum(xs) <= self.capacities[n] + sum([1 if type(x) is float else 0 for x in xs]))

            Gs = [0] * self.players
            for query_node in self.query_nodes:
                scale = 1 if not self.telescope_requests else 1. / len(self.query_nodes)
                rates = np.mean(self.query_nodes_trace[query_node], axis=0) * scale * self.utility_weight
                for item in np.where(rates != 0)[0]:
                    candidates = np.array(self.query_node_candidates[(query_node, item)][0]).astype(int)
                    states = [X[n][item] for n in candidates]
                    costs = np.array(self.query_node_candidates[(query_node, item)][1])
                    for i in range(len(costs) - 1):
                        sum_x = cp.sum(states[:i + 1])
                        Gs[self.cache_owners[query_node]] += rates[item] * (costs[i + 1] - costs[i]) * cp.min(
                            cp.hstack((1.0, sum_x)))
            objective = 0
            for i, G in enumerate(Gs):
                if alpha == 1:
                    objective += cp.log(G - self.external_disagreement_points[i])
                elif alpha == 0:
                    objective += (G - self.external_disagreement_points[i]) * W[i]
                else:
                    objective += ((cp.power((G - self.external_disagreement_points[i]), (1 - alpha))) * W[i] - 1) / (
                            1 - alpha)

            problem = cp.Problem(cp.Maximize(objective), C)
            problem.solve(solver='SCS')
            logging.debug(f'{problem.status} - problem solved alpha = {alpha}, w = {W}: {[G.value for G in Gs]}')
            self.fractional_caches_opt = np.zeros((self.graph_size, self.catalog_size))
            for n in range(self.fractional_caches_opt.shape[0]):
                for i in range(self.fractional_caches_opt.shape[1]):
                    self.fractional_caches_opt[n][i] = X[n][i].value if type(X[n][i]) is not float else X[n][i]
            self.stats_static[f'opt-{alpha}-{W[0]}'] = [G.value for G in Gs]

    def compute_max_gain(self):
        C_repo = np.zeros(self.players)
        for query_node in self.query_nodes:
            owner = self.cache_owners[query_node]
            scale = 1 if not self.telescope_requests else 1. / len(self.query_nodes)
            batch = self.rates[query_node] * scale
            for item in np.where(batch != 0)[0]:
                costs = np.array(self.query_node_candidates[(query_node, item)][1])
                C_repo[owner] += (costs[-1]) * batch[item]
        self.stats_static['max_gain'] = C_repo
        self.utility_weight = 1 / np.max(C_repo)

    def average_gain(self):
        G_integral = np.zeros(self.players)
        G_opt_fractional = np.zeros(self.players)
        G_fractional = np.zeros(self.players)
        C_repo = np.zeros(self.players)
        query_nodes = self.query_nodes

        for query_node in query_nodes:
            owner = self.cache_owners[query_node]
            scale = 1 if not self.telescope_requests else 1. / len(query_nodes)
            batch = self.rates[query_node] * scale * self.utility_weight
            for item in np.where(batch != 0)[0]:

                candidates = np.array(self.query_node_candidates[(query_node, item)][0]).astype(int)
                candidates_integral_states = np.array([self.integral_caches[n][item] for n in
                                                       candidates])
                candidates_fractional_states = np.array([self.fractional_caches[n][item] for n in
                                                         candidates])
                candidates_opt_fractional_states = np.array([self.fractional_caches_opt[n][item] for n in
                                                             candidates])
                costs = np.array(self.query_node_candidates[(query_node, item)][1])
                x_i = np.where(candidates_integral_states == 1)[0][0]
                G_integral[owner] += (costs[-1] - costs[x_i]) * batch[item]
                for i in range(len(costs) - 1):
                    sum_x = np.sum(candidates_fractional_states[:i + 1])
                    sum_x_opt = np.sum(candidates_opt_fractional_states[:i + 1])
                    G_fractional[owner] += batch[item] * (costs[i + 1] - costs[i]) * np.min(
                        [1, sum_x])
                    G_opt_fractional[owner] += batch[item] * (costs[i + 1] - costs[i]) * np.min(
                        [1, sum_x_opt])
                C_repo[owner] += (costs[-1]) * batch[item]
        self.stats_dynamic['average_fractional_gain'].append(G_fractional)
        self.stats_dynamic['average_integral_gain'].append(G_integral)
        self.stats_dynamic['average_repo_cost'].append(C_repo)
        self.stats_dynamic['average_opt_fractional_gain'].append(G_opt_fractional)

    def gain(self, adapt_state=True, increment_time=True):
        G_integral = np.zeros(self.players)
        G_opt_fractional = np.zeros(self.players)
        G_fractional = np.zeros(self.players)
        C_repo = np.zeros(self.players)
        if increment_time:
            self.t += 1
        if self.telescope_requests:
            query_nodes = [np.random.choice(self.query_nodes)]
        else:
            query_nodes = self.query_nodes



        for query_node in query_nodes:
            owner = self.cache_owners[query_node]
            batch = self.query_nodes_trace[query_node][self.t - 1] * self.utility_weight
            for item in np.where(batch != 0)[0]:
                candidates = np.array(self.query_node_candidates[(query_node, item)][0]).astype(int)
                candidates_integral_states = np.array([self.integral_caches[n][item] for n in
                                                       candidates])
                candidates_fractional_states = np.array([self.fractional_caches[n][item] for n in
                                                         candidates])
                candidates_opt_fractional_states = np.array([self.fractional_caches_opt[n][item] for n in
                                                             candidates])
                costs = np.array(self.query_node_candidates[(query_node, item)][1])
                x_i = np.where(candidates_integral_states == 1)[0][0]
                G_integral[owner] += (costs[-1] - costs[x_i]) * batch[item]
                for i in range(len(costs) - 1):
                    sum_x = np.sum(candidates_fractional_states[:i + 1])
                    sum_x_opt = np.sum(candidates_opt_fractional_states[:i + 1])
                    if sum_x <= 1 and adapt_state:
                        self.subgradients[owner][candidates[:i + 1], item] += batch[item] * (costs[i + 1] - costs[i])
                    G_fractional[owner] += batch[item] * (costs[i + 1] - costs[i]) * np.min(
                        [1, sum_x])
                    G_opt_fractional[owner] += batch[item] * (costs[i + 1] - costs[i]) * np.min(
                        [1, sum_x_opt])
                C_repo[owner] += (costs[-1]) * batch[item]
        self.stats_dynamic['fractional_gain'].append(G_fractional)
        self.stats_dynamic['integral_gain'].append(G_integral)
        self.stats_dynamic['repo_cost'].append(C_repo)
        self.stats_dynamic['opt_fractional_gain'].append(G_opt_fractional)
        if adapt_state:
            self.adapt_state()

    @abstractmethod
    def adapt_state(self):
        pass

    def construct_point_cloud(self):
        for i in range(self.n_utility_point_cloud):
            self.fractional_caches_original = np.copy(self.fractional_caches)
            for n in range(self.graph_size):
                if n not in self.repo_nodes:
                    self.fractional_caches[n] = sample_simplex(self.catalog_size, self.capacities[n])
            self.average_gain()
            self.fractional_caches = np.copy(self.fractional_caches_original)
        self.stats_static['feasibility_point_cloud'] = self.stats_dynamic['average_fractional_gain']
        self.stats_dynamic['average_fractional_gain'] = []
        self.stats_dynamic['average_integral_gain'] = []
        self.stats_dynamic['average_repo_cost'] = []
        self.stats_dynamic['average_opt_fractional_gain'] = []
