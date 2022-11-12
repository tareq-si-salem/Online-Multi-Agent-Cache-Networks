import logging
import random

import cvxpy as cp
import networkx as nx
import numpy as np

import topologies


def graphGenerator(args):
    temp_graph = None
    if args.graph_type == "path":
        temp_graph = nx.Graph()
        temp_graph.add_node(0)
        for n in range(1, args.graph_size):
            temp_graph.add_node(n)
            temp_graph.add_edge(n - 1, n)
        logging.debug(temp_graph)
    if args.graph_type == "erdos_renyi":
        temp_graph = nx.erdos_renyi_graph(args.graph_size, args.graph_p)
    if args.graph_type == "balanced_tree":
        ndim = int(np.ceil(np.log(args.graph_size) / np.log(args.graph_degree)))
        temp_graph = nx.balanced_tree(args.graph_degree, ndim)
    if args.graph_type == "cicular_ladder":
        ndim = int(np.ceil(args.graph_size * 0.5))
        temp_graph = nx.circular_ladder_graph(ndim)
    if args.graph_type == "cycle":
        temp_graph = nx.cycle_graph(args.graph_size)
    if args.graph_type == 'grid_2d':
        ndim = int(np.ceil(np.sqrt(args.graph_size)))
        temp_graph = nx.grid_2d_graph(ndim, ndim)
    if args.graph_type == 'lollipop':
        ndim = int(np.ceil(args.graph_size * 0.5))
        temp_graph = nx.lollipop_graph(ndim, ndim)
    if args.graph_type == 'expander':
        ndim = int(np.ceil(np.sqrt(args.graph_size)))
        temp_graph = nx.margulis_gabber_galil_graph(ndim)
    if args.graph_type == "hypercube":
        ndim = int(np.ceil(np.log(args.graph_size) / np.log(2.0)))
        temp_graph = nx.hypercube_graph(ndim)
    if args.graph_type == "star":
        ndim = args.graph_size - 1
        temp_graph = nx.star_graph(ndim)
    if args.graph_type == 'barabasi_albert':
        temp_graph = nx.barabasi_albert_graph(args.graph_size, args.graph_degree)
    if args.graph_type == 'watts_strogatz':
        temp_graph = nx.connected_watts_strogatz_graph(args.graph_size, args.graph_degree, args.graph_p)
    if args.graph_type == 'regular':
        temp_graph = nx.random_regular_graph(args.graph_degree, args.graph_size)
    if args.graph_type == 'powerlaw_tree':
        temp_graph = nx.random_powerlaw_tree(args.graph_size)
    if args.graph_type == 'small_world':
        ndim = int(np.ceil(np.sqrt(args.graph_size)))
        temp_graph = nx.navigable_small_world_graph(ndim)
    if args.graph_type == 'geant':
        temp_graph = topologies.GEANT()
    if args.graph_type == 'dtelekom':
        temp_graph = topologies.Dtelekom()
    if args.graph_type == 'abilene':
        temp_graph = topologies.Abilene()
    if args.graph_type == 'servicenetwork':
        temp_graph = topologies.ServiceNetwork()

    number_map = dict(list(zip(temp_graph.nodes(), list(range(len(temp_graph.nodes()))))))
    graph = nx.Graph()
    graph.add_nodes_from(list(number_map.values()))
    for (x, y) in temp_graph.edges():
        xx = number_map[x]
        yy = number_map[y]
        graph.add_edges_from(((xx, yy), (yy, xx)))
    return graph


def listify(arg, type):
    if arg == '_':
        return []
    if len(arg) == 0:
        return
    out = list(map(type, arg.split('-')))
    return out


def zipf_distribution(s, N):
    c = sum((1 / np.arange(1, N + 1) ** s))
    return np.arange(1, N + 1) ** (-s) / c


def inv_dict(dictionary):
    inv_dictionary = {}
    for k, v in dictionary.items():
        inv_dictionary[v] = inv_dictionary.get(v, []) + [k]
    return inv_dictionary


def refresh_weights(edges, min_weight, max_weight):
    weights = {}
    for (x, y) in edges:
        weights[(x, y)] = (random.uniform(min_weight, max_weight))
    return weights


def _is_feasible(x, k):
    return np.sum(x) <= k and np.all(x <= 1) and np.all(x >= 1)


def sample_simplex(N, K):
    z = np.zeros(N)
    per = np.arange(z.size)
    np.random.shuffle(per)
    for i in per:
        x = np.random.uniform(0, np.min([K - sum(z), 1]))
        z[i] = x
        if sum(z) >= K:
            break
    return z


def round(x, xi=.5):
    permutation = np.arange(x.size)
    sum = 0
    I = []
    for i in range(x.size):
        sum += x[permutation[i]]
        if sum - len(I) >= xi:
            I.append(permutation[i])
    z = np.zeros(x.size)
    z[np.array(I)] = 1
    return z


# CVXPY-based projection
class EuclideanProjection:
    def __init__(self, catalog_size, cache_size):
        self.catalog_size = catalog_size
        self.cache_size = cache_size
        if catalog_size == 0:
            return
        x = cp.Variable(catalog_size, nonneg=True)
        y_param = cp.Parameter(catalog_size)
        constraints = [x <= 1, cp.sum(x) <= cache_size]
        obj = cp.Minimize(cp.sum_squares(x - y_param))  # ( cp.sum((x - y_param) ** 2))
        prob = cp.Problem(obj, constraints)
        self.prob = prob
        self.y_param = y_param
        self.x = x

    def project(self, y, warm_start=True):
        self.y_param.value = y
        self.prob.solve(warm_start=warm_start, solver='MOSEK')
        return self.x.value


# Manual projection
class EuclideanProjection:
    def __init__(self, catalog_size, cache_size):
        self.catalog_size = catalog_size
        self.cache_size = cache_size
        self._y = np.zeros(self.catalog_size + 2)
        self.inv_map = np.arange(catalog_size)
        self.a, self.b = 0, catalog_size  # initial kkt params predict
        self.delta = 1
        self.init = True

    def _check_KKT(self, range_a, b_max, z, map, inv_map):
        D, y, k = self.catalog_size, self._y, self.cache_size
        y[D + 1] = np.inf
        y[0] = -np.inf
        y[1:D + 1] = np.copy(z[map])
        delta = 1e-16
        for a in range_a:
            if k == D - a and y[a + 1] - y[a] >= 1 - delta:
                b = a
                y[:a + 1] = 0
                y[b + 1:] = 1
                self.a = a
                self.b = b
                return y[1:1 + D][inv_map]
            for b in range(a + 1, b_max):
                Tks = np.sum(y[a + 1:b + 1])
                g = (k + b - D - Tks) / (b - a)
                if y[a] + g <= delta and y[a + 1] + g > -delta and y[b] + g < 1 + delta and y[b + 1] + g >= 1 - delta:
                    y = y + g
                    y[:a + 1] = 0
                    y[b + 1:] = 1
                    self.a = a
                    self.b = b
                    return y[1:1 + D][inv_map]

    def project(self, z):
        D = self.catalog_size
        self.map = np.argsort(z)
        self.inv_map[self.map] = np.arange(D).astype(int)
        self.init = False
        range_a = range(0, D + 2)
        b_max = D + 1
        return self._check_KKT(range_a, b_max, z, self.map, self.inv_map)
