import logging
import os.path
import pickle
import random
import time

import numpy as np

from args import argparser
from policies.LFUCacheNetwork import LFUCacheNetwork
from policies.LRUCacheNetwork import LRUCacheNetwork
from policies.OnlineFairCacheNetwork import OnlineFairCacheNetwork
from policies.OnlineSlotFairCacheNetwork import OnlineSlotFairCacheNetwork
from tools import graphGenerator, zipf_distribution, inv_dict, listify


class TimeProbe:
    def __init__(self):
        self.t = 0

    def start(self):
        self.t = time.perf_counter()

    def record(self):
        if self.t != 0:
            return time.perf_counter() - self.t
        else:
            return np.nan


if __name__ == '__main__':
    args = argparser.parse_args()

    logging.basicConfig( level=args.debug_level)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    args.query_nodes = listify(args.query_nodes, int)
    args.traces = listify(args.traces, str)
    logging.debug(args.traces)
    if args.external_disagreement_points == '':
        args.external_disagreement_points = ('0-' * args.players)[:-1]
    logging.debug(args.external_disagreement_points)
    args.external_disagreement_points = listify(args.external_disagreement_points, float)
    args.custom_weights = listify(args.custom_weights, float)
    logging.debug(args.custom_weights)
    args.umin_umax = listify(args.umin_umax, float)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    graph = graphGenerator(args)
    graph_size = graph.number_of_nodes()
    args.graph_size = graph_size
    edge_size = graph.number_of_edges()
    capacities = dict((x, random.randint(args.min_capacity, args.max_capacity)) for x in graph.nodes())
    if args.graph_type == 'balanced_tree':
        sources = [0]
    elif args.graph_type == 'cycle':
        sources = [2]
    else:
        sources = list(
            random.sample(set(np.arange(graph_size)), k=1 if args.graph_type == 'cycle' else args.repo_nodes))
    owners = {}
    caches_I = list(set(np.argsort(list(dict(graph.degree).values()))).difference(sources))
    if args.graph_type == 'balanced_tree':
        leaves = list(filter(lambda x: graph.degree[x] == 1, caches_I))
    else:
        leaves = []
    for n in caches_I:
        if not n in leaves:
            owners[n] = np.random.choice(np.arange(args.players),
                                         p=zipf_distribution(args.resources_bias, args.players))

    for i, n in enumerate(leaves):
        owners[n] = i % args.players

    cache_owners = dict([(c, owners[c]) for c in caches_I])
    owners_cache = inv_dict(cache_owners)
    for n in caches_I:
        graph.nodes()[n]['owner'] = cache_owners[n]
    for n in sources:
        graph.nodes()[n]['owner'] = -1
    if len(args.query_nodes) != 1:
        query_nodes = list(set(args.query_nodes))
    else:
        query_nodes = []
        N = np.copy(args.query_nodes)[0]
        if args.graph_type == 'balanced_tree':
            for i in range(args.players):
                leaf = list(filter(lambda x: graph.degree[x] == 1, owners_cache[i]))
                query_nodes = np.hstack((query_nodes, random.sample(leaf, min([N, len(leaf)]))))
        else:
            for i in range(args.players):
                query_nodes = np.hstack((query_nodes, random.sample(owners_cache[i], N)))
    query_nodes = query_nodes.astype(int)
    args.query_nodes = query_nodes.astype(int)
    traces = []
    for trace_loc in args.traces:
        if trace_loc == '':
            continue
        with open(trace_loc, 'rb') as f:
            traces.append(pickle.load(f))
    query_nodes_trace = dict([(q, traces[owners[q] % len(traces)]) for q in query_nodes])
    logging.debug(f'Graph size: {args.graph_size}')

    np.random.shuffle(sources)
    item_sources = dict([(item, np.random.choice(sources[:args.repo_nodes])) for item in np.arange(args.catalog_size)])
    logging.debug('Sources:', sources)
    network_properties = {
        'graph_size': graph_size,
        'catalog_size': args.catalog_size,
        'graph': graph,
        'capacities': capacities,
        'item_sources': item_sources,
        'cache_owners': cache_owners,
        'query_nodes': query_nodes,
        'query_nodes_trace': query_nodes_trace,
        'players': args.players,
        'alpha': args.alpha,
        'external_disagreement_points': args.external_disagreement_points,
        'repo_nodes': sources[:args.repo_nodes],
        'construct_pareto_front': args.construct_pareto_front,
        'custom_weights': args.custom_weights,
        'telescope_requests': args.telescope_requests,
        'scale_repo_weight': args.scale_repo_weight,
        'umin_umax': args.umin_umax,
        'construct_utility_point_cloud': args.construct_utility_point_cloud,
        'n_utility_point_cloud': args.n_utility_point_cloud,
        'n_pareto_front': args.n_pareto_front,
        'cached_offline_results': args.cached_offline_results,
        'min_weight': args.min_weight,
        'max_weight': args.max_weight,
        'fairslotted_freeze_period': args.fairslotted_freeze_period

    }
    if args.cache_type == 'fair':
        network = OnlineFairCacheNetwork(network_properties)
    elif args.cache_type == 'fairslotted':
        network = OnlineSlotFairCacheNetwork(network_properties)
    elif args.cache_type == 'lru':
        network = LRUCacheNetwork(network_properties)
    elif args.cache_type == 'lfu':
        network = LFUCacheNetwork(network_properties)
    time_probe = TimeProbe()

    if args.record_offline_stats_only:
        network.stats_static['args'] = args
        with open(args.output + os.path.sep + args.experiment_name + '_static_.pk', 'wb') as f:
            pickle.dump((args, network.stats_static), f)
    else:
        if args.cached_offline_results:
            with open(args.output + os.path.sep + args.experiment_name + '_static_.pk', 'rb') as f:
                _, stats_static = pickle.load(f)
        else:
            with open(args.output + os.path.sep + args.experiment_name + '_static_.pk', 'wb') as f:
                pickle.dump((args, network.stats_static), f)
        network.stats_dynamic[f'opt-{args.alpha}-1.0'] = network.stats_static[f'opt-{args.alpha}-1.0']
        for t in range(args.time_horizon):
            network.average_gain()
            network.gain()
            if t % 100 == 0:
                logging.info(
                    f'\rProgress: {t / args.time_horizon * 100:.1f} % | Time: {time_probe.record()}, Check: {np.min(network.fractional_caches)}')
                time_probe.start()
        with open(args.output + os.path.sep + args.experiment_name + '_' + args.experiment_subname + '_dynamic_.pk',
                  'wb') as f:
            network.stats_dynamic['args'] = args
            pickle.dump(network.stats_dynamic, f)
    logging.info('finished')
