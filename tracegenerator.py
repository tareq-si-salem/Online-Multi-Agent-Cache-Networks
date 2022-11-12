import argparse
import os
import pickle
import random

import numpy as np

argparser = argparse.ArgumentParser(description='Trace generator')
argparser.add_argument('--time_horizon', default=5_000, type=int, help='Time horizon')
argparser.add_argument('--catalog_size', default=20, type=int, help='Catalog size')
argparser.add_argument('--distribution_roll', default=0, type=int, help='Distribution roll')
argparser.add_argument('--batch_min_size', default=100, type=int, help='Batch min size')
argparser.add_argument('--batch_max_size', default=100, type=int, help='Batch max size')
argparser.add_argument('--zipfs_exponent', default=1.2, type=float, help="Zipfs law exponent")
argparser.add_argument('--random_seed', default=42, type=int, help='Random seed')
argparser.add_argument('--traces_dir', default='traces/', type=str, help='Trace output')
argparser.add_argument('--adversarial_1', action='store_true', help='Adversarial trace type')
argparser.add_argument('--adversarial_2', action='store_true', help='Adversarial trace type')
argparser.add_argument('--shuffle', action='store_true', help='Shuffle the request process. Convert any trace to a stationary one, but maintaining the same '
                                                              'static optimum')


def zipf_distribution(s, N):
    c = sum((1 / np.arange(1, N + 1) ** s))
    return np.arange(1, N + 1) ** (-s) / c


def to_vec(x, catalog_size):
    y = np.zeros(catalog_size)
    y[x.R] = x.t
    return y


if __name__ == '__main__':
    args = argparser.parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    N = args.catalog_size
    T = args.time_horizon
    s = args.zipfs_exponent
    batches = np.random.uniform(args.batch_min_size, args.batch_max_size, size=T).astype(int)
    dist = zipf_distribution(s, N)
    if args.distribution_roll > 0:
        dist = np.roll(dist, args.distribution_roll)
    requests = []
    R = []

    dist1 = zipf_distribution(s, N)
    dist2 = zipf_distribution(s // 2, N)
    dist2 = np.roll(dist2, 10)
    D = 50
    R = []
    for i in range(sum(batches) // (D * 2)):
        rs = np.random.choice(np.arange(N), p=dist1, size=D)
        R.extend(list(rs))
        rs = np.random.choice(np.arange(N), p=dist2, size=D)
        R.extend(list(rs))
    if args.shuffle:
        np.random.shuffle(R)
    if args.adversarial_2:
        R = R[D:] + R[:D]
    for i, b in enumerate(batches):
        if args.adversarial_1 or args.adversarial_2:
            rs = R[(sum(batches[:i])):(sum(batches[:i])) + b]
        else:
            rs = list(np.random.choice(np.arange(N), p=dist, size=b))
        rs_vec = np.zeros(N)
        for c in np.unique(rs):
            rs_vec[c] = rs.count(c)
        requests.append(rs_vec)
    requests = np.array(requests).astype(int)
    if args.adversarial_1:
        name = f'trace_catalog_{N}_T_{T}_B_{args.batch_min_size}_{args.batch_max_size}_s_{args.zipfs_exponent}_roll_{args.distribution_roll}_adv_1.pk'
    elif args.adversarial_2:
        name = f'trace_catalog_{N}_T_{T}_B_{args.batch_min_size}_{args.batch_max_size}_s_{args.zipfs_exponent}_roll_{args.distribution_roll}_adv_2.pk'
    else:
        name = f'trace_catalog_{N}_T_{T}_B_{args.batch_min_size}_{args.batch_max_size}_s_{args.zipfs_exponent}_roll_{args.distribution_roll}.pk'
    pickle.dump(requests, open(
        os.path.join(args.traces_dir, name),
        "wb"))
