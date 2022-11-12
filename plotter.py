import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cycler
from scipy.spatial import ConvexHull

from tools import listify

argparser = argparse.ArgumentParser(description='Trace Generator')
argparser.add_argument('--task', default='2dplot', type=str,
                       choices=['2dplotnopolicies', '2dplot', 'barplot', 'barplot-multiplayer',
                                'barplot-multiplayer-pof',
                                'barplot-multiplayer-temp', 'barplotopt',
                                'plotopt', 'topology_draw'], help='')
argparser.add_argument('--output', default='./test/fig.pdf', type=str, help='')
argparser.add_argument('--input_dir', default='./res/2players-1-50/', type=str, help='')
argparser.add_argument('--policies', default='fair-fairslotted-lfu-lru', type=str, help='')
argparser.add_argument('--params', default='3.0', type=str, help='')
argparser.add_argument('--ylim', default='0.4-0.6', type=str, help='')
argparser.add_argument('--xlim', default='0.4-0.5', type=str, help='')
argparser.add_argument('--alpha', default=0.0, type=float, help='')
argparser.add_argument('--param_is_disagreement', action='store_true')
argparser.add_argument('--player', default=2, type=int, help='')
argparser.add_argument('--show_single', action='store_true', help='')
argparser.add_argument('--sort_direction', default=1, type=int, help='')
argparser.add_argument('--legend', action='store_true', help='')

plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['axes.prop_cycle'] = cycler('color',
                                         ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e'])
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
sns.set_context(rc={'patch.linewidth': 0.5})


def tavg(a, sampler=1):
    I = np.arange(0, len(a), sampler).astype(int)
    return (np.cumsum(a) / np.arange(1, len(a) + 1))[I]


def f_alpha(x, alpha=1):
    if alpha != 1:
        return (x ** (1 - alpha)) / (1 - alpha)
    else:
        return np.log(x)


def load_data(dir):
    stats = {}
    for file in os.listdir(dir):
        with open(dir + file, 'rb') as f:
            if 'static' in file:
                args, res = pickle.load(f)
            else:
                res = pickle.load(f)
            stats[tuple(file.split('_')[:-1])] = res
    return stats


color_id = {
    'lfu': 3,
    'lru': 2,
    'fairslotted': 1,
    'fair': 0,
}
display_name = {'lfu': r'$\tt{LFU}$',
                'lru': r'$\tt{LRU}$',
                'fair': r'$\tt{OHF}$',
                'fairslotted': r'$\tt{OSF}$',
                }
topology_display_name = {'geant': r'$\tt{GEANT}$', 'tree': r'  $\tt{Tree}$', 'abilene': r' $\tt{Abilene}$', '2d_grid': r'   $\tt{Grid}$'}
if __name__ == '__main__':
    args = argparser.parse_args()
    args.policies = listify(args.policies, str)
    args.params = listify(args.params, str)
    args.xlim = listify(args.xlim, float)
    args.ylim = listify(args.ylim, float)
    fig = plt.figure(figsize=(3, 3))
    if 'barplot-multiplayer' in args.task:
        stats = load_data(args.input_dir.replace('x', '2'))
    elif args.task == 'barplot':
        stats = []
    else:
        stats = load_data(args.input_dir)
    exps = list(stats)
    if args.task == 'barplotopt':
        xs = []
        ys_eff = []
        ys_fair = []
        for i, exp in enumerate(exps):
            stat_static = stats[exp]
            _args = stat_static['args']
            xs.append(str(_args.custom_weights))
            ys_eff.append(stat_static['opt-0.0-0.5'])
            ys_fair.append(stat_static[f'opt-{_args.alpha}-1.0'])
        plt.bar(np.array(range(len(xs))) - 0.11, np.
                array(ys_eff)[:, 0], 0.1, label=rf'Utility agent 1, $\alpha = 0$')
        plt.bar(np.array(range(len(xs))) - 0.11, np.array(ys_eff)[:, 1], 0.1, label=rf'Utility agent 2, $\alpha = 0$',
                bottom=np.array(ys_eff)[:, 0])
        plt.bar(np.array(range(len(xs))) + 0.11, np.array(ys_fair)[:, 0], 0.1,
                label=rf'Utility agent 1, $\alpha = {_args.alpha}$')
        plt.bar(np.array(range(len(xs))) + 0.11, np.array(ys_fair)[:, 1], 0.1,
                label=rf'Utility agent 2, $\alpha = {_args.alpha}$', bottom=np.array(ys_fair)[:, 0])
        fig.get_axes()[0].set_xticks(np.array(range(len(xs))), xs, rotation=90)
        legend = plt.legend(ncol=1, frameon=True, prop={'size': 5})
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_linewidth(0.5)
    if args.task == '2dplotnopolicies':
        fig = plt.figure(figsize=(3, 2.5))
        label0 = None
        label1 = None
        seen = []
        labels_alphas = {}
        index = np.argsort([float(exp[0].split('+')[-1]) for exp in exps])
        if args.show_single:
            index = index[:1]
        _show_single = True
        _i = 0
        lines = {}
        lines0 = []
        _colors = {}
        _xlim = []
        _ylim = []
        for i in index:
            exp = exps[i]
            stat_static = stats[exp]
            _args = stat_static['args']
            scale = [1, 1]
            param, alpha = exp[0].split('+')
            alpha = float(alpha)

            if not alpha in labels_alphas.keys():
                labels_alphas[alpha] = rf'OPT $\alpha = {_args.alpha}$'
                _colors[alpha] = f'C{_i}'
                lines[alpha] = []
                _i += 1

            else:
                labels_alphas[alpha] = '_nolegend_'
            seen.append(param)
            if label1 is None:
                label0 = r'OPT $\alpha = 0$'
                label1 = rf'OPT $\alpha = {_args.alpha}$'
            if 'traces' in args.input_dir:
                name = f'Setting: distribution shifts = 0, {param}'
            elif 'retrievalcosts' in args.input_dir:
                name = f"Setting: retrieval costs = {param.replace('-', ', ')}"
            elif 'exponents' in args.input_dir:
                name = f'Setting: distribution exponents = 1.2, {param}'
            opt0 = stat_static['opt-0.0-0.5']
            opt1 = stat_static[f'opt-{_args.alpha}-1.0']
            plt.scatter([opt0[0]], [opt0[1]], color='k', marker='*', label=label0)
            plt.scatter([opt1[0]], [opt1[1]], color=_colors[alpha], marker='*', label=labels_alphas[alpha])
            lines[alpha].append([opt1[0], opt1[1]])
            lines0.append([opt0[0], opt0[1]])
            logging.debug(stat_static[f'opt-{_args.alpha}-1.0'])
            opt0 = stat_static['opt-0.0-0.5']
            label0 = '_nolegend_'

        params = [(exp[0].split('+')[0]) for exp in exps]

        _i = 0
        for param in np.sort(np.unique(params)):
            for exp in exps:
                if (exp[0].split('+')[0]) == param:
                    stat_static = stats[exp]
                    pareto_keys = filter(lambda key: 'opt-0.0-' in key, stat_static.keys())
                    us_pareto = np.array([stat_static[key] for key in pareto_keys])

                    _xlim = [np.min(us_pareto[:, 0]), np.max(us_pareto[:, 0])]
                    _ylim = [np.min(us_pareto[:, 1]), np.max(us_pareto[:, 1])]
                    if label1 is None:
                        label0 = r'$\alpha = 0$'
                        label1 = rf'$\alpha = {_args.alpha}$'
                    if 'retrievalcosts' in args.input_dir:
                        label = f"Pareto front: $w_{{(1,3)}} = {param.split('-')[-1]}$"
                    elif 'exponents' in args.input_dir:
                        label = rf'Pareto front: $\sigma = {param}$'
                    plt.plot(us_pareto[:, 0], us_pareto[:, 1], color=f'k', linestyle='--', linewidth=_i * 0.4 + .75,
                             alpha=.75, label=label)

                    _i += 1
                    break
        us = np.array(lines0)
        _index = np.argsort(us[:, 1])
        plt.plot(us[_index, 0], us[_index, 1], color=f'k')
        for i, alpha in enumerate(lines):
            us = np.array(lines[alpha])
            _index = np.argsort(us[:, args.sort_direction])
            plt.plot(us[_index, 0], us[_index, 1], color=_colors[alpha])
        plt.xlabel('Utility agent 1')
        plt.ylabel('Utility agent 2')
        legend = plt.legend(ncol=1, frameon=True, prop={'size': 5})
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_linewidth(0.5)
    elif args.task == 'plotopt':
        label0 = None
        label1 = None
        xs = []
        alphas = []
        ys_eff = []
        ys_fair = []

        for i, exp in enumerate(exps):
            stat_static = stats[exp]
            _args = stat_static['args']
            scale = [1, 1]
            param, alpha = exp[0].split('+')
            opt0 = stat_static['opt-0.0-0.5']
            opt1 = stat_static[f'opt-{_args.alpha}-1.0']
            xs.append(float(param.split('-')[-1]))
            alphas.append(alpha)
            ys_eff.append(opt0)
            ys_fair.append(opt1)
        xs = np.array(xs)
        alphas = np.array(alphas)
        ys_eff = np.array(ys_eff)
        ys_fair = np.array(ys_fair)

        for i, alpha in enumerate(np.sort(np.unique(alphas))):
            mask = alphas == alpha
            indx = np.argsort(xs[mask])
            logging.debug(indx)
            plt.plot(xs[mask][indx], (ys_fair[mask][indx])[:, 0] + (ys_fair[mask][indx])[:, 1],
                     label=fr'$\alpha = {alpha}$', color=f'C{i}')

        plt.plot(xs[mask][indx], (ys_eff[mask][indx])[:, 0] + (ys_eff[mask][indx])[:, 1], label=fr'$\alpha = {0}$',
                 color='k')

        plt.xlabel('Parameter $\sigma$')
        plt.ylabel('Utility ')
    elif args.task == '2dplot':
        stat_static = stats[('exp', 'static')]
        scale = [1, 1]
        us = np.array(stat_static['feasibility_point_cloud'])
        pareto_keys = filter(lambda key: 'opt-0.0-' in key, stat_static.keys())
        us_pareto = np.array([stat_static[key] for key in pareto_keys])
        pts = np.vstack((us, us_pareto))
        extra_pts = []
        NBS = []
        for i, sub_exp in enumerate(exps):
            if sub_exp[1] == 'static':
                continue
            stat_dynamic = stats[sub_exp]
            alpha = stat_dynamic['args'].alpha
            if 'fair' == sub_exp[1]:
                nbs = [stat_dynamic[f'opt-{alpha}-1.0'][0], stat_dynamic[f'opt-{alpha}-1.0'][1]]
                plt.scatter([nbs[0]], [nbs[1]], color='k', marker='*', s=100)
                if not '2players-online' in args.input_dir:
                    plt.text(nbs[0] + 0.005, nbs[1] + 0.005, rf"$u^d={stat_dynamic['args'].external_disagreement_points[-1]}$", fontdict={'size': 7})
                else:
                    plt.text(nbs[0] + 0.005, nbs[1] + 0.005, rf'$\alpha={alpha}$', fontdict={'size': 7})

            U = np.array(stat_dynamic['fractional_gain'])
            u0, u1 = np.mean(U[:, 0]), np.mean(U[:, 1])
            if not (np.isnan(u0) or np.isnan(u1)):
                extra_pts.append([u0, u1])
                extra_pts.append([u1, u0])

        _xlim = [np.min(us_pareto[:, 0]) - 0.025, np.max(us_pareto[:, 0]) + 0.025]
        _ylim = [np.min(us_pareto[:, 1]) - 0.025, np.max(us_pareto[:, 1]) + 0.025]
        plt.plot(us_pareto[:, 0] * scale[0], us_pareto[:, 1] * scale[1], color='k', linestyle='--', linewidth=2,
                 label='Pareto front')
        hull = ConvexHull(pts)
        plt.fill(pts[hull.vertices, 0] * scale[0], pts[hull.vertices, 1] * scale[1], 'whitesmoke', alpha=0.25,
                 label='Feasible region', hatch='\\', edgecolor='gray')
        plt.scatter([nbs[0]], [nbs[1]], color='k', marker='*', label='OPT', s=100)

        for policy in args.policies:
            policy_exps = list(filter(lambda x: x[1] == policy, list(exps)))
            for exp in policy_exps:
                if policy == 'static':
                    continue
                else:
                    name = display_name[policy] if exp[2] == str(args.alpha) or exp[2] == 'dynamic' else '_nolegend_'
                stat_dynamic = stats[exp]
                us = np.array(stat_dynamic['fractional_gain'])
                logging.debug(us.shape)
                plt.plot((tavg(us[:, 0], 10)[1:] * scale[0]), (tavg(us[:, 1], 10)[1:] * scale[1]),
                         alpha=.25,
                         marker='o',
                         color=f'C{color_id[policy]}',
                         ms=1, label=name)

        plt.xlabel('Utility agent 1')
        plt.ylabel('Utility agent 2')
        if args.legend:
            legend = plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.25, 0, 0), frameon=True)
            legend.get_frame().set_edgecolor('k')
            legend.get_frame().set_linewidth(0.5)
            for legobj in legend.legendHandles:
                legobj.set_linewidth(2.0)

        plt.xlim(_xlim)
        plt.ylim(_ylim)
    elif args.task == 'barplot':
        fig = plt.figure(figsize=(3, 2))

        scale = [1, 1]
        data = []
        topologies = ['tree', '2d_grid', 'abilene', 'geant']
        for topology in topologies:
            stats = load_data(args.input_dir.replace('x', topology))
            exps = list(stats)
            stat_static = stats[('exp', 'static')]
            for policy in ['OPT'] + args.policies:
                for param in args.params:

                    if policy != 'OPT':
                        policy_exps = list(
                            filter(lambda x: x[1] == policy and ((x[2] == 'dynamic') or x[2] == param), list(exps)))
                    else:
                        policy_exps = list(
                            filter(lambda x: x[1] == 'fair' and ((x[2] == 'dynamic') or x[2] == param), list(exps)))
                    for exp in policy_exps:
                        stat_dynamic = stats[exp]
                        us = np.array(stat_dynamic['fractional_gain'])
                        N = us.shape[1]
                        u = np.sum([np.mean(us[:, i]) for i in range(us.shape[1])])
                        name = display_name[exp[1]] + param
                        if policy != 'OPT':
                            data.append([topology_display_name[topology], display_name[policy], param, u] + [np.mean(us[:, i]) for i in range(N)])
                        else:
                            alpha = stat_dynamic['args'].alpha
                            logging.debug('OPT', alpha)
                            us_opt = stat_dynamic[f'opt-{alpha}-1.0']
                            uopt = np.sum([us_opt[i] for i in range(len(us_opt))])
                            data.append(
                                [topology_display_name[topology], 'OPT', param, uopt] + [
                                    stat_dynamic[f'opt-{alpha}-1.0'][i] for i in range(len(us_opt))])
        ddf = pd.DataFrame(data, columns=['topology', 'policy', 'param', 'u'] + [f'Agent {i + 1}' for i in range(N)])
        ddf['x'] = ddf.topology + ', ' + ddf.policy

        ax = ddf.sort_values(by=['topology', 'param', 'policy']).drop(columns=['topology', 'policy', 'param', 'u']).set_index('x').plot.bar(stacked=True,
                                                                                                                                            edgecolor='k')
        fig = ax.get_figure()
        plt.xlabel(rf'Topology, Policy')
        plt.ylabel(r'Utility')
        legend = plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.2, 0, 0), frameon=True)
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_linewidth(0.5)
    elif args.task == 'barplot-multiplayer':
        fig = plt.figure(figsize=(3, 2))
        scale = [1, 1]
        data = []
        opt_recorded = False
        for policy in (['OPT1'] + ['OPT0'] + args.policies):
            for param in args.params:
                stats = load_data(args.input_dir.replace('x', param))
                exps = list(stats)
                stat_static = stats[('exp', 'static')]
                if not 'OPT' in policy:
                    policy_exps = list(
                        filter(lambda x: x[1] == policy and ((x[2] == 'dynamic') or x[2] == str(args.alpha)),
                               list(exps)))

                else:
                    policy_exps = list(
                        filter(lambda x: x[1] == 'fair' and ((x[2] == 'dynamic') or x[2] == str(args.alpha)),
                               list(exps)))
                    logging.debug(policy, policy_exps)
                for exp in policy_exps:
                    stat_dynamic = stats[exp]
                    if args.param_is_disagreement:
                        ud = [0, float(param)]
                    else:
                        ud = [0] * int(param)

                    us = np.array(stat_dynamic['fractional_gain'])
                    N = us.shape[1]
                    logging.debug(policy, [np.mean(us[:, i]) for i in range(N)], param, exp)
                    u = (np.sum(
                        [f_alpha(np.mean(us[:, i] - ud[i]), alpha=0) for i in range(N)]))
                    name = display_name[exp[1]] + param
                    if not 'OPT' in policy:
                        pt = [display_name[policy], param] + [np.mean(us[:, i]) for i in range(N)] + [0] * (4 - N)
                        data.append(pt)
                    elif policy == 'OPT0':
                        us_opt = stat_static['opt-0.0-1.0']
                        uopt = (np.sum([f_alpha(us_opt[i] - ud[i], alpha=0) for i in range(N)]))
                        pt = [fr'OPT ($\alpha=0$)', param] + [stat_static[f'opt-{0.0}-1.0'][i] for i in range(N)] + [
                            0] * (4 - N)
                        data.append(pt)
                    else:
                        alpha = stat_dynamic['args'].alpha
                        us_opt = stat_dynamic[f'opt-{alpha}-1.0']
                        pt = [fr'OPT ($\alpha={int(args.alpha)}$)', param] + [stat_dynamic[f'opt-{alpha}-1.0'][i] for i in
                                                                              range(N)] + [0] * (4 - N)
                        data.append(pt)

        df = pd.DataFrame(data, columns=['policy', 'param'] + [f'Agent {i + 1}' for i in range(4)])
        ddata = []
        for param in args.params:
            for policy in (args.policies + [fr'OPT ($\alpha={int(args.alpha)}$)'] + [fr'OPT ($\alpha=0$)']):
                if not 'OPT' in policy:
                    policy = display_name[policy]
                ddata.append(df[np.logical_and(df.policy == policy, df.param == str(param))].values[0])
        df = pd.DataFrame(ddata, columns=['policy', 'param'] + [f'Agent {i + 1}' for i in range(4)])
        df['Setting'] = df.policy

        df = df.drop(columns=['policy', 'param'])

        ax = df.set_index(['Setting']).plot.bar(stacked=True, legend=False, edgecolor='k')
        fig = ax.get_figure()

        bbox_props = dict(boxstyle="round", fc="w", ec="0.1", alpha=0.9)
        plt.text(1, .25, "2 agents", ha="center", va="center", size=7.5, bbox=bbox_props)
        plt.text(4, .25, "3 agents", ha="center", va="center", size=7.5, bbox=bbox_props)
        plt.text(7, .25, "4 agents", ha="center", va="center", size=7.5, bbox=bbox_props)

        logging.debug(ax.get_xlim(), ax.get_ylim())
        plt.xlabel(r'Policy')
        plt.ylabel(r'Utility')
        if args.legend:
            legend = plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.2, 0, 0), frameon=True,
                                prop={'size': 6})
            legend.get_frame().set_edgecolor('k')
            legend.get_frame().set_linewidth(0.5)

    elif args.task == 'barplot-multiplayer-pof':
        fig = plt.figure(figsize=(3, 2))
        scale = [1, 1]
        data = []
        opt_recorded = False
        alphas = [1.0, 2.0, 3.0]

        for policy in (['OPT1'] + ['OPT0'] + args.policies):
            for param in args.params:
                for alpha in alphas:
                    stats = load_data(args.input_dir.replace('x', param))
                    exps = list(stats)
                    stat_static = stats[('exp', 'static')]
                    if not 'OPT' in policy:
                        policy_exps = list(
                            filter(lambda x: x[1] == policy and ((x[2] == 'dynamic') or x[2] == str(alpha)),
                                   list(exps)))

                    else:
                        policy_exps = list(
                            filter(lambda x: x[1] == 'fair' and ((x[2] == 'dynamic') or x[2] == str(alpha)),
                                   list(exps)))
                    for exp in policy_exps:
                        stat_dynamic = stats[exp]
                        us = np.array(stat_dynamic['fractional_gain'])
                        N = us.shape[1]
                        name = display_name[exp[1]] + param
                        if not 'OPT' in policy:
                            pt = [display_name[policy], param, alpha] + [np.sum([np.mean(us[:, i]) for i in range(N)])]
                            data.append(pt)
                        elif policy == 'OPT0':
                            us_opt = stat_static['opt-0.0-1.0']
                            pt = [fr'OPT$({{\alpha=0}}$)', param, 0.0] + [
                                np.sum([stat_static[f'opt-{0.0}-1.0'][i] for i in
                                        range(N)])]
                            data.append(pt)
                        else:
                            alpha = stat_dynamic['args'].alpha
                            us_opt = stat_dynamic[f'opt-{alpha}-1.0']
                            pt = [fr'OPT$({{\alpha={alpha}}}$)', param, alpha] + [
                                np.sum([stat_dynamic[f'opt-{alpha}-1.0'][i] for i in
                                        range(N)])]
                            data.append(pt)
        df = pd.DataFrame(data, columns=['policy', 'param', 'alpha', 'u'])

        for param in args.params:
            param = str(param)
            opt = df[np.logical_and(df.policy == r'OPT$({\alpha=0}$)', df.param == param)].iloc[0, 3]
            df.loc[df.param == param, 'u'] = (opt - df[df.param == param]['u']) / opt

        _df = df[df.policy == r'OPT$({\alpha=0}$)']
        d = -.15
        for alpha in alphas:
            mask = np.logical_and(df.policy == rf"OPT$({{\alpha={alpha}}}$)", df.alpha == alpha)
            plt.bar((df[mask].param).astype(int) + d, df[mask].u, label=fr'$\alpha = {alpha}$', width=.15, edgecolor='k')
            d += .15
            logging.debug(df[mask].u)
        plt.xticks([2, 3, 4])
        plt.xlabel(r'Agents')
        plt.ylabel(r'Price of Fairness')
        legend = plt.legend(ncol=3, frameon=True, prop={'size': 6})
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_linewidth(0.5)


    elif args.task == 'barplot-multiplayer-temp':
        fig = plt.figure(figsize=(3, 2.5))
        data = {}
        for policy in (['OPT1'] + ['OPT0'] + args.policies):
            for param in args.params:
                stats = load_data(args.input_dir.replace('x', param))
                exps = list(stats)
                stat_static = stats[('exp', 'static')]
                if not 'OPT' in policy:
                    policy_exps = list(
                        filter(lambda x: x[1] == policy and ((x[2] == 'dynamic') or x[2] == str(args.alpha)),
                               list(exps)))

                else:
                    policy_exps = list(
                        filter(lambda x: x[1] == 'fair' and ((x[2] == 'dynamic') or x[2] == str(args.alpha)),
                               list(exps)))
                for exp in policy_exps:
                    stat_dynamic = stats[exp]
                    us = np.array(stat_dynamic['fractional_gain'])
                    N = us.shape[1]
                    u = (np.sum(
                        [f_alpha(np.mean(us[:, i]), alpha=0) for i in range(N)]))
                    name = display_name[exp[1]] + param
                    if not 'OPT' in policy:
                        pt = [display_name[policy], param] + [np.mean(us[:, i]) for i in range(N)] + [0] * (4 - N)
                        data[(policy, param)] = us
                    elif policy == 'OPT0':
                        us_opt = stat_static['opt-0.0-1.0']
                        data[('OPT0', param)] = us_opt
                    else:
                        alpha = stat_dynamic['args'].alpha
                        us_opt = stat_dynamic[f'opt-{alpha}-1.0']
                        data[('OPT1', param)] = us_opt
        for policy in args.policies:
            us = data[(policy, str(args.player))]
            D = 10
            x = np.arange(tavg(us[:, 0], D).shape[0]) * D
            markers = ['o', 'P', 's', 'X']
            for i in range(us.shape[1]):
                plt.plot(x, tavg(us[:, i], D), label=f'Agent {i + 1}', color=f'C{i}', marker=markers[i], markevery=50)
            gaplabel = 'Utility gap'
            for i in range(us.shape[1]):
                name = rf'OPT ($\alpha=0$)' if i == 0 else '_nolegend_'
                plt.plot(x, [data[('OPT0', str(args.player))][i]] * len(x), color=f'C{i}', linestyle='--', label=name,
                         linewidth=0.5)
                name = rf'OPT ($\alpha={alpha}$)' if i == 0 else '_nolegend_'
                plt.plot(x, [data[('OPT1', str(args.player))][i]] * len(x), color=f'C{i}', linestyle='-', label=name,
                         linewidth=0.5)
                d0 = data[('OPT0', str(args.player))][i]
                d1 = data[('OPT1', str(args.player))][i]
                g = .005 if max([d0, d1]) - min([d0, d1]) >= .005 else 0
                plt.arrow(4500, min([d0, d1]), 0, max([d0, d1]) - min([d0, d1]) - g
                          , head_width=100, head_length=g, linewidth=.5, color='k', label=gaplabel, fc='k', ec='k', zorder=3)
                gaplabel = '_nolegend_'
                plt.arrow(4500, max([d0, d1]), 0, min([d0, d1]) - max([d0, d1]) + g
                          , head_width=100, head_length=g, linewidth=.5, color='k', label=gaplabel, fc='k', ec='k', zorder=3)

            plt.xlabel(r'Requests')
            plt.ylabel(r'Utility')
            if args.legend:
                legend = plt.legend(ncol=7, loc='upper center', bbox_to_anchor=(.5, 1.5, 0, 0), frameon=True)
                legend.get_frame().set_edgecolor('k')
                legend.get_frame().set_linewidth(0.5)
    elif args.task == 'topology_draw':
        for dir in [2, 3, 4]:
            fig = plt.figure(figsize=(2.5, 2.5))
            stats = load_data(f'./res/2players-topology-tree-multiplayer-{dir}/')
            stats.keys()
            stat_static = stats[('exp', 'static')]
            I = (stat_static['players'])
            graph = stat_static['graph']
            query_nodes = stat_static['query_nodes']
            extraq = query_nodes[0]
            repo_nodes = stat_static['repo_nodes']
            pos = nx.nx_pydot.graphviz_layout(graph)
            logging.debug(pos)
            colors = ['#D5E8D4', '#FFE6CC', '#DAE8FC', '#E1D5E7', '#F5F5F5']
            linecolors = ['#82B366', '#D79B00', '#6C8EBF', '#9673A6', '#666666']
            for p in range(I):
                # Query Nodes
                nodes = list(filter(lambda n: int(graph.nodes()[n]['owner']) == p and n not in query_nodes, list(graph)))
                if p == 0:
                    logging.debug([int(n) for n in nodes] + [int(extraq)])
                    nodes = [int(n) for n in nodes] + [int(extraq)]
                p_query_nodes = list(filter(lambda n: int(graph.nodes()[n]['owner']) == p and n in query_nodes, list(graph)))

                subgraph = nx.subgraph(graph, nodes)

                nx.draw_networkx_nodes(subgraph, pos=pos, node_size=100, node_color=colors[p], edgecolors=linecolors[p],
                                       label=f'Agent ${p + 1}$ node', )

                subgraph = nx.subgraph(graph, p_query_nodes)
                nx.draw_networkx_nodes(subgraph, pos=pos, node_shape='s', node_size=100, node_color=colors[p],
                                       edgecolors=linecolors[p],
                                       label=f'Agent ${p + 1}$ query node')
            subgraph = nx.subgraph(graph, repo_nodes)
            nx.draw_networkx_nodes(subgraph, pos=pos, node_size=100, node_color='C0',
                                   edgecolors='C0',
                                   label='Repository node')
            labels = nx.get_edge_attributes(graph, 'weight')
            for key in labels:
                labels[key] = int(labels[key]) / 2
            nx.draw_networkx_edges(graph, pos=pos, width=list(labels.values()), edge_color='#6C8EBF')
            for i, key in enumerate(graph.edges):
                labels[key] = int(stat_static['args'].custom_weights[i] * 2)
            nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels, clip_on=True)
            ax = fig.get_axes()[0]
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            fig.get_axes()[0].set_rasterized(True)
            plt.savefig(f'./out_figs/2players-topology-tree-multiplayer-{dir}.pdf')

        legend = plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.4, 0, 0), frameon=True)
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_linewidth(0.5)
        plt.savefig(f'./out_figs/2players-topology-tree-multiplayer-legend.pdf')

        for dir in ['tree', '2d_grid', 'abilene', 'geant']:  #
            fig = plt.figure(figsize=(3, 3))
            stats = load_data(f'./res/2players-topology-{dir}/')
            stats.keys()
            stat_static = stats[('exp', 'static')]
            I = (stat_static['players'])
            graph = stat_static['graph']
            query_nodes = stat_static['query_nodes']
            repo_nodes = stat_static['repo_nodes']
            pos = nx.nx_pydot.graphviz_layout(graph)
            logging.debug(len(graph), len(graph.edges))
            colors = ['#D5E8D4', '#FFE6CC', '#DAE8FC']
            linecolors = ['#82B366', '#D79B00', '#6C8EBF']

            for p in range(I):
                # Query Nodes
                nodes = list(filter(lambda n: int(graph.nodes()[n]['owner']) == p and n not in query_nodes, list(graph)))
                p_query_nodes = list(filter(lambda n: int(graph.nodes()[n]['owner']) == p and n in query_nodes, list(graph)))
                subgraph = nx.subgraph(graph, p_query_nodes)
                nx.draw_networkx_nodes(subgraph, pos=pos, node_shape='s', node_size=100, node_color=colors[p],
                                       edgecolors=linecolors[p],
                                       label=f'Player ${p + 1}$ query node')
                subgraph = nx.subgraph(graph, nodes)
                nx.draw_networkx_nodes(subgraph, pos=pos, node_size=100, node_color=colors[p], edgecolors=linecolors[p],
                                       label=f'Player ${p + 1}$ node', )
            subgraph = nx.subgraph(graph, repo_nodes)
            nx.draw_networkx_nodes(subgraph, pos=pos, node_size=100, node_color='C0',
                                   edgecolors='C0',
                                   label='Repository node')
            labels = nx.get_edge_attributes(graph, 'weight')

            for key in labels:
                labels[key] = int(labels[key]) / 2
            nx.draw_networkx_edges(graph, pos=pos, width=list(labels.values()), edge_color='#6C8EBF')
            for key in labels:
                labels[key] = int(labels[key] * 2)
            nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels)
            ax = fig.get_axes()[0]
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

            fig.get_axes()[0].set_rasterized(True)
            plt.savefig(f'./out_figs/topology-{dir}.pdf')

        legend = plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.4, 0, 0), frameon=True)
        legend.get_frame().set_edgecolor('k')
        legend.get_frame().set_linewidth(0.5)
        plt.savefig(f'./out_figs/topology-legend.pdf')

    fig.get_axes()[0].set_rasterized(True)
    if args.task != 'barplot' and args.task != 'barplot-multiplayer-temp' and args.task != 'barplot-multiplayer':
        plt.tight_layout()
    if args.task != 'topology_draw':
        plt.savefig(args.output, dpi=200)
        plt.show()

    logging.debug('finished')
