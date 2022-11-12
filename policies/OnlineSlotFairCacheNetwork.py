import numpy as np

import tools
from policies.AbstractCacheNetwork import AbstractCacheNetwork


class OnlineSlotFairCacheNetwork(AbstractCacheNetwork):
    def __init__(self, network_properties):
        super().__init__(network_properties)
        self.gradient_normsX = 0
        self.gradient_normsXVec = np.zeros(self.graph_size)

        self.gradient_normsDV = 0
        self.gradient_normsDVVec = np.zeros(self.players)
        self.etas = np.zeros(self.graph_size)
        self.euclidean_projections = [
            tools.EuclideanProjection(self.catalog_size - np.sum(~self.mask_caches[n]), self.capacities[n]) for n in
            np.arange(self.graph_size)]
        self.average_fractional_gains = np.zeros(self.players)

    def global_euclidean_project(self):
        for n in range(self.graph_size):
            if not tools._is_feasible(self.fractional_caches[n][self.mask_caches[n]], self.capacities[n]):
                self.fractional_caches[n][self.mask_caches[n]] = self.euclidean_projections[n].project(
                    self.fractional_caches[n][self.mask_caches[n]])

    def adapt_state(self):
        W = self.fairslotted_freeze_period
        if W == 1:
            self.average_fractional_gains = np.array(self.stats_dynamic['fractional_gain'][-1]) - np.array(
                self.external_disagreement_points)
        if self.t % W == 0 and self.t != 1:
            for owner in range(self.players):
                for n in range(self.graph_size):
                    w = (self.average_fractional_gains[owner]) ** self.alpha if self.alpha != 0 else 1
                    grad_norm = np.linalg.norm(self.subgradients[owner][n] / w,
                                               2) ** 2
                    self.gradient_normsX += grad_norm
                    self.gradient_normsXVec[n] += grad_norm
            etaX = self.diamX / np.sqrt(self.gradient_normsX)
            for n in range(self.graph_size):
                self.fractional_caches[n] = self.fractional_caches[n] + etaX * \
                                            np.sum([self.subgradients[owner][n] / (
                                                    self.average_fractional_gains[owner] ** self.alpha) for
                                                    owner in
                                                    range(self.players)], axis=0)
            self.global_euclidean_project()

            self.average_fractional_gains *= 0
            for owner in range(self.players):
                self.subgradients[owner] *= 0
        else:
            fractional_gains = self.stats_dynamic['fractional_gain'][-1]
            self.average_fractional_gains += np.array(fractional_gains) - np.array(self.external_disagreement_points)

        self.stats_dynamic['etas'].append(list(self.etas))
