import numpy as np

import tools
from policies.AbstractCacheNetwork import AbstractCacheNetwork


class OnlineFairCacheNetwork(AbstractCacheNetwork):
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
        self.umin, self.umax = self.umin_umax
        self.dv =np.ones(self.players)#np.array([(self.umin**(-self.alpha) +self.umax**(-self.alpha))*0.5 for i in range(self.players)])
    def global_euclidean_project(self):
        for n in range(self.graph_size):
            if not tools._is_feasible(self.fractional_caches[n][self.mask_caches[n]], self.capacities[n]):
                self.fractional_caches[n][self.mask_caches[n]] = self.euclidean_projections[n].project(
                    self.fractional_caches[n][self.mask_caches[n]])

    def adapt_state(self):
        fractional_gains = self.stats_dynamic['fractional_gain'][-1]
        for owner in range(self.players):
            for n in range(self.graph_size):
                grad_norm = np.linalg.norm(self.subgradients[owner][n] * self.dv[owner],
                                           2) ** 2
                self.gradient_normsX += grad_norm
                self.gradient_normsXVec[n] += grad_norm
        etaX = self.diamX / np.sqrt(self.gradient_normsX)
        for n in range(self.graph_size):

            self.fractional_caches[n] = self.fractional_caches[n] + etaX * \
                                        np.sum([self.dv[owner] * self.subgradients[owner][n] for owner in
                                                range(self.players)], axis=0)
        if self.alpha != 0:
            self.dv_grad = np.array(
                [- 1.0 / (self.dv[p]) ** (1/self.alpha) + fractional_gains[p] - self.external_disagreement_points[p] for p in
                 range(self.players)])
            self.dv = self.dv - self.alpha / (self.umin ** (1 + 1 / self.alpha) * self.t) * self.dv_grad
            self.dv[self.dv > 1. / self.umin ** (self.alpha)] = 1. / self.umin ** (self.alpha)
            self.dv[self.dv < 1. / self.umax ** (self.alpha)] = 1. / self.umax ** (self.alpha)
        self.global_euclidean_project()
        for owner in range(self.players):
            self.subgradients[owner] *= 0
        self.stats_dynamic['etas'].append(list(self.etas))
        self.stats_dynamic['dvs'].append(list(self.dv))
