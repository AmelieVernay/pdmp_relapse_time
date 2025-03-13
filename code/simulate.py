import itertools

import numpy as np
import pandas as pd


class Subject:
    def __init__(
            self,
            zeta0,
            v_remission,
            v_relapse,
            weibull_shape,
            weibull_scale,
            visit_every,
            sigma,
            start=None,
            follow_up=None,
    ):
        self.start = np.random.uniform(15, 55, 1) if start is None else start
        self.follow_up = np.random.uniform(900, 1900, 1)[0] if follow_up is None else follow_up
        self.zeta0 = zeta0
        self.v_remission = v_remission
        self.v_relapse = v_relapse
        self.weibull_shape = weibull_shape
        self.weibull_scale = weibull_scale
        self.visit_every = visit_every
        self.sigma = sigma  # noise level

    def compute_parameters(self):
        self.T1 = ((np.log(self.zeta0) - np.log(self.start)) / self.v_remission)[0]
        self.T2 = (np.random.weibull(a=self.weibull_shape, size=1) * self.weibull_scale + self.T1)[0]
        self.censored = self.last_visit < self.T2
        self.relapse = None if self.censored else self.T2
        
    def set_true_traj(self):
        self.x_obs = np.arange(0, self.follow_up, self.visit_every)
        self.last_visit = max(self.x_obs)
        self.compute_parameters()

        if self.censored:
            self.y_true = np.piecewise(
                self.x_obs,
                [
                    self.x_obs <= self.T1,
                    self.x_obs > self.T1
                ],
                [
                    lambda x: self.start * np.exp(self.v_remission * x),
                    self.zeta0
                ]
            )
        else:
            self.y_true = np.piecewise(
                self.x_obs,
                [
                    self.x_obs <= self.T1,
                    (self.x_obs > self.T1) & (self.x_obs <= self.T2),
                    self.x_obs > self.T2],
                [
                    lambda x: self.start * np.exp(self.v_remission * x),
                    self.zeta0,
                    lambda x: self.zeta0 * np.exp(self.v_relapse * (x - self.T2))
                ]
            )
            
    
    def truncate(self, threshold=60):
        '''Truncate trajectory after T2 if more than two values above threshold.

        Unnecessary steps in here because exp is strictly increasing but kept
        as is if ever I need to apply truncate() on y_obs rather than on y_true.
        '''
        above_thresh = np.array(np.where(self.y_true > threshold)[0])
        above_thresh_after_T2 = np.where(above_thresh > int(int(self.T2.item()) / self.visit_every))[0]
        if above_thresh_after_T2.shape[0] > 2:
            self.y_true = self.y_true[0:above_thresh[above_thresh_after_T2[1]]]
            self.x_obs = self.x_obs[0:above_thresh[above_thresh_after_T2[1]]]
            self.last_visit = max(self.x_obs)

    
    def simulate(self):
        self.set_true_traj()
        self.truncate()  # assess huge values at traj end
        self.y_obs = np.maximum(0, self.y_true + np.random.normal(0, self.sigma, len(self.x_obs)))


def simulate_cohort(zeta0, v_remission, v_relapse, weibull_shape, weibull_scale, visit_freqs, sigmas, n_samples):  # XXX visit_freqs & sigmas should be a list
    data = {}
    variable_values = [x for x in itertools.product(visit_freqs, sigmas, n_samples)]
    
    for v in variable_values:
        for i in range(v[2]):
            s = Subject(
                zeta0=zeta0,
                v_remission=v_remission,
                v_relapse=v_relapse,
                weibull_shape=weibull_shape,
                weibull_scale=weibull_scale,
                visit_every=v[0],
                sigma=v[1],
            )
            s.simulate()
            
            subject_id = f'{int(i)}' + f'{int(v[0])}' + f'{int(v[1])}' + f'{int(v[2])}'
            data[subject_id] = {
                'subject': subject_id,
                'cum_day_diff': s.x_obs,
                'y_true': s.y_true,
                'spike': s.y_obs,
                'v_remission_true': s.v_remission,
                'v_relapse_true': s.v_relapse,
                'T1_true': s.T1,             # time of remission start
                'T2_true': s.relapse,        # time of relapse start if any, None otherwise
                'saved_T2': s.T2,            # simulated T2 (becomes None in parameter `T2_true` if falls after last_visit. Not used but kept is just in case)
                'follow_up': s.follow_up,    # simulated horizon
                'last_visit': s.last_visit,  # actual horizon
                'weibull_shape_true': s.weibull_shape,
                'weibull_scale_true': s.weibull_scale,
                'remission_thresh': s.zeta0,
                'censored_true': s.censored, # True if last_visit < T2_true
                'relapse_true': s.relapse is not None,
                'visit_every': s.visit_every,
                'sigma': s.sigma,
                'n_samples': v[2],           # not subject related but good info to track ntl
            }
    return data

    

