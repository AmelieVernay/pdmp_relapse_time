import json

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.optimize import curve_fit


class Trajectory:
    def __init__(self, data, x_key, y_key, remission_key):
        self.x = np.asarray(data[x_key])
        self.y = np.asarray(data[y_key])
        self.remission_threshold = data[remission_key]
        self.xmax = self.x[-1]
    
    def reverse_trajectory(self, start_idx):
        self.x = self.x[int(start_idx):]
        self.x = -(self.x - self.xmax)[::-1]
        self.y = self.y[int(start_idx):][::-1]
    
    @staticmethod
    def not_exceeding_thresh(y, thresh=5, up_to_idx=5):
        return np.all(y[:up_to_idx] < thresh)
    
    def fit_flow_curve(self, x, y, flow, **kwargs):
        try:
            popt, _ = curve_fit(flow, x, np.log(y + 1), **kwargs)
            self._a = np.exp(popt[0])
            self._b = popt[1]
        except RuntimeError as e:
            print(e)

    def intersect_remission(self):
        self._jump_time = np.log(self._a / self.remission_threshold) / self._b
        self._pts_before_jump  = np.sum(self.x <= self._jump_time)

    def compute_fit_error(self):
        self.error_before_jump = np.linalg.norm(
            self.y[0:self._pts_before_jump]
              - self._a*np.exp(-self.x[0:self._pts_before_jump]*self._b),
            ord=2
        )**2
        self.error_after_jump  = np.linalg.norm(
            self.y[self._pts_before_jump:] - self.remission_threshold,
            ord=2
        )**2
        self.fit_error = self.error_before_jump + self.error_after_jump

    @staticmethod
    def linear_flow(x, a, b):
        return a - b*x  # linearization of a' * np.exp(-x*b)

    def get_jump_time(self, start_idx=None):

        if start_idx is not None:
            self.reverse_trajectory(start_idx)
        
        if self.not_exceeding_thresh(self.y):
            return None, None, None, None, None

        delta_opt = np.inf
        idx_opt = self.a = self.b = None
        for idx in range(3, len(self.x)):
            xdata = self.x[0:idx]
            ydata = self.y[0:idx]
            
            self.fit_flow_curve(xdata, ydata, self.linear_flow)
            self.intersect_remission()
            self.compute_fit_error()
            
            # update parameters if fitting error decreases
            if (self.fit_error < delta_opt and self._jump_time > 0 and self._jump_time < self.xmax):
                self.jump_time = self._jump_time
                delta_opt = self.fit_error
                idx_opt = idx  # number of points used to fit the best curve
                self.a = self._a
                self.b = self._b
            
            # stop loop if fitting error stops decreasing anymore
            # and at least 15 points have been visited
            if (self.fit_error > delta_opt and self._jump_time > 0 and idx > 15):
                break
        
        self.pts_before_jump = np.sum(self.x <= self.jump_time)
        # get jump time relative to reversed and truncated trajectory
        if start_idx is not None:
            self.jump_time = -self.jump_time + self.xmax
            self.pts_before_jump = np.sum(self.x >= self.jump_time)
        
        if self.b is not None and self.b < 0.005:  # could have used bounds in curve_fit() but leads to bad estimates at the boundary and is very slow
            return None, None, None, None, None
        
        return self.jump_time, idx_opt, self.pts_before_jump, self.a, self.b


def cohort_analysis(data):
    subjects = data.keys()

    for i, s in enumerate(list(subjects)):
        print(f'Fitting trajectories: {(i + 1)/len(subjects):2.0%}', end='\r')
        trajectory = Trajectory(data=data[s], x_key='cum_day_diff', y_key='spike', remission_key='remission_thresh')
        try:
            T1, k_rem, j_rem, a_rem, b_rem = trajectory.get_jump_time()
        except:
            T1, k_rem, j_rem, a_rem, b_rem = None
        # sometimes the first spike is very low and T1 can not be estimated
        if T1 is None:
            data.pop(s)
            continue
        
        try:
            T2, k_rel, j_rel, a_rel, b_rel = trajectory.get_jump_time(start_idx=j_rem)
        except:
            T2, k_rel, j_rel, a_rel, b_rel = None
        
        if (T2 is not None and (T2 < 0 or T2 < T1)): T2 = b_rel = None
        
        data[s].update({
            'T1_hat': T1, 'T2_hat': T2,
            'pts_to_fit_T1': k_rem, 'pts_to_fit_T2': k_rel,
            'pts_before_T1': j_rem, 'pts_before_T2': j_rel,
            'a_T1': a_rem, 'a_T2': a_rel,  # corresponds to the 'a' in a - b*x
            'v_remission_hat': b_rem, 'v_relapse_hat': b_rel,
            # survival_time is never None and relapse_time is not None only if relapse occurs
            'relapse_time_hat':  T2 - T1 if T2 is not None else None,
            'survival_time_hat': T2 - T1 if T2 is not None else data[s]['last_visit'] - T1,
            'censored_hat': T2 is None,
        })
    
    print()
        
    return data


def get_survival_times(data, path, variable=None):
    keys_to_extract = ['subject', 'T1_true', 'T1_hat', 'T2_true', 'T2_hat', 'weibull_shape_true', 'weibull_scale_true', 'censored_true', 'censored_hat', 'relapse_time_hat', 'survival_time_hat', 'visit_every', 'sigma', 'n_samples']
    results = {subject: {key: data[subject].get(key) for key in keys_to_extract} for subject in data}
    results = pd.DataFrame.from_dict(results, orient='index').convert_dtypes(convert_boolean=False)
    results['event_hat'] = 1 - results['censored_hat']
    results['variable_parameter'] = variable
    results.to_csv(path, index=False)


def get_survival_times_application(data, path):
    keys_to_extract = ['subject', 'T1_hat', 'T2_hat', 'censored_hat', 'relapse_time_hat', 'survival_time_hat']
    results = {subject: {key: data[subject].get(key) for key in keys_to_extract} for subject in data}
    results = pd.DataFrame.from_dict(results, orient='index').convert_dtypes(convert_boolean=False)
    results['event_hat'] = 1 - results['censored_hat']
    results.to_csv(path, index=False)
