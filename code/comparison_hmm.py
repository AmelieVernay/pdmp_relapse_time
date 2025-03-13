import numpy as np
import pandas as pd
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

from hmmlearn import hmm
from utils import shifted_differences, only_relapse, suppress_stdout_stderr


class ConstrainedGaussianHMM(hmm.GaussianHMM):
    def _do_mstep(self, stats):
        
        # do the standard HMM learning step
        super()._do_mstep(stats)
                
        # sort states are where
        self.means_[np.isnan(self.means_)] = 0
        m = np.squeeze(self.means_)
        s = np.argsort(m)
        s1, s2, s3 = 0, 2, 1

        # manipulate the transition matrix as you see fit
        self.transmat_[s1, s3] = 0.0
        self.transmat_[s2, s1] = 0.0
        self.transmat_[s3, s1] = 0.0
        self.transmat_[s3, s2] = 0.0
        self.transmat_[s3, s3] = 1.0


def constrained_hmm(signal, signal_length=None, n_components=3):
    if signal_length is None: signal_length = signal.shape[0]
    
    models = list()
    scores = list()
    
    for idx in range(10):
        # ten different random starting states
        model = ConstrainedGaussianHMM(n_components=n_components, random_state=idx, n_iter=10, covariance_type='full', params='tmc', init_params='mc')
            
        model.transmat_ = np.array([     #      -1   0    1
            [0.3, 0.0, 0.7],             # -1 [0.3, 0.7, 0.0],
            [0.0, 1.0, 0.0],             #  0 [0.0, 0.4, 0.6],
            [0.0, 0.6, 0.4]              #  1 [0.0, 0.0, 1.0],
        ])
        model.startprob_ = np.array([1., 0., 0.])
        
        with suppress_stdout_stderr():
            model.fit(np.array(signal)[:, None], signal_length)
        models.append(model)
        scores.append(model.score(np.array(signal)[:, None]))

    # get the best model
    model = models[np.argmax(scores)]
    # use the Viterbi algorithm to predict the most likely sequence of states
    states = model.predict(np.array(signal)[:, None])

    return model, states


def order_states(states):
    result = states.copy()
    result[result == 2] = 3
    result[result == 1] = 2
    result[result == 3] = 1
    return result


def cohort_hmm(data):
    results = {}
    for i, k in enumerate(data.keys()):
        print(f'Fitting HMM {i + 1}/{len(data.keys())}', end='\r')
        signal = shifted_differences(data[k]['spike'])
        try:
            _, states = constrained_hmm(signal)
            states = order_states(states)
            states_true = data[k]['states_true']
            rs = rand_score(states_true, states)
            ars = adjusted_rand_score(states_true, states)
            converged = True
        except:
            rs = ars = None
            converged = False
        results[k] = {'subject': k, 'rs_hmm': rs, 'ars_hmm': ars, 'hmm_converged': converged}
    print()
    return results

