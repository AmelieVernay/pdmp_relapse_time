import json

import numpy as np
import ruptures as rpt
from sklearn.metrics.cluster import rand_score, adjusted_rand_score

from utils import shifted_differences, only_relapse, bkps_to_states



def cohort_change_points(data):
    results = {}
    for i, k in enumerate(data.keys()):
        print(f'Fitting change points {i + 1}/{len(data.keys())}', end='\r')
        signal = shifted_differences(data[k]['spike'])
        algo = rpt.Dynp(model='l2').fit(signal)
        bkps = algo.predict(n_bkps=2)
        states = bkps_to_states(bkps, len(signal))
        states_true = data[k]['states_true']
        rs = rand_score(states_true, states)
        ars = adjusted_rand_score(states_true, states)
        results[k] = {'subject': k, 'rs_rpt': rs, 'ars_rpt': ars}
    print()
    return results

