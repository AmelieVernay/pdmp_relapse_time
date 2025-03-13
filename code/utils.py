import json
import numpy as np


from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull


# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# improved a little and used to store numpy arrays as JSON and restore them as numpy later on
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# suppress stderr convergence monitor messages from hmmlearn.hmm.GaussianHMM.fit()
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def shifted_differences(signal, to_log=True):
    signal = np.array(signal)
    if to_log: signal = np.log(signal + 0.01)
    return (signal - np.roll(signal, 1))[1:]


def only_relapse(data):
    return {k: data[k] for k in data.keys() if data[k]['T2_true'] is not None and data[k]['T2_hat'] is not None}


def bkps_to_states(bkps, signal_length):
    states = np.zeros(signal_length)
    states[0:bkps[0]] = 0
    states[bkps[0]:bkps[1]] = 1
    states[bkps[1]:bkps[2]] = 2
    return states.astype(int)

