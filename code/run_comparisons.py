import click
import json
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.cluster import rand_score, adjusted_rand_score

from simulate import simulate_cohort
from fit_parameters import cohort_analysis
from comparison_hmm import cohort_hmm
from comparison_change_points import cohort_change_points
from utils import NumpyEncoder, only_relapse, bkps_to_states


@click.command()
@click.option('--data', '-d', default=None, help='Path to data (.json) on which to perform the analysis. If None, data will be simulated. If not None, --n-reps should be 1.')
@click.option('--zeta-zero', type=float, default=1., help='Remission threshold.')
@click.option('--v-remission', type=float, help='Slope in remission mode.')
@click.option('--v-relapse', type=float, help='Slope in relapse mode.')
@click.option('--w-shape', type=float, help='Shape for the Weibull distribution of survival time.')
@click.option('--w-scale', type=float, help='Scale for the Weibull distribution of survival time.')
@click.option('--visit-freqs', multiple=True, help='Range of visit frequencies.')
@click.option('--sigmas', multiple=True, help='Range of standard deviations for the random gaussian noise added to each trajectory. Must be non-negatives.')
@click.option('--n-samples', multiple=True, help='Range of number of samples to simulate and perform analysis on.')
@click.option('--variable', default='visit_every', help='Nuisance parameters that varies for the experimentation. Should be one of "visit_every" or "sigma" or "n_samples". Used in groupby when computing results.')  # XXX: not used here
@click.option('--save-intermediate', '-s', default=True, is_flag=True, help='If True, save fitted parameters before survival analysis.')  # XXX change default to False when stable?
@click.option('--results', default='./results/', help='Path name to result folder.')  # XXX
@click.option('--config-file', default=None, help='YAML configuration file containing running options.')
def run_analysis(**kwargs):
    config_file = kwargs['config_file']
    if config_file is not None:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            config['config_file'] = None
            # overwrite default kwargs with the ones specified in config file
            kwargs = kwargs | config

    data_path = kwargs['data']
    zeta0 = kwargs['zeta_zero']
    v_remission = kwargs['v_remission']
    v_relapse = kwargs['v_relapse']
    weibull_shape = kwargs['w_shape']
    weibull_scale = kwargs['w_scale']
    save = kwargs['save_intermediate']
    file_path = kwargs['results']
    
    variable_param = kwargs['variable']
    
    sigmas = [float(i) for i in kwargs['sigmas']]
    visit_freqs = [int(i) for i in kwargs['visit_freqs']]
    n_samples = [int(i) for i in kwargs['n_samples']]
    
    # simulate or grab data
    if data_path is None:
        data = simulate_cohort(zeta0, v_remission, v_relapse, weibull_shape, weibull_scale, visit_freqs, sigmas, n_samples)
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
    # fit parameters
    fitted_parameters = cohort_analysis(data)
    # only perform comparisons on trajectories with relapse
    data = only_relapse(fitted_parameters)
    print(f'Performing comparisons on {len(data)} trajectories with relapse.')

    res_pdmp = {}
    for k in data.keys():
        bkps_true = [
            np.abs(np.array(data[k]['cum_day_diff']) - data[k]['T1_true']).argmin(),
            np.abs(np.array(data[k]['cum_day_diff']) - data[k]['T2_true']).argmin(),
            len(data[k]['cum_day_diff'])
        ]
        bkps_pdmp = [
            np.abs(np.array(data[k]['cum_day_diff']) - data[k]['T1_hat']).argmin(),
            np.abs(np.array(data[k]['cum_day_diff']) - data[k]['T2_hat']).argmin(),
            len(data[k]['cum_day_diff'])
        ]
        # get sequences of states in between breakpoints
        states_true = bkps_to_states(bkps_true, len(data[k]['cum_day_diff'][1:]))
        states_pdmp = bkps_to_states(bkps_pdmp, len(data[k]['cum_day_diff'][1:]))
        
        rs = rand_score(states_true, states_pdmp)
        ars = adjusted_rand_score(states_true, states_pdmp)
        
        res_pdmp[k] = {'subject': k, 'rs_pdmp': rs, 'ars_pdmp': ars}
        data[k].update({'states_true': states_true})

    # perform analysis on same cohort with HMM and change points
    res_hmm = cohort_hmm(data)
    res_rpt = cohort_change_points(data)

    # merge and save results
    full_results = {k: res_pdmp[k] | res_hmm[k] | res_rpt[k] for k in res_pdmp.keys()}
    full_results = pd.DataFrame.from_dict(full_results, 'index').reset_index()
    Path(f'{file_path}').mkdir(parents=True, exist_ok=True)
    full_results.to_csv(f'{file_path}/comparisons.csv', index=False)


if __name__ == "__main__":
    run_analysis()
