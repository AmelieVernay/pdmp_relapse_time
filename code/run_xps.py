import click
import json
import subprocess
import time
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from simulate import simulate_cohort
from fit_parameters import cohort_analysis, get_survival_times
from evaluate_model import get_errors, get_statistics, get_confusion_matrix, aggregate_statistics, aggregate_confusion_matrices
from utils import NumpyEncoder


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
@click.option('--variable', default='visit_every', help='Nuisance parameters that varies for the experimentation. Should be one of "visit_every" or "sigma" or "n_samples". Used in groupby when computing results.')
@click.option('--n-reps', type=int, default=100, help='Number of batch to simulate and perform analysis on.')
@click.option('--save-intermediate', '-s', default=False, is_flag=True, help='If True, save fitted parameters before survival analysis.')
@click.option('--results', default='../results/', help='Path name to result folder.')
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
    n_reps = kwargs['n_reps']
    save = kwargs['save_intermediate']
    file_path = kwargs['results']
    
    variable_param = kwargs['variable']
    
    sigmas = [float(i) for i in kwargs['sigmas']]
    visit_freqs = [int(i) for i in kwargs['visit_freqs']]
    n_samples = [int(i) for i in kwargs['n_samples']]
    
    errors_stats = []
    conf_matrices = []
    times = []
    for rep in range(n_reps):
        print(f'Run {rep + 1}/{n_reps}', end='\n')
        # get data
        if data_path is None:
            data = simulate_cohort(zeta0, v_remission, v_relapse, weibull_shape, weibull_scale, visit_freqs, sigmas, n_samples)
        else:
            with open(data_path, 'r') as f:
                data = json.load(f)
        # fit parameters
        start = time.perf_counter()
        fitted_parameters = cohort_analysis(data)
        if save:
            with open('fitted_parameters.json', 'w') as f:
                json.dump(fitted_parameters, f, indent=4, cls=NumpyEncoder)
        get_survival_times(fitted_parameters, 'fitted_survival_times.csv', variable=variable_param)
        # perform survival analysis
        subprocess.call(['Rscript', 'survival_regression.R'])
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        # compute errors
        errors = get_errors('fitted_survival_times.csv', 'survival_analysis.csv', batch_num=rep + 1, variable=variable_param)
        errors_stats.append(get_statistics(errors, variable=variable_param))
        # compute confusion matrix
        conf_matrices.append(get_confusion_matrix(errors, variable=variable_param))
        
    full_results = aggregate_statistics(errors_stats)
    confusion_matrices = aggregate_confusion_matrices(conf_matrices, variable=variable_param)
    
    Path(f'{file_path}').mkdir(parents=True, exist_ok=True)
    full_results.to_csv(f'{file_path}/xps_statistics.csv', index=False)
    confusion_matrices.to_csv(f'{file_path}/confusion_matrices.csv', index=False)
    errors.to_csv(f'{file_path}/estimation_errors_last_batch.csv', index=False)
    pd.DataFrame({'running_time': times}).to_csv(f'{file_path}/running_times.csv', index=False)
    
    print(f'Job done. Results saved in {file_path}')


if __name__ == "__main__":
    run_analysis()

