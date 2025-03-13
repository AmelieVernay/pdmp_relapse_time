import click
import json

from simulate import simulate_cohort
from utils import NumpyEncoder


@click.command()
@click.option('--zeta-zero', type=float, default=1., help='Remission threshold.')
@click.option('--v-remission', type=float, help='Slope in remission mode.')
@click.option('--v-relapse', type=float, help='Slope in relapse mode.')
@click.option('--w-shape', type=float, help='Shape for the Weibull distribution of survival time.')
@click.option('--w-scale', type=float, help='Scale for the Weibull distribution of survival time.')
@click.option('--visit-freqs', multiple=True, help='Range of visit frequencies.')
@click.option('--sigmas', multiple=True, help='Range of standard deviations for the random gaussian noise added to each trajectory. Must be non-negatives.')
@click.option('--n-samples', '-n', multiple=True, help='Range of number of subjects to simulate.')
@click.option('--path', default='simulations.json', help='Path for the results.')  # XXX seperate path and file name
@click.option('--config-file', default=None, help='YAML configuration file containing running options.')
def run_simulation(**kwargs):
    config_file = kwargs['config_file']
    if config_file is not None:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            config['config_file'] = None
            # overwrite default kwargs with ones given in config
            kwargs = kwargs | config

    # XXX maybe don't need to put args in variables (edit: but yes if config file?) and make it more concise
    path = kwargs['path']
    zeta0 = kwargs['zeta_zero']
    v_remission = kwargs['v_remission']
    v_relapse = kwargs['v_relapse']
    weibull_shape = kwargs['w_shape']
    weibull_scale = kwargs['w_scale']

    visit_freqs = [int(i) for i in kwargs['visit_freqs']]
    sigmas = [float(i) for i in kwargs['sigmas']]
    n_samples = [int(i) for i in kwargs['n_samples']]
    
    # run simulations
    data = simulate_cohort(zeta0, v_remission, v_relapse, weibull_shape, weibull_scale, visit_freqs, sigmas, n_samples)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
        
        
if __name__ == '__main__':
    run_simulation()
