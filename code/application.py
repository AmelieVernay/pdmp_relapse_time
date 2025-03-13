import json
import subprocess

import numpy as np

from fit_parameters import cohort_analysis, get_survival_times_application
from utils import NumpyEncoder


data_path = 'ifmdfci_preprocessed.json'
with open(data_path, 'r') as f:
    data = json.load(f)
# fit parameters
fitted_parameters = cohort_analysis(data)
with open('fitted_parameters_real_data.json', 'w') as f:
    json.dump(fitted_parameters, f, indent=4, cls=NumpyEncoder)
get_survival_times_application(fitted_parameters, 'fitted_survival_times.csv')
# perform survival analysis
subprocess.call(['Rscript', 'survival_regression.R'])
