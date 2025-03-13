import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# custom pandas aggregate method
percentage = lambda x: x.sum()*100 / len(x)
percentage.__name__ = 'percentage'


def get_errors(data_fit, data_survival, variable, batch_num=None):
    fit = pd.read_csv(data_fit)
    surv = pd.read_csv(data_survival)
    
    full = pd.merge(fit, surv, on=variable, how='left')
    
    full['T1_abs_err'] = (full['T1_true'] - full['T1_hat']).abs()
    full['T2_abs_err'] = (full['T2_true'] - full['T2_hat']).abs()
    full['relapse_time_abs_err'] = ((full['T2_true'] - full['T1_true']) - (full['T2_hat'] - full['T1_hat'])).abs()
    full['weibull_shape_nrm_err'] = (full['weibull_shape_true'] - full['weibull_shape_hat']).abs() / full['weibull_shape_true']
    full['weibull_scale_nrm_err'] = (full['weibull_scale_true'] - full['weibull_scale_hat']).abs() / full['weibull_scale_true']
    full['relapse_hat_whereas_censored_true'] = (full['T2_hat'].notna() & full['T2_true'].isna())
    full['censored_hat_whereas_relapse_true'] = (full['T2_hat'].isna() & full['T2_true'].notna())
    full['relapse_hat_and_relapse_true'] = (full['T2_hat'].notna() & full['T2_true'].notna())
    full['censored_hat_and_censored_true'] = (full['T2_hat'].isna() & full['T2_true'].isna())
    
    if batch_num is not None: full['batch_num'] = np.repeat(batch_num, len(full))
    
    return full


def get_statistics(data, variable):
    result = data.groupby(variable).agg(
        {
            'T1_abs_err': ['mean', 'std'],
            'T2_abs_err': [np.nanmean, np.nanstd],  # take the mean and std only if relapse predicted
            'relapse_time_abs_err': [np.nanmean, np.nanstd],
            'weibull_shape_nrm_err': 'first',
            'weibull_scale_nrm_err': 'first',
            'relapse_hat_whereas_censored_true': percentage,
            'censored_hat_whereas_relapse_true': percentage,
            'relapse_hat_and_relapse_true': percentage,
            'censored_hat_and_censored_true': percentage,
            'visit_every': 'first',
            'sigma': 'first',
            'n_samples': 'first',
            'batch_num': 'first',
        }
    ).reset_index()
    result.columns = result.columns.map('_'.join).str.strip('_')
    return result


def aggregate_statistics(data):
    return pd.concat(data)


def get_confusion_matrix(data, variable):
    names = ['true_relapse', 'false_censoring', 'false_relapse', 'true_censoring']
    variable_range = list(pd.unique(data[variable]))
    cms_array = {v: confusion_matrix(data[data[variable] == v].censored_true, data[data[variable] == v].censored_hat) for v in variable_range}
    cms_dict  = {key: dict(zip(names, [i for i in value.ravel()])) for key, value in cms_array.items()}
    return pd.DataFrame(cms_dict).T.reset_index().rename(columns={'index': variable})


def aggregate_confusion_matrices(data, variable):
    result = pd.concat(data)
    result = result.groupby(variable).agg(['mean', 'std'])
    result.columns = [f'{name}_{stat}' for name, stat in result.columns]
    return result.round(2).reset_index()



