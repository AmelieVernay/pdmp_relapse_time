# File used to generate the figures of the paper.
# Adapt file path in `res_folder` to create your own figures.
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import seaborn as sns


# --------------- plt global parameters ---------------
plt.rc('text', usetex = True)

params = {
      'axes.titlesize': 30,
      'axes.labelsize': 20,
      'xtick.labelsize' :15,
      'ytick.labelsize': 15,
      'text.latex.preamble' : [r'\usepackage{amsmath}'],
      'font.family': 'serif',
      'font.serif': 'Latin Modern Roman',
      'axes.grid': True,
      'axes.grid.axis': 'y',
      'axes.labelpad': 30.0,
      'grid.linestyle': '-',
      'grid.alpha': 0.4,
      'figure.autolayout': True, # tight_layout()
      'patch.edgecolor': 'w',
     }
plt.rcParams.update(params)

# simply change scenario to generate every associated figures
scenario = 'scenario1'
res_folder = f'./results/{scenario}'
res_stats = pd.read_csv(f'{res_folder}/xps_statistics.csv', delimiter=',')
res_batch = pd.read_csv(f'{res_folder}/estimation_errors_last_batch.csv', delimiter=',')

# variable parameter depending on the scenario (delta or sigma or n)
variable = res_batch['variable_parameter'].iloc[0]
variable_range = list(pd.unique(res_batch[variable]))
labels = {
    'visit_every': ('Visit interval (in days)', '\delta'),
    'sigma': ('Noise level', '\sigma'),
    'n_samples': ('Number of trajectories', 'n')
}


# ------------ histograms of relapse times ------------
def histplot_relapse_times(data, weibull_true, weibull_hat, color, binwidth, title=None, label=None):
    plt.figure(figsize=(12, 8))
    ax = sns.histplot(data, color=color, stat='density', binwidth=binwidth)
    # change boxes transparency
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .55))
    # remove borders except left & bottom ones
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Remove y ticks but not y labels
    ax.tick_params(axis='y', which=u'both',length=0)

    plt.xlabel(r'$\widehat{T_2} - \widehat{T_1}$', fontsize=25)
    plt.ylabel(label, rotation=0)
    
    x = np.arange(0, 3000)  # (0, max(data))
    ax.plot(
        x,
        weibull_pdf(x, weibull_true[0], weibull_true[1]),
        color='black',
        alpha=.8,
        linestyle='dashed',
        label='Ground truth'
    )
    ax.plot(
        x,
        weibull_pdf(x, weibull_hat[0], weibull_hat[1]),
        color='black',
        alpha=.8,
        label='Estimated'
    )
    # legend
    leg = ax.legend(loc='best', prop={'family': 'serif'})
    # line width in legend box
    for line in leg.get_lines():
        line.set_linewidth(1.5)
    plt.ylim(0.000, 0.003)
    if title is not None: plt.title(title)
    return plt


def weibull_pdf(x, shape, scale):
    return (shape/scale) * (x/scale)**(shape-1) * np.exp(-(x/scale)**shape)


for v in variable_range:
    data = res_batch[res_batch[variable] == v]
    
    delta = data['visit_every'].iloc[0]
    sigma = data['sigma'].iloc[0]
    n_samples = data['n_samples'].iloc[0]
    
    histplot_relapse_times(
        data[data.relapse_time_hat.notna()].relapse_time_hat,
        weibull_true=(data['weibull_shape_true'].iloc[0], data['weibull_scale_true'].iloc[0]),
        weibull_hat=(data['weibull_shape_hat'].iloc[0], data['weibull_scale_hat'].iloc[0]),
        color='#EB811B',
        binwidth=30,
    )
    image_format = 'png' # e.g .png, .svg, etc.
    image_name = f'relapse_times_{scenario}_{variable}{str(v).replace(".", "")}.png'
    plt.savefig(f'figures/{image_name}', format=image_format, dpi=250)


# ------- boxplots of Weibull parameter errors -------
variable_appended = f'{variable}_first'
res_weibull_errors = res_stats[[variable_appended, 'weibull_shape_nrm_err_first', 'weibull_scale_nrm_err_first']].melt(id_vars=[variable_appended], var_name='Error type', value_name='Relative error')

palette = {'weibull_shape_nrm_err_first': '#EB811B', 'weibull_scale_nrm_err_first': '#A51E37'}
plt.figure(figsize=(15, 10))
ax = sns.boxplot(
    x=variable_appended,
    y='Relative error',
    hue='Error type',
    palette=palette,
    data=res_weibull_errors,
    showfliers=False
)
# change boxes transparency
for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .75))
# remove borders except left & bottom ones
for s in ['top', 'left', 'right']:
    ax.spines[s].set_visible(False)
# Remove y ticks but not y labels
ax.tick_params(axis='y', which=u'both',length=0)

legend_labels = [r'$\frac{|\alpha - \widehat{\alpha}|}{\alpha}$', r'$\frac{|\beta - \widehat{\beta}|}{\beta}$']
legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=to_rgba(color, 0.75), edgecolor='black') for color in palette.values()]
ax.legend(legend_handles, legend_labels, loc='upper left', prop={'size': 20})
plt.xlabel(labels[variable][0], fontsize=25)
plt.ylabel('Relative error', fontsize=25)
plt.ylim(0.000, 0.55)

image_format='png'
image_name = f'weibull_parameters_{scenario}.png'
plt.savefig(f'figures/{image_name}', format=image_format, dpi=250)


# ----------- boxplots of jump time errors -----------
res_batch_sub = res_batch[res_batch.relapse_time_hat.notna()]
res_jump_errors = res_batch_sub[[variable, 'T1_abs_err', 'T2_abs_err']].melt(id_vars=[variable], var_name='Error type', value_name='Absolute error')

palette = {'T1_abs_err': '#3792A4', 'T2_abs_err': '#5E518A'}
plt.figure(figsize=(15, 10))
ax = sns.boxplot(
    x=variable,
    y='Absolute error',
    hue='Error type',
    palette=palette,
    data=res_jump_errors,
    showfliers=False
)
# change boxes transparency
for patch in ax.patches:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .75))
# remove borders except left & bottom ones
for s in ['top', 'left', 'right']:
    ax.spines[s].set_visible(False)
# Remove y ticks but not y labels
ax.tick_params(axis='y', which=u'both',length=0)

ymin, ymax = ax.get_ylim()
plt.yticks(np.arange(0, ymax, 10))

legend_labels = [r'$|T_1 - \widehat{T_1}|$', r'$|T_2 - \widehat{T_2}|$']
legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=to_rgba(color, 0.75), edgecolor='black') for color in palette.values()]
ax.legend(legend_handles, legend_labels, loc='upper left', prop={'size': 20})
plt.xlabel(labels[variable][0], fontsize=25)
plt.ylabel('Absolute error', fontsize=25)
image_format='png'
image_name = f'jumps_{scenario}.png'
plt.savefig(f'figures/{image_name}', format=image_format, dpi=250)

# save number in each group to csv for tables in supp mat
table_name = 'boxplot_counts.csv'
counts = res_jump_errors.groupby(variable)['Error type'].count().reset_index()
counts.rename(columns={'Error type': 'boxcount'}, inplace=True)
counts.to_csv(f'results/{scenario}/{table_name}', index=False)

# ------------- lineplots of Weibull pdfs -------------
fig = plt.figure()
ax = plt.axes()
x = np.arange(0, 3000)
colors = ['#264653','#2a9d8f','#e9c46a','#f4a261','#e76f51', '#BC6C25']

for i, v in enumerate(variable_range):
    data = res_batch[res_batch[variable] == v]
    ax.plot(
        x,
        weibull_pdf(x, data['weibull_shape_hat'].iloc[0], data['weibull_scale_hat'].iloc[0]),
        color=colors[i],
        label=f'${{{labels[variable][1]}}} = {{{v}}}$'
    )
ax.plot(
    x,
    weibull_pdf(x, data['weibull_shape_true'].iloc[0],  data['weibull_scale_true'].iloc[0]),
    color='black',
    alpha=.8,
    linestyle='dashed',
    label='GT'
)
leg = ax.legend(loc='best', prop={'family': 'serif'})
# line width in legend box
for line in leg.get_lines():
    line.set_linewidth(1.5)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
# Change the fontsize of minor ticks label
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)
# Remove y ticks but not y labels
ax.tick_params(axis='y', which=u'both',length=0)
image_format = 'png' # e.g .png, .svg, etc.
image_name = f'weibull_pdfs_{scenario}_{variable}.png'
plt.savefig(f'figures/{image_name}', format=image_format, dpi=250)
