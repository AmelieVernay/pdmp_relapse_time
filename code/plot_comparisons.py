# File used to generate the figures of the paper.
# Adapt file path in `results` to create your own figures.
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

results = pd.read_csv(f'results/comparisons/comparisons.csv', delimiter=',')

rs = results[['rs_pdmp', 'rs_hmm', 'rs_rpt']].melt(var_name='Method', value_name='RS')
ars = results[['ars_pdmp', 'ars_hmm', 'ars_rpt']].melt(var_name='Method', value_name='ARS')

palette = {'rs_pdmp': '#0E172A', 'rs_hmm': '#632137', 'rs_rpt': '#1d8782'}

plt.figure(figsize=(5, 10))
ax = sns.boxplot(
    x='Method',
    y='RS',
    palette=palette,
    data=rs,
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
# Remove xticks labels and xticks but not xaxis legend
ax.set_xticklabels([])
ax.set_xticks([])

legend_labels = ['PDMP', 'HMM', 'Change points']

legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=to_rgba(color, 0.75), edgecolor='black') for color in palette.values()]
ax.legend(legend_handles, legend_labels, loc='lower left', prop={'size': 20})
plt.xlabel('Method', fontsize=25)
plt.ylabel('Rand Index', fontsize=25)

image_format='png'
image_name = 'comparisons_rs.png'
plt.savefig(f'figures/{image_name}', format=image_format, dpi=250)

palette = {'ars_pdmp': '#0E172A', 'ars_hmm': '#632137', 'ars_rpt': '#1d8782'}

plt.figure(figsize=(5, 10))
ax = sns.boxplot(
    x='Method',
    y='ARS',
    palette=palette,
    data=ars,
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
# Remove xticks labels and xticks but not xaxis legend
ax.set_xticklabels([])
ax.set_xticks([])

legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=to_rgba(color, 0.75), edgecolor='black') for color in palette.values()]
ax.legend(legend_handles, legend_labels, loc='lower left', prop={'size': 20})
plt.xlabel('Method', fontsize=25)
plt.ylabel('Adjusted Rand Index', fontsize=25)

image_format='png'
image_name = 'comparisons_ars.png'
plt.savefig(f'figures/{image_name}', format=image_format, dpi=250)
