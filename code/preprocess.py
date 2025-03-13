import json

import numpy as np
import pandas as pd

from utils import NumpyEncoder

data = pd.read_csv('../data/ifmdfci.csv', delimiter=',')

# remove observations if first observed spike is lower than second one
data = data.groupby('subject').apply(lambda x: x.loc[((x['spike'].shift(fill_value=0) <= x['spike'])).idxmin() - 1:])
data.reset_index(drop=True, inplace=True)

# remove subjects for which the spike on first visit is below threshold
data = data.groupby('subject').filter(lambda x: x['spike'].iloc[0] > 5)
data.reset_index(drop=True, inplace=True)

# remove subjects never reaching remission
data = data.groupby('subject').filter(lambda x: sum(x['spike'].lt(5)) > 1)
data.reset_index(drop=True, inplace=True)

# remove isolated zeros (two neighboors > 5)
m1 = (data['spike'] == 0)
m2 = np.abs(data['spike'] - data['spike'].shift(-1)) > 5
m3 = np.abs(data['spike'] - data['spike'].shift(1)) > 5
m4 = (data['subject'] == data['subject'].shift(1))
m5 = (data['subject'] == data['subject'].shift(-1))
data = data[~(m1 & m2 & m3 & m4 & m5)]
data.reset_index(drop=True, inplace=True)

# remove subjects with too little visits
data = data.groupby('subject').filter(lambda x: len(x) > 10)
data.reset_index(drop=True, inplace=True)

# save preprocessed dataset
#data.to_csv('ifmdfci_preprocessed.csv', index=False)

# convert to json
subjects = pd.unique(data['subject'])

d = {}
for s in subjects:
    d[s] = {
        'subject': s,
        'spike': np.asarray(data[data['subject'] == s]['spike']),
        'cum_day_diff': np.asarray(data[data['subject'] == s]['cum_day_diff']),
        'remission_thresh': 1,
        'last_visit': max(np.asarray(data[data['subject'] == s]['cum_day_diff']))
    }


with open('../data/ifmdfci_preprocessed.json', 'w') as f:
    json.dump(d, f, indent=4, cls=NumpyEncoder)
