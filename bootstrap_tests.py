import os
os.chdir('bootstrap')

from nhst.multiple_testing import stepwise_resampling

## make some fake data

import numpy as np
sigma_y = np.ndarray((5,5), buffer=
    np.array([[1,.5, 0, 0, 0],
             [.5, 1, .3, 0, 0],
             [0, .3, 2, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]))

Y = np.random.multivariate_normal(np.array([30,30,-40, 1, 0]), sigma_y, 1000)

from scipy import stats

def t_stat(data, m0=0):
    """calculates a t test statistic.

    Args:
        data (np.array) : A numpy array.
        m0 (float) : the null-hypothesis mean.

    Returns:
        t test statistic.
    """
    return (np.mean(data) - m0)/(np.std(data)/np.sqrt(len(data)))


stepwise_resampling(Y, t_stat, 0, alpha=0.05)


help(stepwise_resampling)
