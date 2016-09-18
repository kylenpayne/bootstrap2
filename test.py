import os
os.chdir('bootstrap')

import numpy as np
from boot import Bootstrap
from boot import Jackknife
from scipy import stats

## test data
test = np.random.binomial(100, .5, 100)
# test
#
# b = Bootstrap(test, np.mean)
#
# b.sample(n_samples=1000)
#
#
# b.bootstrap_results[4]


jk = Jackknife(test, np.mean)
jk.jackknife_statistic()

jk.jacknife_results
