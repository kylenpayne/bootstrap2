import os
os.chdir('bootstrap')

import numpy as np
from boot import Bootstrap
from boot import Jackknife
from scipy import stats

## test data
from datetime import datetime

times = np.zeros((1000, 2))

for n in range(0, 1000):
    test = np.random.binomial(100, .5, n)
    start = datetime.now()
    b = Bootstrap(test, np.mean)
    b.sample(n_samples=1000)
    diff = datetime.now() - start
    times[n, 0] = n
    times[n, 1] = diff.seconds
    if n % 100 == 0:
        print(n, '\n')


jk = Jackknife(test, np.mean)
jk.jackknife_statistic()

jk.jacknife_results


test = np.random.binomial(100, .5, 100)
start = datetime.now()
b = Bootstrap(test, np.mean)
b.sample(n_samples=1000, ci_type='pivotal')

b.sem
