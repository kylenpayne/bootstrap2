""" This module contains functions for performing a resampling-based
step-down multiple testing procedure for a set of test statistics"""

import os
import sys
## sys.path.insert(0, os.path.abspath('..'))
from boot import Bootstrap
import numpy as np

def stepwise_resampling(data, test_stat, t0, alpha=0.05,
                        alternative="two-sided", n_boot=100):
    """Performs stepwise_resampling multiple comparisions procedure for
    given alternative and number of boot samples.

    Args:
        data (np.ndarray): A N \times K numpy array of the data of sample
        size N for K multiple tests.
        (or)
        data (list): A length K list of np.arrays with lengths N_1, ..., N_K
        containing data for multiple tests.
        test_stat: A function or list of functions that define the functional
        form of the test statistic.
        alpha: The significance level of the test.
        alternative: A string in ['two-sided', 'less-than', 'greater-than']
        that determines the alternative in the simple null hypothesis.
        n_boot: The number of bootstrap samples performed to the sample.

    Return:
        A dictionary with the significant p-values, test statistics

    Method based on SR 3 in -
    Troendle, J. F. (1995). A stepwise resampling method of multiple hypothesis
    testing. Journal of the American Statistical Association, 90(429), 370-378.
    """

    ## -- main algorithm
    if type(data) is list:
        n_tests = len(data)
        if type(test_stat) is not list:
            test_stat = [test_stat]*n_tests
        elif (type(test_stat) is list
        and len(test_stat) != n_tests):
            raise ValueError('''The number of supplied test statistics
            must either be 1 or the number of data arrays passed.''')

        ## get a list of the initial t values
        t_init = []
        for k in range(n_tests):
            ts = test_stat[k]
            t_init.append(test_stat(data[k]))

    elif ((type(data) is np.ndarray)
       and data.ndim == 2):
        n_tests = data.shape[1]
        if type(test_stat) is not list:
            test_stat = [test_stat]*n_tests
        elif (type(test_stat) is list
        and len(test_stat) != n_tests):
            raise ValueError('''The number of supplied test statistics
            must either be 1 or the number of data arrays passed.''')

        t_init = []
        for k in range(n_tests):
            ts = test_stat[k]
            t_init.append(ts(data[:, k]))


    ## get the original list of the t-statistics
    ## with the original ordering as a list of tuples
    ##
    sorted_t_init = sorted(enumerate(t_init), key=lambda x: x[1])

    # get the list of the original ordering
    t_init_indices = [st[0] for st in sorted_t_init]
    t_init_values = [st[1] for st in sorted_t_init]

    ## reorders the data into the order given
    if type(data) is list:
        data_ordered = [data[k] for k in t_init_indices]
    elif type(data) is np.ndarray:
        data_ordered = np.zeros_like(data)
        ## create a new array of the data in the
        ## order of the t-statistics i.e.
        ## t_1 < ... < t_k
        ## => data_1 < ... < data_k
        for i, k in enumerate(t_init_indices):
            data_ordered[:,i] = data[:, k]

    test_stat_ordered = [test_stat[k] for k in t_init_indices]
    ## --- start of the algorithm

    boot_results = np.ndarray((n_tests, n_boot))
    for k in range(n_tests):
        if type(data) is list:
            b = Bootstrap(data_ordered[k], test_stat[k])
        elif type(data) is np.ndarray:
            b = Bootstrap(data_ordered[:,k], test_stat[k])
        b.sample(n_samples=n_boot, alpha=alpha)
        ## but the bootstrap results for the kth test
        ## into the boot_results array.
        boot_results[k,:] = np.array(b.statistics)

    hyp = test_results = np.empty((n_tests, 1))
    w  = 1
    while True:
        ## --- set the significance level
        print(w)
        print(test_stat_ordered)
        print(n_tests)
        alpha_w = np.mean(boot_results.max(axis=0)
                > test_stat_ordered[n_tests - w ])

        if alpha_w >= alpha:
            hyp[0:n_tests-w+1] = 0
            break
        else:
            hyp[n_tests-w+1:] = 1
            w += 1

    for i, test in enumerate(hyp):
        if test == 0:
            k = t_init_values[i]
            test_results[k] = False
        elif test == 1:
            k = t_init_values[i]
            test_results[k] = True

    ## returns an array of True False values for tests to reject
    return test_results
