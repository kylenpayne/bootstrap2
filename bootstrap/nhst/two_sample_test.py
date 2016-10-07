"""This module contains resampling based two-sample tests."""

def two_sample_testing(sampleA, sampleB,
                       statistic_func=None, n_samples=50):
    """
    Compares two samples via bootstrapping to determine if they came from
    the same distribution.

    Args:
    ---------
    sampleA : np.array
        Array of data from sample A
    sampleB : np.array
        Array of data form sample B
    statistic_func : function
        Function that compares two data sets and retuns a statistic. Function
        must accept two args, (np.array, np.array), where each array is a
        sample.
        Example statistics_func that compares the mean of two data sets:
            lambda data1, data2: np.mean(data1) - np.mean(data2)
    n_samples : int
        number of bootstrap samples to generate

    Returns:
    ---------
    sig_lvl : float
        bootstrapped achieved significance level
    """
    if statistic_func is None:
        statistic_func = compare_means

    observed_statistic = statistic_func(sampleA, sampleB)
    combined_sample = np.append(sampleA, sampleB)

    # Count the number of bootstrap samples with statistic > observed_statistic
    m = len(sampleA)
    counter = 0
    bs = Bootstrap(combined_sample)
    for sample in range(n_samples):
        boot_sample = bs.sample()
        boot_sampleA = boot_sample[:m]
        boot_sampleB = boot_sample[m:]
        boot_statistic = statistic_func(boot_sampleA, boot_sampleB)
        if boot_statistic > observed_statistic:
            counter += 1

    ASL = counter / float(n_samples)
    return ASL

def compare_means(sampleA, sampleB):
    """
    Compares the mean of two samples

    Args:
    ---------
    sampleA (np.array): Array of data from sample A
    sampleB (np.array): Array of data form sample B

    Returns:
    ---------
    difference : float
        difference in mean between the two samples
    """
    difference = np.mean(sampleA) - np.mean(sampleB)
    return difference

def t_test_statistic(sampleA, sampleB):
    """
    Computes the t test statistic of two samples

    Args:
    ---------
    sampleA : np.array
        Array of data from sample A
    sampleB : np.array
        Array of data form sample B

    Returns:
    ---------
    t_stat : float
        t test statistic of two samples
    """
    difference = compare_means(sampleA, sampleB)
    # Store lengths of samples
    n = len(sampleA)
    m = len(sampleB)
    stdev = (np.var(sampleA)/n + np.var(sampleB)/m)**0.5
    t_stat = difference / stdev
    return t_stat
