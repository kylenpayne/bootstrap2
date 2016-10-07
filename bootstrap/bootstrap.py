"""This module contains the Bootstrap class, which represents the
bootstrap resampled estimator object.

TODO: Implement ahead of time compilation of the
_bootstrap_sample() and _bootstrap_matrixsample() methods.

"""
from statistic import Statistic
from collections import namedtuple
import numpy as np
from scipy import stats
from tabulate import tabulate


class Bootstrap(Statistic):
    """A class for the Bootstrap object.
    """
    def __init__(self, data, func=None):
        """Initializes an object of the Bootstrap class.

        Args:
        ---------
        data : array or matrix
            data array for computing bootstrap
        func : function
            statistical function that defines statistic.

        Returns:
        ---------
        None
        """
        super(Bootstrap, self).__init__(data, func)

    ## bootstrap sample
    def _bootstrap_sample(self):
        """Resamples data by random sample with replacement

        Args
        ---------
        None

        Returns:
        ---------
        resamples : array
            bootstrap resampled data
        """
        dists = ['normal', 'uniform', 'poisson']
        if self.parametric and self.parametric not in dists:
            raise ValueError("Invalid parametric argument.")

        sample_size = len(self.data)
        if self.parametric == dists[0]:
            # Normal distribution
            mean_estimate = np.mean(self.data)
            std_estimate = np.std(self.data)
            return np.random.normal(mean_estimate, std_estimate, size=sample_size)
        elif self.parametric == dists[1]:
            # Uniform distributuon
            min_estimate, max_estimate = np.min(self.data), np.max(self.data)
            return np.random.uniform(min_estimate, max_estimate, size=sample_size)
        elif self.parametric == dists[2]:
            # Poisson distribution
            lambda_estimate = np.mean(self.data)
            return np.random.poisson(lam=lambda_estimate, size=sample_size)
        else:
            inds = [np.random.randint(0, sample_size) for i in range(sample_size)]
            return self.data[inds]

    # just-in-time compiled function for faster performance
    # @jit
    def _bootstrap_matrixsample(self):
            """Resamples a matrix by rows or columns

            Args:
            ---------
            None

            Returns:
            ---------
            resamples : matrix
                bootstrap resampled data
            """

            if axis == 0:
                n_rows = np.shape(self.data)[0]
                samples = np.random.randint(n_rows, size=n_rows)
                bootstrap_matrix = data[samples, :]
            elif axis == 1:
                n_cols = np.shape(self.data)[1]
                samples = np.random.randint(n_cols, size=n_cols)
                bootstrap_matrix = self.data[:, samples]
            return bootstrap_matrix

    def _bootstrap_statistic(self):
        """Bootstraps a statistic and calculates the standard error of the statistic
        Args:
        ---------
        None
        Returns:
        ---------
        None
        """
        plugin_estimate = self.func(self.data)
        statistics = []
        # Compute statistics and mean it to get statistic's value
        # this way gets rid checking the data_type every interation
        if  (self.data_type is np.matrix
            or (self.data_type is np.ndarray
            and self.data.ndim == 2)):

            for sample in range(self.n_samples):
                resample = self._bootstrap_matrixsample()
                statistic = self.func(resample)
                statistics.append(statistic)
        else:
            for sample in range(self.n_samples):
                resample = self._bootstrap_sample()
                statistic = self.func(resample)
                statistics.append(statistic)

        self.statistic = np.mean(statistics)

        # Compute bias and, if requested, correct for it
        self.bias = self.statistic - plugin_estimate
        if self.bias_correction:
            self.statistic = self.statistic - self.bias

        self.sem = stats.sem(statistics)
        self.statistics = statistics

        # CI for the statistic
        self.confidence_interval = self.calculate_ci()

        """ --- the results can be queried using
        1. a method for any of the data directly,
            b = Bootstrap(data, np.mean)
            b.sample()
            b.statistic -> returns the bootstrap estimated statistic
            b.sem -> returns the bootstrap standard error
            b.bias -> returns the bias of the bootstrap estimator
            b.ci -> returns the confidence interval of bootstrap estimator
        2. a summary method,
            b.summary() -> returns a printed summary of the bootstrap results.
        3. a bootstrap results method.
            returns a namedtuple of the bootstrap results.
            b.bootstrap_results()
        """

        # Pack together the results
        bootstrap_results = namedtuple('bootstrap_results',
                                        'statistics statistic bias sem ci')
        self.bootstrap_results = bootstrap_results(
                                     statistics=self.statistics,
                                     statistic=self.statistic,
                                     bias=self.bias,
                                     sem=self.sem,
                                     ci=self.confidence_interval)


    def calculate_ci(self, alpha=None, bca=None):
        """
        Calculates bootstrapped confidence interval using percentile
        intervals.

        Args:
        ---------
        alpha (float): percentile used for upper and lower bounds of confidence
                interval.  NOTE: Currently, both upper and lower bounds can have
                the same alpha.
        bca (bool): If true, use bias correction and accelerated (BCa) method
        Returns: tuple (ci_low, ci_high)
        ---------
        confidence_interval : (float, float)
            (ci_low, ci_high)
            ci_low - lower bound on confidence interval
            ci_high - upper bound on confidence interval
        """
        ## this works if the user wants to
        ## recalculate the confidence interval
        ## after the initial sampling with a new
        ## confidence level.
        if alpha is None and self.alpha is None:
            self.alpha = alpha = 0.05
        elif alpha is None and self.alpha is not None:
            alpha = self.alpha
        if bca is not None:
            self.bca = bca
        if self.ci_type == 'percentile':
            # If BCa method, update alpha
            if self.bca:
                # Calculate bias term, z
                plugin_estimate = self.func(self.data)
                num_below_plugin_est = len(np.where(self.statistics < plugin_estimate)[0])
                bias_frac = num_below_plugin_est / len(self.statistics)
                z = stats.norm.ppf(bias_frac)
                # Calculate acceleration term, a
                ## initialize a jackknife object.
                j_knife = Jackknife()

                ## jackknife results is a dict, use .items() to get the vals.
                j_statistic, j_sem, j_values = \
                    j_knife().jackknife_statistic(self.data, self.func).jackknife_results.items()

                numerator, denominator = 0, 0
                for value in j_values:
                    numerator += (value - j_statistic)**3
                    denominator += (value - j_statistic)**2
                a = numerator / (6 * denominator**(3/2))
                bca_alpha = stats.norm.cdf(z + (z + stats.norm.ppf(self.alpha)) /
                                           1 - a * (z + stats.norm.ppf(self.alpha)))
                alpha = bca_alpha

            sorted_statistics = np.sort(self.statistics)
            low_index = int(np.floor(alpha * len(self.statistics)))
            high_index = int(np.ceil((1 - alpha) * len(self.statistics)))

            # Correct for 0 based indexing
            if low_index > 0:
                low_index -= 1
            high_index -= 1
            low_value = sorted_statistics[low_index]
            high_value = sorted_statistics[high_index]
            return (low_value, high_value)
        elif self.ci_type == 'pivotal':
            plugin_estimate = self.func(self.data)
            sorted_statistics = np.sort(self.statistics)
            high_index = int(np.floor(alpha*len(self.statistics)))
            low_index = int(np.ceil((1-alpha)*len(self.statistics)))
            return (2*plugin_estimate - sorted_statistics[low_index],
                    2*plugin_estimate - sorted_statistics[high_index])



    def sample(self, parametric=None, n_samples=50, bias_correction=False,
                     bca=False, axis=1, alpha=0.05, ci_type='pivotal'):
        """Resamples data by random sample with replacement.
        If a function is passed, then the returned values are
        resampled statistic defined by the function.

        Args:
        ---------
        parametric : str in ['normal', 'uniform', 'poisson']
            parametric distribution to resample from,
            if False, use nonparametric bootstrap sampling

        Returns:
        ---------
        resamples : array
            bootstrap resampled data

        """
        self.ci_type = ci_type
        self.bca = bca
        self.parametric = parametric
        self.n_samples = n_samples
        self.alpha = alpha
        self.bias_correction = bias_correction
        # No statistical function is given, and the
        # data type is a one dimensional array or list
        if self.func is None:

            if  ((self.data_type == np.ndarray
                and self.data.ndim < 2)
                or (self.data_type == list
                and not any(isinstance(i, list) for i in self.data))):

                # turn list data into a numpy array. this is mostly
                # so the data behaves well with the jit functions.
                if self.data_type == list:
                    self.data = np.array(self.data)
                # calculate the bootstrap sample
                return self._bootstrap_sample()

            if ((self.data_type == np.ndarray and self.data.ndim == 2) or
                (self.data_type == np.matrix)):
                return self._bootstrap_matrixsample()

        # If the function is not None.
        elif self.func is not None:
            self._bootstrap_statistic()

    # @property
    def se(self):
        """ Returns the standard error of the bootstrapped estimator
        Args:
        ---------
        None

        Returns:
        ---------
        sem : float
            bootstrap standard error estimate
        """
        return self.sem

    # @property
    def bias(self):
        """Returns the bis of the bootstrap estimator

        Args:
        ---------
        None

        Returns:
        ---------
        bias : float
            bootstrap bias estimate
        """
        return self.bias

    def summary(self):
        """Return a summary of the bootstrap procedure.

        Args:
        ---------
        None

        Returns:
        ---------
        None : Prints out a summary.
        """
        return tabulate(self.bootstrap_results)



## end of Bootstrap class


## TODO: Finish the docstring for the Jackknife class.

class Jackknife(Statistic):
    """A jacknife statstic object.
    """

    def __init__(self, data=None, func=None):
        """
        Initializes the Jackknife object.

        Args:
        ---------
        data : np.array
            array of data to resample
        index : int
            Index of array to leave out in jackknife sample

        """
        super(Jackknife, self).__init__(data, func)



    def _jackknife_sample(self, data=None, index=None):
        """
        Single jackknife sample of data

        Args:
        ---------
        data : np.array
            array of data to resample
        index : int
            Index of array to leave out in jackknife sample

        Returns:
        ---------
        resamples : array
            jackknife resampled data
        """

        ## if no data passed, copy from self.data
        ## if no self.data, then cause a error
        if data is None:
            try:
                data = self.data
            except NameError:
                raise ValueError("""data must be passed to the Jackknife object, or the
                                     _jackknife_sample method.""")

        jackknife = np.delete(data, index)
        return jackknife




    def jackknife_statistic(self, data=None, func=None):
        """Jackknifes a statistic and calculates the standard error of the statistic

        Args:
        ---------
        data : array
            array of data to calculate statistic and SE of statistic
        func : function
            statistical function to calculate on data
            examples: np.mean, np.median

        Returns:
        ---------
        jackknifed_stat : (float, float, float)
            (statistic, sem, statistics)
        Returns the jackknifed statistic and the SEM of the statistic
        """
        if data is None:
            try:
                data = self.data
            except NameError:
                raise ValueError("""data must be passed to the Jackknife object, or the
                                     _jackknife_statistic method.""")
        if func is None:
            try:
                func = self.func
            except NameError:
                raise ValueError("""func must be passed to the Jackknife object, or the
                                     jackknife_statistic method.""")

        n_samples = len(self.data)
        statistics = []

        for sample in range(n_samples):
            jack_sample = self._jackknife_sample(data, sample)
            statistic = func(jack_sample)
            statistics.append(statistic)

        self.est = np.mean(statistics)
        self.statistics = statistics
        self.sem = stats.sem(statistics)
        ## put the jackknife results in a tuple.
        self.jacknife_results = {'Jackknife Estimate' : self.est,
                                 'Standard Error' : self.sem,
                                 'Jackknife Samples' : self.statistics}

    def summary(self):
        """Returns a pretty summary of the Jackknife statistic.
        """
        return tabulate(self.jacknife_results)

## end of Jackknife class
