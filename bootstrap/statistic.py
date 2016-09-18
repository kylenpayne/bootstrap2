"""This module implements an abstract statistic class for any resampled statistic
object to inherit. This is a useful implementation as the arguments, data,
IO, and many other features should be standardized across the multiple types of
objects. This will cut down on the overall boilerplate code that would need to
be written to implement resampling statistical objects."""

from abc import ABCMeta
from abc import abstractmethod
import numpy as np

class Statistic():
    """Defines a statistic base class for all
    resampling statistics.
    """
    def __init__(self, *args, **kwargs):
        """ Initializes the statistic class.

        Args:
        -------

        data : A numpy array, pandas DataFrame of a list-like object of
        these.
        func : A function that defines the form of the statistic, or a like-like
        object of functions.

        Returns:
        ---------
        None
        """
        ## sets the columns equal to None, if the data is a dataframe or
        ## list of dataframes, then this will be set to a string or list
        ## of the string names of the columns to pull out of the dataframe.
        self.columns = None
        self.func = None
        self.data = args[0]
        self.func = args[1]

        for name, value in kwargs.items():
            if name == "data":
                self.data = value
            elif name == "func":
                self.func = value
            # elif name == "columns":
            #     ## columns are if the data is a pd.DataFrame.
            #     ## these are the columns to pull out of the dataframe
            #     ## to perform the sampling on.
            #     self.columns = value
        # self.data = data
        # self.func = func
        # self.statistic = self.apply()
        self.data_type = type(self.data)

        ## it appears that most numpy arrays are of type np.ndarray
        if self.data_type is np.ndarray:
            pass
        ## if the data_type is a pd.Series, just grab the values
        elif self.data_type is pd.Series:
            ## this should return a numpy array.
            self.data = data.values

        ## if the data_type is a pd.DataFrame, get the columns
        ## check that the columns are numeric and then set that as the data
        elif self.data_type is pd.DataFrame and self.columns == None:
            ValueError("""A pandas DataFrame object must be passed with
                          columns to subset on.""")
        ## convert the DataFrame into a numpy array.
        elif self.data_type is pd.DataFrame and self.columns != None:
            self.data = pd.DataFrame.ix[:, columns].values
        ## if the data is a list, then check the type of the data in the list.
        elif self.data_type is list:
            data_types = [type(data) for data in self.data]
            ## check that the types of data in the list are homogenous
            if len(list(set(data_types))) > 1:
                ValueError("""Elements of a list of data must
                              be the same type""")
    #
    # @abstractmethod
    # def apply(self):
    #     """Applies the method to the data.
    #     For bootstrapping, this would be an application of the bootstrap.
    #     For jackknife, this would be an application of the jackknife.
    #     """
    #     pass
    #
    # @abstractmethod
    # def sample(self):
    #     """Abstract method for the bootstrap and the jackknife statistic.
    #     """
    #     pass
    #
    # @abstractmethod
    # def summary(self):
    #     """Produce a summary of the Statistic object"""
    #     pass
    #
    # @abstractmethod
    # def bias(self):
    #     """Estimate of the bias of the estimator
    #     """
    #     pass
    #
    # @abstractmethod
    # def se(self):
    #     """Estimate of the standard error of the estimator.
    #     """
    #     pass
