import sys
import numpy as np


epsilon = 1e-8


def compute_average(data):
    """ Return the average of each column. """
    sample_size = data.size
    return np.sum(data, axis=0) / len(data) if sample_size != 0 else None


def compute_median(data):
    """ Return the median of each column. """
    sample_size = data.shape[0]
    if sample_size != 0:
        m = sample_size // 2
        median = data[m, :] if sample_size % 2 else (data[m, :] + data[m - 1, :]) / 2
    else:
        raise Exception("Array is empty.")
    return median


def compute_variance(data):
    """ Return variance of each column. """
    sample_size = data.shape[0]
    if sample_size != 0:
        avg = compute_average(data)
        std2 = np.sum(np.power(data - avg, 2), axis=0) / sample_size
    else:
        raise Exception("Array is empty.")
    return std2


def compute_std(data):
    """ Return standard deviation of each column. """
    if data.shape[0] != 0:
        sigma_2 = compute_variance(data)
        std = np.sqrt(sigma_2)
    else:
        raise Exception("Array is empty.")
    return std


def compute_half_sum_extreme_values(data):
    """ Return the half-sum of "extreme" observations. """
    sample_size = data.shape[0]
    if sample_size != 0:
        maximum = np.max(data, axis=0)
        minimum = np.min(data, axis=0)
        half_sum_extreme = (maximum - minimum) / 2
    else:
        raise Exception("Array is empty.")
    return half_sum_extreme


def compute_med_absolute_deviation(data):
    """ Return median absolute deviation of each column. """
    sample_size = data.shape[0]
    avg_abs_deviation = None
    if sample_size != 0:
        med = compute_median(data)
        avg_abs_deviation = np.sum(np.absolute(data - med),
                                   axis=0) / sample_size
    else:
        raise Exception("Array is empty.")
    return avg_abs_deviation


def compute_range(data):
    """ Return range of each column. """
    if data.shape[0] != 0:
        maximum = np.max(data, axis=0)
        minimum = np.min(data, axis=0)
        ptp = np.subtract(maximum, minimum)
    else:
        raise Exception("Array is empty.")
    return ptp


def centralize(data):
    """ A method for normalizing data to mean=0. """
    if data.shape[0] != 0:
        centralized = data - compute_average(data)
    else:
        raise Exception("Array is empty.")
    return centralized


def z_score(data):
    """ A method for normalizing data to mean=0 and standard deviation=1. """
    if data.shape[0] != 0:
        std = compute_std(data)
        centralized = centralize(data)
        z_score = centralized / (std + epsilon)
    else:
        raise Exception("Array is empty.")
    return z_score


def min_max_normalization(data):
    """ Return normalized data in the range [0, 1]. """
    if data.shape[0] != 0:
        ptp = compute_range(data)
        minimum = np.min(data, axis=0)
        min_max_scaled = (data - minimum) / (ptp + epsilon)
    else:
        raise Exception("Array is empty.")
    return min_max_scaled


def hypercube_normalization(data):
    """ Return normalized data in the range [-1, 1]. """
    if data.shape[0] != 0:
        min_max = min_max_normalization(data)
        hypercube_scaled = 2 * min_max - 1
    else:
        raise Exception("Array is empty.")
    return hypercube_scaled


def stream_avg(stdin, feature_n):
    """
    A method for computing average of the observations are received as a stream.
    Parameters:
    -----------
    stdin: io.TextIOWrapper
        Data stream
    feature_n: int:
        The number of attributes of the dataset sample
    """
    recurrent_avg = np.array([0.0 for _ in range(feature_n)])
    for i, data in enumerate(iter(stdin.readline())):
        data = np.array(data.split()[:feature_n], dtype='float64')
        recurrent_avg += (data - recurrent_avg) / (i + 1)
    return recurrent_avg 


def stream_median(stdin, feature_n):
    """
    A method for computing median of the observations are received as a stream.
    Parameters:
    -----------
    stdin: io.TextIOWrapper
        Data stream
    feature_n: int:
        The number of attributes of the dataset sample
    """
    recurrent_med = np.array([0.0 for _ in range(feature_n)])
    for i, data in enumerate(iter(stdin.readline())):
        data = np.array(data.split()[:feature_n], dtype='float64')
        recurrent_med += np.sign(data - recurrent_med) / (i + 1)
    return recurrent_med