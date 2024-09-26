import csv
import math

from typing import List
from pathlib import Path

import numpy as np
import scipy as sp

from tabulate import tabulate
from uncertainties.umath import *


def Read(file_path: Path) -> csv.reader:
    """
    :param file_path:
    :return:
    Open a csv file using stdlib cvs.
    Skip the header row and return the reader.
    """
    fd = open(file_path)
    reader = csv.reader(fd)
    next(reader)
    return reader


def Flatten(super_list: list[list]):
    return [item for sub_list in super_list for item in sub_list]


def min_transfer(total: int, parts: List[int]) -> List[int]:
    """
    :param total:
    :param parts:
    :return:
    Given an integer total and an integer list of partitions,
    construct the minimum number of partitions that sum to the total as a list of integers.
    Allow repeated partitions.
    ex. construct 10 from [1,4,7] returns [4,4,1,1]
    TODO: memoization for recursion
    """
    parts.sort(reverse=True)
    if len(parts) == 0:
        return None
    for i in range(len(parts)):
        if parts[i] == total:
            return [parts[i]]
        if parts[i] < total:
            curr = min_transfer(total - parts[i], parts[i:])
            if curr:
                curr.append(parts[i])
            next = min_transfer(total - parts[i], parts[i + 1:])
            if next:
                next.append(parts[i])
            if curr is None or next is None:
                return curr if curr is not None else next
            return curr if len(curr) < len(next) else next
    else:
        return None


def ODR(x, y, x_dev, y_dev):
    model = sp.odr.Model(linear_model)
    weights = lambda errs: [(1 / (err + max(errs) * 1e-10) ** 2) for err in errs]
    y_weights = weights(y_dev)
    x_weights = weights(x_dev)
    data = sp.odr.Data(x, y, we=y_weights, wd=x_weights)
    odr = sp.odr.ODR(data, model, beta0=[1., 0.])
    return odr.run()


def linear_model(params, x):
    m, b = params
    return m * x + b


def pretty_print(table: [], title=None, number=False):
    if number:
        num = [i for i in range(1, len(table[0][1]) + 1)]
        table.insert(0, ('num', num))
    headers = [tup[0] for tup in table]
    # create table of values as list of sublist of values
    # than transpose to create list of rows
    columns = [list(tup[1]) for tup in table]
    rows = [list(column) for column in zip(*columns)]
    print(title if title else '', '\n', tabulate(rows, headers=headers, tablefmt='grid'))


def ortho_projection(x, y, m, b):
    """
    Calculate orthogonal projection of point onto a line.
    Return tuple (x_projection, y1_projection).
    """
    m1 = -1 / m
    b1 = y - m1 * x
    x_proj = (b1 - b) / (m - m1)
    y_proj = m * x_proj + b
    return x_proj, y_proj


def points_distance(x0, y0, x1, y1):
    x = (x1 - x0) ** 2
    y = (y1 - y0) ** 2
    return math.sqrt(x + y)


def ortho_projection_displacement(x, y, m, b):
    x_proj, y_proj = ortho_projection(x, y, m, b)
    dist = points_distance(x, y, x_proj, y_proj)
    if y_proj < y:
        dist *= -1
    return dist


def cali_curve_unknown_error(m, x, y, y_sample, residuals):
    """
    :param m: slope of calibration curve
    :param x: set of individual x values in calibration curve
    :param y: set of individual y values in calibration curve
    :param y_sample: unknown signal replicates
    :param residuals: residuals of calibration points to linear regression (vertical or orthogonal)
    :return: error in unknown x value
    """
    assert len(x) == len(y)
    s_y = curve_uncertainty(residuals)
    term1 = 1 / len(y_sample)
    term2 = 1 / len(x)
    term3_nom = (np.mean(y_sample) - np.mean(y)) ** 2
    term3_denom = m ** 2 * sum([(x_i - np.mean(x)) ** 2 for x_i in x])
    term3 = term3_nom / term3_denom
    factor1 = s_y / abs(m)
    factor2 = math.sqrt(term1 + term2 + term3)
    return factor1 * factor2


def std_add_curve_unknown_error(m, x, y, y_sample, residuals):
    assert (len(x) == len(y))
    s_y = curve_uncertainty(residuals)
    term1 = 1 / len(x)
    term2 = ((np.mean(y_sample) - np.mean(y)) / (m * sum(x - np.mean(x)))) ** 2
    factor1 = s_y / math.abs(m)
    factor2 = math.sqrt(term1 + term2)
    return factor1 * factor2


def curve_uncertainty(residuals):
    nom = sum([r ** 2 for r in residuals])
    denom = len(residuals) - 2
    return math.sqrt(nom / denom)


def F_test(x0, x1):
    """
    Tests if the standard deviations of two distributions are different.
    :param x0: distribution one
    :param x1: distribution two
    :return: F_calculated, compare to F_table, iff F_c > F_t then the standard deviations are different
    """
    assert len(x0) == len(x1)
    x0_std, x1_std = np.std(x0), np.std(x1)
    a, b = (x0_std, x1_std) if x0_std > x1_std else (x1_std, x0_std)
    return (a / b) ** 2


def t_test1(x, val):
    """
    Re-arrangement of standard error of the mean.
    Compare measurement mean to known value.
    :param x: measured values
    :param val: comparison value
    :return: t_calculated, compare to t_table for (n-1) dof and desired CI
    """
    factor1 = math.abs(np.mean(x) - val) / np.std(x)
    factor2 = math.sqrt(len(x))
    return factor1 * factor2


def t_test2_F_pass(x0, x1):
    """
    For two samples using the same method or the same sample with two different methods,
    where the F-test is passed, i.e. std devs are not different, F_c < F_t
    There are n_0 + n_1 - 2 dof
    """
    factor1 = math.abs(np.mean(x0) - np.mean(x1)) / s_pooled(x0, x1)
    factor2 = math.sqrt((len(x0) * len(x1)) / (len(x0) + len(x1)))
    return factor1 * factor2

def s_pooled(x0, x1):
    nom = np.std(x0) ** 2 * (len(x0) - 1) + np.std(x1) ** 2 * (len(x1) - 1)
    denom = len(x0) + len(x1) - 2
    return math.sqrt(nom / denom)


def t_test2_F_fail(x0, x1):
    """
    For two samples using the same method or the same sample with two different methods,
    where the F-test is failed, i.e. std devs are different, F_c !< F_t
    """
    rel_std = lambda x: np.std(x) ** 2 / len(x)
    nom = math.abs(np.mean(x0) + np.mean(x1))
    denom = math.sqrt(rel_std(x0) + rel_std(x1))
    t_calc = nom / denom

    dof_nom = (rel_std(x0) + rel_std(x1)) ** 2
    dof_denom = rel_std(x0) ** 2 / (len(x0) - 1) + rel_std(x1) ** 2 / len(x1) - 1
    dof = dof_nom / dof_denom

    return (t_calc, dof)


def t_test3(x0, x1):
    """
    For comparing two sets of measurements measured two different ways,
    where each measurement is in each set is not expected to be same,
    i.e. measuring outside temperature at different times.
    Compare to t_table for n - 1 dof.  Iff t_calc > t_table then they are different
    """
    d = [x_0 - x_1 for x_0, x_1 in zip(x0, x1)]
    return math.sqrt(len(d)) * np.mean(d) / np.std(d)


def grubbs_test(x, val):
    """
    Returns difference between x_mean and val in x as percent (0 to 1) of dev.
    Iff G_calc > G_table, val can be discarded.
    Only valid for 1-D data point.
    Can word with n-D data point by applying grubbs test to the residuals.
    """
    return abs(val - np.mean(x)) / np.std(x)

def LOD(blanks):
    return np.mean(blanks) + 3 * np.std(blanks)


def LOQ(blanks):
    return np.mean(blanks) + 10 * np.std(blanks)

