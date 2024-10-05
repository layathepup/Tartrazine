import math
import os

import numpy as np
import scipy as sp
from pint import UnitRegistry
from uncertainties import ufloat as uf
from uncertainties.umath import *
from matplotlib import pyplot as plt

from util import *
from data_loader import *


tart_mm = 534.3 * ureg.g / ureg.mole

stds_abs = np.empty(0)
for trials in stds_abs_raw:
    std = [t.magnitude.nominal_value for t in trials]
    mean = np.mean(std)
    dev = np.std(std, ddof=1)
    std = Q_(mean).plus_minus(dev)
    stds_abs = np.append(stds_abs, std)



stds_table = [
    ('Tartrazine', stds_conc),
    #('T1 Abs', stds_abs_raw[:, 0]),
    #('T2 Abs', stds_abs_raw[:, 1]),
    #('T3 Abs', stds_abs_raw[:, 2]),
    ('Corrected Abs', stds_abs)
]
pretty_print(stds_table, numbered=0)

stds_conc_repeated = np.array([[std] * 3 for std in stds_conc]).flatten()
stds_abs_flattened = stds_abs_raw.flatten()
x_plot = [conc.magnitude.nominal_value for conc in stds_conc_repeated]
y_plot = [abs.magnitude.nominal_value for abs in stds_abs_flattened]
x_err = [conc.magnitude.std_dev for conc in stds_conc_repeated]
y_err = [abs.magnitude.std_dev for abs in stds_abs_flattened]
m, b = ODR(x_plot, y_plot, x_err, y_err)
y_fit = lambda y: (y - b.nominal_value) / m.nominal_value
x_fit = lambda x: m.nominal_value * x + b.nominal_value

residuals = ortho_residuals(x_plot, y_plot, m.nominal_value, b.nominal_value)
residuals = np.array(residuals)

fig = pretty_plot(x_plot, y_plot, x_err, y_err, residuals,
                  "Absorption of Tartrazine", m=m, b=b, x_label='Tartrazine / ppm',
                  y_label='Absorption / dimensionless')
fig.savefig('abs.jpg')
print(m, b)

unks_abs = [np.mean(sample) for sample in unks_abs_raw]
unks_conc_raw = [[y_fit(a).magnitude.nominal_value for a in sample] for sample in unks_abs_raw]
unks_conc = [np.mean(sample) for sample in unks_conc_raw]
unks_conc_err = [cali_curve_unknown_error(m.nominal_value, x_plot, y_plot, sample, residuals)
                for sample in unks_conc_raw]

unks_table = [
    ("Absorption", unks_abs),
    ("Concentration / ppm", unks_conc),
    ("Error / ppm", unks_conc_err)
]
pretty_print(unks_table, numbered=1)

#LOD, LOQ
blank_conc = y_fit(stds_abs[0])
LOD = blank_conc.magnitude.nominal_value + 3 * blank_conc.magnitude.std_dev
LOQ = blank_conc.magnitude.nominal_value + 10 * blank_conc.magnitude.std_dev
print('LOD:', LOD, 'LOQ:', LOQ)
pass