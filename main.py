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



stds_abs = np.empty(0)
for std in stds_abs_raw:
    abs = sum(std) / len(std)
    stds_abs = np.append(stds_abs, abs)



stds_table = [
    ('Tartrazine', stds_conc),
    #('T1 Abs', stds_abs_raw[:, 0]),
    #('T2 Abs', stds_abs_raw[:, 1]),
    #('T3 Abs', stds_abs_raw[:, 2]),
    ('Corrected Abs', stds_abs)
]
pretty_print(stds_table, numbered=True)

stds_conc_repeated = np.array([[std] * 3 for std in stds_conc]).flatten()
stds_abs_flattened = stds_abs_raw.flatten()
x_plot = [conc.magnitude.nominal_value for conc in stds_conc_repeated]
y_plot = [abs.magnitude.nominal_value for abs in stds_abs_flattened]
x_err = [conc.magnitude.std_dev for conc in stds_conc_repeated]
y_err = [abs.magnitude.std_dev for abs in stds_abs_flattened]
m, b = ODR(x_plot, y_plot, x_err, y_err)

residuals = ortho_residuals(x_plot, y_plot, m.nominal_value, b.nominal_value)
residuals = np.array(residuals)

fig = pretty_plot(x_plot, y_plot, x_err, y_err, residuals,
                  "Absorption of Tartrazine", m=m, b=b, x_label='Tartrazine / ppm',
                  y_label='Absorption / dimensionless')
fig.savefig('abs.jpg')
print(m, b)

tart_mm = 534.3 * ureg.g / ureg.mole
(m * tart_mm)
