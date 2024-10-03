from pathlib import Path

from uncertainties import ufloat as uf
from uncertainties.umath import *
from pint import UnitRegistry
from util import *

ureg = UnitRegistry()
Q_ = ureg.Quantity


DAT = Path.cwd() / 'dat'
TOL = Path.cwd() / 'tolerances'
IMG = Path.cwd() / 'img'


# generate dictionaries of glassware tolerances
dicts = [pipettes_TD, vol_flasks_TC] = {}, {}
filenames = ['pipettes_TD.csv', 'vol_flasks_TC.csv']
readers = [Read(Path(TOL) / filename) for filename in filenames]
for filename, reader in zip(filenames, readers):
    dict_name = Path(filename).stem
    locals()[dict_name] = {}
    for row in reader:
        nom, mean, err = int(row[0]), float(row[1]), float(row[2])
        locals()[dict_name][nom] = Q_(mean * ureg.ml).plus_minus(err)


# from a csv of transfer to total volume then from that another transfer to another total volume and so on
# transfer, volume, transfer, volume ...
stock_conc = (0.4534 * ureg.g).plus_minus(0.0001) / (500.00 * ureg.ml).plus_minus(0.20)
stock_conc = vol_flasks_TC[10] * stock_conc / vol_flasks_TC[100]
density_w = 1 * ureg.g / ureg.ml
stock_conc = stock_conc / (density_w + stock_conc)
stock_conc.ito(ureg.ppm)

stds_conc = np.array((), dtype=object)
reader = Read(DAT / 'standards_concentrations.csv')
for row in reader:
    total_dilution = 1
    for i in range(1, len(row), 2):
        stock, total = int(row[i]), int(row[i+1])
        transfers = min_transfer(stock, list(pipettes_TD.keys()))
        total_transfer = (0 * ureg.ml).plus_minus(0)
        for t in transfers:
            total_transfer += 0 if t == 0 else pipettes_TD[t]
        dilution = total_transfer / vol_flasks_TC[total]
        total_dilution *= dilution
    conc = total_dilution * stock_conc
    stds_conc = np.append(stds_conc, conc)
stds_conc.sort()

stds_abs_raw = np.empty((0, 3))
reader = Read(DAT / 'standards_absorbances.csv')
for row in reader:
    absorbs = []
    for i in range(1, len(row), 2):
        abs_raw, baseln = float(row[i]), float(row[i+1])
        abs = Q_((abs_raw - baseln) * ureg.dimensionless).plus_minus(0.005)
        absorbs.append(abs)
    stds_abs_raw = np.vstack([stds_abs_raw, absorbs])
stds_abs_raw = stds_abs_raw[np.argsort(np.mean(stds_abs_raw, axis=1))]
blank = np.mean(stds_abs_raw[0])
stds_abs_raw = np.array([[a - blank.magnitude.nominal_value for a in trial] for trial in stds_abs_raw])

pass