# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def compute_nightly_average(cat_mag):
    return np.asarray([[np.average(mag[:, i]) for i in xrange(len(mag[0, :]))] for mag in cat_mag])


def compute_statistics(cat_mag):
    avg_mag = [np.average(cat_mag[:, k]) for k in xrange(len(cat_mag[0, :]))]
    std_mag = [np.std(cat_mag[:, k]) for k in xrange(len(cat_mag[0, :]))]
    return np.array(avg_mag), np.array(std_mag)


def plot(x, y, o):
    plt.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '.')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\overline{m}$ (mag)')
    ax.set_ylabel(r'$\sigma_{m}$ (mag)')
    ax.set_xlim((min(x)*(1-0.05), max(x)*(1+0.05)))
    ax.set_ylim((min(y)*(1-0.05), max(y)*(1+0.05)))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.table(cellText=[['N', r'$\overline{{\sigma}}$'],
                        [1, '{:.3f}'.format(y[0])],
                        [5, '{:.3f}'.format(np.average(y[0:5]))],
                        [10, '{:.3f}'.format(np.average(y[0:10]))],
                        [25, '{:.3f}'.format(np.average(y[0:25]))],
                        [50, '{:.3f}'.format(np.average(y[0:50]))],
                        [100, '{:.3f}'.format(np.average(y[0:100]))]],
              colWidths=[0.1, 0.1],
              loc='center left')
    fig.savefig(o, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def perform_test(cat_mag, o, ind=-1, ind_comp=-1, ind_ref=-1):
    cat_mag_avg = compute_nightly_average(cat_mag)
    avg_mag, std_mag = compute_statistics(cat_mag_avg)
    # x = avg_mag[std_mag.argsort()]
    # y = sorted(std_mag)
    plot(avg_mag, std_mag, o)
    if ind != -1:
        flag = np.zeros(len(avg_mag))
        flag[ind] = 1
        flag[ind_comp] = 2
        flag[ind_ref] = 3
        np.savetxt(o[:-3]+'dat', np.transpose((avg_mag, std_mag, flag)), delimiter=' ')


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')
