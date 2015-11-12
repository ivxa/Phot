# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import matplotlib # Needed due to a bug related with Ureka
import os
import sys
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcdefaults()


def read_data(dir_base):
    filters = ['B', 'V', 'R', 'I']

    JD = {}
    MJD = {}
    MAG = {}
    ERR = {}
    for f in filters:
        dir_in = os.path.join(dir_base, f, 'data/MJD_MAG_ERR-{}-nightly_average.dat'.format(f))
        MJD[f], MAG[f], ERR[f] = np.loadtxt(dir_in, unpack=True, delimiter=' ')
        JD[f] = np.floor(MJD[f]+2400000.5) # JD starts at 12:00, use the same nights

    for f in filters:
        f1, f2, f3 = [v for v in filters if v != f]
        ix = [ix for (ix, v) in enumerate(JD[f]) if (v in JD[f1]) and (v in JD[f2]) and (v in JD[f3])]
        MJD[f] = MJD[f][ix]
        MAG[f] = MAG[f][ix]
        ERR[f] = ERR[f][ix]

    for i in range(1, len(filters)):
        assert len(MJD[filters[0]]) == len(MJD[filters[i]])
        assert abs(MJD[filters[0]]-MJD[filters[i]]).max() < 2.0/24.
    return MJD, MAG, ERR


def make_dir(dir_out):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)


def make_color_dict(F1, F2, MJD, MAG, ERR):
    col = {}
    col['COL'] = MAG[F1]-MAG[F2]
    col['COLe'] = np.sqrt(ERR[F1]**2.+ERR[F2]**2.)
    col['MJD'] = (MJD[F1]+MJD[F2])/2.
    col['MJDe'] = 1/np.sqrt(2)*np.sqrt((MJD[F1]-MJD[F2])**2.)
    return col


def plot_single_color(col, xlab, dir_out, fname):
    f = plt.figure()
    f.clf()
    ax = f.gca()
    ax.errorbar(col['MJD'], col['COL'], yerr=col['COLe'], xerr=col['MJDe'], fmt='k.', markersize=8, elinewidth=1.0, capsize=0)
    ax.set_ylabel(xlab)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.set_xlabel('MJD')
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    f.savefig(os.path.join(dir_out, fname), bbox_inches='tight', pad_inches=0.05)


def plot_filters(MJD, MAG, ERR, dir_out):
    filters = ['B', 'V', 'R', 'I']
    fig, ax = plt.subplots(4, sharex=True, figsize=(3,8))
    ax[3].set_xlabel('MJD')
    for (i, f) in enumerate(filters):
        ax[i].yaxis.set_major_locator(MultipleLocator(0.02))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i].yaxis.set_minor_locator(MultipleLocator(0.01))
        ax[i].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i].errorbar(MJD[f], MAG[f], yerr=ERR[f], fmt='k.', markersize=8, elinewidth=1.0, capsize=0)
        ax[i].set_ylabel('{} [mag]'.format(f))
        ax[i].set_ylim(ax[i].get_ylim()[::-1])
    fig.savefig(os.path.join(dir_out, 'filters.eps'), bbox_inches='tight', pad_inches=0.05)


def main():
    dir_base = '/home/gamma/garrofa/xparedes/Dropbox/photometry_tjo/mwc656'
    dir_out = os.path.join(dir_base, 'colors')
    make_dir(dir_out)
    MJD, MAG, ERR = read_data(dir_base)
    f = [('B', 'V'), ('B', 'R'), ('B', 'I'), ('V', 'R'), ('V', 'I'), ('R', 'I')]

    plot_filters(MJD, MAG, ERR, dir_out)

    fig, ax = plt.subplots(3, 2, sharex=True)
    fig.subplots_adjust(wspace=0.3)
    ax[2, 0].set_xlabel('MJD')
    ax[2, 1].set_xlabel('MJD')
    for k in range(len(f)):
        if k < 3:
            i = k
            j = 0
        else:
            i = k-3
            j = 1
        ax[i, j].yaxis.set_major_locator(MultipleLocator(0.01))
        ax[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[i, j].xaxis.set_minor_locator(MultipleLocator(5))

        col = make_color_dict(f[k][0], f[k][1], MJD, MAG, ERR)
        col_name = '{}-{}'.format(f[k][0], f[k][1])
        plot_single_color(col, '{} [mag]'.format(col_name), dir_out, '{}.eps'.format(col_name))

        ax[i, j].errorbar(col['MJD'], col['COL'], yerr=col['COLe'], xerr=col['MJDe'], fmt='k.', markersize=8, elinewidth=1.0, capsize=0)
        ax[i, j].set_ylabel('{} [mag]'.format(col_name))
    fig.savefig(os.path.join(dir_out, 'colors.eps'), bbox_inches='tight', pad_inches=0.05)


if __name__ == '__main__':
    try:
        main()
        print '\nNormal termination.'
    except Exception, err:
        sys.stderr.write('\nERROR: %s\n' % str(err))
