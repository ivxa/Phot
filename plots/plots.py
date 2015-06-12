# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import itertools
import sys
param = {}
execfile(sys.argv[1])


def auto_crop_img(filename):
    """Call epstools from bash to autocrop image"""
    import subprocess
    import os

    try:
        cwd, img_name = os.path.split(filename)

        bash_cmd = 'epstool --copy --bbox %s %s' % (img_name, 'tmp_'+img_name)
        process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE, cwd=cwd)

        process.wait()
        bash_cmd2 = 'mv %s %s' % ('tmp_'+img_name, img_name)
        process2 = subprocess.Popen(bash_cmd2.split(), stdout=subprocess.PIPE, cwd=cwd)
    except:
        raise RuntimeError('Unable to tight layout. Increase pad_inches?')


def load_data(i):
    mjd_list, mag_list, std_list = np.loadtxt(i + 'data/MJD_MAG_ERR-{}-all_frames.dat'
                                              .format(param['field_name']), unpack=True, delimiter=' ')
    mjd, nightly_avg_mag, nightly_std_mag = np.loadtxt(i + 'data/MJD_MAG_ERR-{}-nightly_average.dat'
                                                       .format(param['field_name']), unpack=True, delimiter=' ')
    return mjd_list, mag_list, std_list, mjd, nightly_avg_mag, nightly_std_mag


def plot_mjd(x, y, yerr, o):
    ymin = min(np.asarray(y)-np.asarray(yerr))
    ymax = max(np.asarray(y)+np.asarray(yerr))

    plt.rcdefaults()
    f, ax = plt.subplots(1, 1)
    ax.errorbar(x, y, yerr=yerr, fmt='k.', markersize=8, elinewidth=1.0, capsize=0)
    ax.set_xlabel(r'MJD')
    ax.set_ylabel(r'$m$ (mag)')
    ax.set_xlim((min(x)-6, max(x)+6))
    ax.set_ylim((ymin*(1-0.001), ymax*(1+0.001)))
    ax.set_ylim(ax.get_ylim()[::-1])
    # from matplotlib.ticker import MultipleLocator
    # ax[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    f.savefig(o, bbox_inches='tight', pad_inches=0.05)
    plt.close(f)
    # auto_crop_img(o)


def compute_orbital_phase(mjd, mag, merr):
    jd = mjd+2400000.5
    jd0 = param['JD0']
    p = param['period']
    phase = np.asarray([(j-jd0)/p-np.int((j-jd0)/p) for j in jd])
    return np.append(phase, phase+1), np.append(mag, mag), np.append(merr, merr)


def plot_phase(i, o):
    x, y, yerr = i
    ymin = min(np.asarray(y)-np.asarray(yerr))
    ymax = max(np.asarray(y)+np.asarray(yerr))

    plt.rcdefaults()
    f, ax = plt.subplots(1, 1)
    ax.errorbar(x, y, yerr=yerr, fmt='k.', markersize=8, elinewidth=1.0, capsize=0)
    ax.set_xlabel(r'MJD')
    ax.set_ylabel(r'$m$ (mag)')
    ax.set_xlim(0, 2)
    ax.set_ylim((ymin*(1-0.001), ymax*(1+0.001)))
    ax.set_ylim(ax.get_ylim()[::-1])
    # from matplotlib.ticker import MultipleLocator
    # ax[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    f.savefig(o, bbox_inches='tight', pad_inches=0.05)
    plt.close(f)
    # auto_crop_img(o)


def arrange_lists_in_cycles(x_list, phase, cycle_list):
    return [[x for (x, p) in zip(x_list, phase)
             if cycle_list[i] < p < cycle_list[i+1]] for i in xrange(len(cycle_list)-1)]


def make_cycles(mjd, mag, merr, testing=0):
    jd = mjd+2400000.5
    jd0 = param['JD0_cycle']
    p = param['period']
    phase = np.asarray([(j-jd0)/p for j in jd])
    cycle_list = sorted(set([np.int(p) for p in phase]))
    cycle_list.extend([cycle_list[-1]+1])
    if testing == 1:
        print 'Cycle list: {}'.format(cycle_list)

    mjd_cycles = arrange_lists_in_cycles(mjd, phase, cycle_list)
    mag_cycles = arrange_lists_in_cycles(mag, phase, cycle_list)
    merr_cycles = arrange_lists_in_cycles(merr, phase, cycle_list)
    return mjd_cycles, mag_cycles, merr_cycles


def plot_mjd_cycles(x_cyc, y_cyc, yerr_cyc, o):
    xmin = min([min(x) for x in x_cyc])
    xmax = max([max(x) for x in x_cyc])
    ymin = min([min(np.asarray(y)-np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])
    ymax = max([max(np.asarray(y)+np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])

    plt.rcdefaults()
    f, ax = plt.subplots(1, 1)
    colormap = eval('plt.cm.'+param['colormap_cycles'])
    col_range = param['colormap_cycles_range']
    ax.set_color_cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
    marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
    for (x, y, yerr) in zip(x_cyc, y_cyc, yerr_cyc):
        ax.errorbar(x, y, yerr=yerr, fmt='.', marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                    markeredgewidth=0.00)
    ax.set_xlabel(r'MJD')
    ax.set_ylabel(r'$m$ (mag)')
    ax.set_xlim(xmin-6, xmax+6)
    ax.set_ylim(ymin*(1-0.001), ymax*(1+0.001))
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    f.savefig(o, bbox_inches='tight', pad_inches=0.05)
    plt.close(f)
    # auto_crop_img(o)


def plot_phase_cycles(x_cyc, y_cyc, yerr_cyc, o):
    ymin = min([min(np.asarray(y)-np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])
    ymax = max([max(np.asarray(y)+np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])

    plt.rcdefaults()

    f, ax = plt.subplots(1, 1)
    colormap = eval('plt.cm.'+param['colormap_cycles'])
    col_range = param['colormap_cycles_range']
    ax.set_color_cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
    marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
    for (x, y, yerr) in zip(x_cyc, y_cyc, yerr_cyc):
        xp, yp, yerrp = compute_orbital_phase(np.asarray(x), np.asarray(y), np.asarray(yerr))
        ax.errorbar(xp, yp, yerr=yerrp, fmt='.', marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                    markeredgewidth=0.00)
    ax.set_xlabel(r'PHA')
    ax.set_ylabel(r'$m$ (mag)')
    ax.set_xlim(0, 2)
    ax.set_ylim(ymin*(1-0.001), ymax*(1+0.001))
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    f.savefig(o, bbox_inches='tight', pad_inches=0.05)
    plt.close(f)
    # auto_crop_img(o)


def make_plots():
    mjd_list, mag_list, std_list, mjd, nightly_avg_mag, nightly_std_mag = load_data(param['output_path'])
    if param['disable_plots_error_bars'] == 1:
        nightly_std_mag *= 0.
        std_list *= 0.

    # Nightly light curves
    # plot_nightly_lc(mjd_list, mag_list, std_list, index_sets, param['output_path']+'nightly_LC/')

    # Multi night light curves
    plot_mjd(mjd_list, mag_list, std_list, param['output_path']
             + 'multi_night_LC/MJD-{}-target-all_frames.eps'.format(param['field_name']))
    plot_mjd(mjd, nightly_avg_mag, nightly_std_mag, param['output_path']
             + 'multi_night_LC/MJD-{}-target-nightly_average.eps'.format(param['field_name']))

    # Multi night phased light curves
    plot_phase(compute_orbital_phase(mjd_list, mag_list, std_list), param['output_path']
               + 'multi_night_LC/PHA-{}-target-all_frames.eps'.format(param['field_name']))
    plot_phase(compute_orbital_phase(mjd, nightly_avg_mag, nightly_std_mag), param['output_path']
               + 'multi_night_LC/PHA-{}-target-nightly_average.eps'.format(param['field_name']))

    # Multi night cycle coloured light curve
    if param['disable_plots_cycles'] == 0:
        mjd_cyc, mag_cyc, merr_cyc = make_cycles(mjd, nightly_avg_mag, nightly_std_mag)
        plot_mjd_cycles(mjd_cyc, mag_cyc, merr_cyc, param['output_path']
                        + 'multi_night_LC/MJD-{}-target-nightly_average_cycles.eps'.format(param['field_name']))

        plot_phase_cycles(mjd_cyc, mag_cyc, merr_cyc, param['output_path']
                          + 'multi_night_LC/PHA-{}-target-nightly_average_cycles.eps'.format(param['field_name']))


if __name__ == '__main__':
    # Testing / plotting from data
    make_plots()
    print('DONE')


