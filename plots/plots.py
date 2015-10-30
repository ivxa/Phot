# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.time import Time
import itertools
import sys
import os
import matplotlib
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


def load_data(i, fname):
    mjd_list, mag_list, std_list = np.loadtxt(i + 'data/S{}_MJD_MAG_ERR-{}-all_frames.dat'
                                              .format(fname, param['field_name']), unpack=True, delimiter=' ')
    mjd, nightly_avg_mag, nightly_std_mag = np.loadtxt(i + 'data/S{}_MJD_MAG_ERR-{}-nightly_average.dat'
                                                       .format(fname, param['field_name']), unpack=True, delimiter=' ')
    return mjd_list, mag_list, std_list, mjd, nightly_avg_mag, nightly_std_mag


def plot1(o, fn, tn, nstars):
    fname = os.path.join(o, 'multi_night_LC/MJD-{}-average.pdf'.format(fn))
    fig, ax = plt.subplots(3, nstars, sharex=True, figsize=(nstars*3.5, 3*2.5))
    fig.subplots_adjust(wspace=0.3)

    for j in range(nstars):
        mjd_list_t, mag_list_t, std_list_t, mjd_t, nightly_avg_mag_t, nightly_std_mag_t = load_data(o, '{}_target'.format(str(j)))
        mjd_list_c, mag_list_c, std_list_c, mjd_c, nightly_avg_mag_c, nightly_std_mag_c = load_data(o, '{}_compar'.format(str(j)))

        target_mean = 'Mean mag: {:.3f} mag'.format(np.average(nightly_avg_mag_t))
        target_std = 'Mean std: {:.3f} mag'.format(np.average(nightly_std_mag_t))

        compar_mean = 'Mean mag: {:.3f} mag'.format(np.average(nightly_avg_mag_c))
        compar_std = 'Mean std: {:.3f} mag'.format(np.average(nightly_std_mag_c))

        info = np.loadtxt(param['output_path']+'data/S{}_info'.format(j))
        d = float(info[0])
        ra = float(info[1])
        dec = float(info[2])
        w = float(info[3])
        compar_coord = 'RA: {:.3f} deg, DEC: {:.3f} deg'.format(ra,dec)
        compar_dist = 'Distance with respect the target: {:.3f} arcmin'.format(d)
        compar_weight = 'Correction weight when included: {:.3}'.format(w)

        t = Time(mjd_c.max(), format='mjd', scale='utc' )
        last_obs = 'Last observation: {}'.format(t.datetime)

        text_info = '- ' +  tn + r':\\' + target_mean + r'\\' + target_std + r'\\\\' + r'- Comparison star {}:\\(Corrected using {} comparison stars)\\'.format(j,nstars-1) + compar_mean + r'\\' + compar_std + r'\\' + compar_coord + r'\\' + compar_dist + r'\\' + compar_weight + r'\\\\' + last_obs

        # %ax[0, j].set_title('')
        ax[0, j].errorbar(mjd_t, nightly_avg_mag_t, yerr=nightly_std_mag_t, fmt='b.', markersize=8, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[0, j].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

        ax[1, j].errorbar(mjd_c, nightly_avg_mag_c, yerr=nightly_std_mag_c, fmt='ks', markersize=5, elinewidth=1.0, capsize=0, label='Comparison star {}'.format(j))
        ax[1, j].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

        offset_value = np.average(nightly_avg_mag_c)-np.average(nightly_avg_mag_t)
        ax[2, j].errorbar(mjd_t, nightly_avg_mag_t, yerr=nightly_std_mag_t, fmt='b.', markersize=8, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[2, j].errorbar(mjd_c, nightly_avg_mag_c-offset_value, yerr=nightly_std_mag_c, fmt='ks', markersize=5, elinewidth=1.0, alpha=0.5, capsize=0, label='Comparison star {} (offset)'.format(j))
        ax[2, j].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)
        ax[2, j].annotate(text_info, xy=(1, 0), xycoords='axes fraction', fontsize=11,
                           xytext=(0, -40), textcoords='offset points',
                           ha='right', va='top')

    for i in range(3):
        ax[i, 0].set_ylabel('$m$ [mag]')
    for j in range(nstars):
        ax[2, j].set_xlabel('MJD')
        for i in range(3):
            # ax[i, j].xaxis.set_minor_locator(MultipleLocator(5))
            # ax[i, j].yaxis.set_major_locator(MultipleLocator(0.01))
            # ax[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[i, j].yaxis.set_minor_locator(MultipleLocator(0.005))
            ax[i, j].set_ylim(ax[i, j].get_ylim()[::-1])
            y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            ax[i,j].yaxis.set_major_formatter(y_formatter)
    plt.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.05)
    # auto_crop_img(fname)

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
    #f.show()
    plt.close(f)
    # auto_crop_img(o)


def compute_orbital_phase(mjd, mag, merr):
    jd = mjd+2400000.5
    jd0 = param['JD0']
    p = param['period']
    phase = np.asarray([(j-jd0)/p-np.int((j-jd0)/p) for j in jd])
    return np.append(phase, phase+1), np.append(mag, mag), np.append(merr, merr)


def compute_orbital_phase_mid(mjd, mag, merr):
    jd = mjd+2400000.5
    jd0 = param['JD0']
    p = param['period']
    phase = np.asarray([(j-jd0)/p-np.int((j-jd0)/p) for j in jd])
    return phase, mag, merr


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
    fig_width_pt = 1*512.1496              # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27              # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0     # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt # width in inches
    fig_height = fig_width*golden_mean     # height in inches
    fig_size =  [fig_width,fig_height]
    params = {'backend': 'ps',
              'font.family':'serif',
              'axes.labelsize': 12,
              'font.size': 8,
              'legend.fontsize': 10,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'text.usetex': True,
              'text.latex.preamble':[r'\usepackage{txfonts}'],
              'ps.usedistiller': 'xpdf',
              'figure.figsize': fig_size}
    plt.rcdefaults()
    plt.rcParams.update(params)

    output_path = param['output_path']
    field_name = param['field_name']
    title_name = param['title_name']
    nstars = int(np.loadtxt(output_path+'data/nstars', dtype=int))

    plot1(output_path, field_name, title_name, nstars)


    # mjd_list, mag_list, std_list, mjd, nightly_avg_mag, nightly_std_mag = load_data(output_path)
    # if param['disable_plots_error_bars'] == 1:
    #     nightly_std_mag *= 0.
    #     std_list *= 0.
    #
    # # Nightly light curves
    # # plot_nightly_lc(mjd_list, mag_list, std_list, index_sets, param['output_path']+'nightly_LC/')
    #
    # # Multi night light curves
    # plot_mjd(mjd_list, mag_list, std_list, param['output_path']
    #          + 'multi_night_LC/MJD-{}-target-all_frames.eps'.format(param['field_name']))
    # plot_mjd(mjd, nightly_avg_mag, nightly_std_mag, param['output_path']
    #          + 'multi_night_LC/MJD-{}-target-nightly_average.eps'.format(param['field_name']))
    #
    # # Multi night phased light curves
    # plot_phase(compute_orbital_phase(mjd_list, mag_list, std_list), param['output_path']
    #            + 'multi_night_LC/PHA-{}-target-all_frames.eps'.format(param['field_name']))
    # plot_phase(compute_orbital_phase(mjd, nightly_avg_mag, nightly_std_mag), param['output_path']
    #            + 'multi_night_LC/PHA-{}-target-nightly_average.eps'.format(param['field_name']))
    # np.savetxt(param['output_path']+'data/'+'PHA_MAG_ERR-{}-nightly_average.dat'.format(param['field_name']),
    #            np.transpose(compute_orbital_phase_mid(mjd, nightly_avg_mag, nightly_std_mag)), delimiter=' ')
    #
    #
    # # Multi night cycle coloured light curve
    # if param['disable_plots_cycles'] == 0:
    #     mjd_cyc, mag_cyc, merr_cyc = make_cycles(mjd, nightly_avg_mag, nightly_std_mag)
    #     plot_mjd_cycles(mjd_cyc, mag_cyc, merr_cyc, param['output_path']
    #                     + 'multi_night_LC/MJD-{}-target-nightly_average_cycles.eps'.format(param['field_name']))
    #
    #     plot_phase_cycles(mjd_cyc, mag_cyc, merr_cyc, param['output_path']
    #                       + 'multi_night_LC/PHA-{}-target-nightly_average_cycles.eps'.format(param['field_name']))
    #

if __name__ == '__main__':
    # Testing / plotting from data
    make_plots()
    print('DONE')
