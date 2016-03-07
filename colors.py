# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import matplotlib # Needed due to a bug related with Ureka
import os
import sys
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
import itertools
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcdefaults()
execfile(sys.argv[1])

# filters = ['B', 'V', 'R', 'I']
# filter_colors = ['blue', 'green', 'red', 'orange']
# filter_pairs = [('B', 'V'), ('B', 'R'), ('B', 'I'), ('V', 'R'), ('V', 'I'), ('R', 'I')]
filters = ['V', 'R', 'I']
filter_colors = ['green', 'red', 'orange']
filter_pairs = [('V', 'R'), ('V', 'I'), ('R', 'I')]
Npairs = 3 # 6

def read_data(dir_base, suffix, dm):
    # filters = ['B', 'V', 'R', 'I']

    JD = {}
    MJD = {}
    MAG = {}
    ERR = {}
    for f in filters:
        dir_in = os.path.join(dir_base, f, 'data/S0_{}_MJD_MAG_ERR-{}-nightly_average.dat'.format(suffix, f))
        MJD[f], MAG[f], ERR[f], _nframes = np.loadtxt(dir_in, unpack=True, delimiter=' ')
        JD[f] = np.floor(MJD[f]+2400000.5) # JD starts at 12:00, use the same nights

    for f in filters:
        if len(filters) == 4:
            f1, f2, f3 = [v for v in filters if v != f]
            ix = [ix for (ix, v) in enumerate(JD[f]) if (v in JD[f1]) and (v in JD[f2]) and (v in JD[f3])]
        elif len(filters) == 3:
            f1, f2 = [v for v in filters if v != f]
            ix = [ix for (ix, v) in enumerate(JD[f]) if (v in JD[f1]) and (v in JD[f2])]
        else:
            print 'Not implemented'
            raise 'ERROR'
        MJD[f] = MJD[f][ix]
        MAG[f] = MAG[f][ix]
        ERR[f] = ERR[f][ix]

    for f in filters:
        MAG[f] -= dm

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


def plot_filters(MJD, MAG, ERR, label1, MJDc, MAGc, ERRc, label2, dir_out, pdf):
    # filters = ['B', 'V', 'R', 'I']
    # filter_colors = ['blue', 'green', 'red', 'orange']

    dy = 0.
    for f in filters:
        mmax = max(MAG[f])
        mmin = min(MAG[f])
        dy_aux = mmax-mmin+2*0.004
        if dy<dy_aux:
            dy=dy_aux

    plt.close('all')
    fig, ax = plt.subplots(4, 3, sharex=True, figsize=(11,9))
    ax[3, 0].set_xlabel('MJD')
    ax[3, 1].set_xlabel('MJD')
    ax[0, 0].set_title(label1)
    ax[0, 1].set_title(label2)
    for (i, f) in enumerate(filters):
        ax[i, 0].yaxis.set_major_locator(MultipleLocator(0.02))
        ax[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i, 0].yaxis.set_minor_locator(MultipleLocator(0.01))
        ax[i, 0].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i, 0].errorbar(MJD[f], MAG[f], yerr=ERR[f], fmt='k.', markersize=8, elinewidth=1.0, capsize=0)
        ax[i, 0].set_ylabel('{} [mag]'.format(f))
        ax[i, 0].set_ylim(ax[i, 0].get_ylim()[::-1])

        ax[i, 1].yaxis.set_major_locator(MultipleLocator(0.02))
        ax[i, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i, 1].yaxis.set_minor_locator(MultipleLocator(0.01))
        ax[i, 1].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i, 1].errorbar(MJDc[f], MAGc[f], yerr=ERRc[f], fmt='g.', markersize=8, elinewidth=1.0, capsize=0)
        ax[i, 1].set_ylabel('{} [mag]'.format(f))
        ax[i, 1].set_ylim(ax[i, 1].get_ylim()[::-1])

        ax[i, 2].yaxis.set_major_locator(MultipleLocator(0.02))
        ax[i, 2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i, 2].yaxis.set_minor_locator(MultipleLocator(0.01))
        ax[i, 2].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i, 2].errorbar(MJD[f], MAG[f], yerr=ERR[f], fmt='k.', markersize=8, elinewidth=1.0, capsize=0, label=label1)
        ax[i, 2].set_ylabel('{} [mag]'.format(f))
        ax[i, 2].set_ylim(ax[i, 2].get_ylim()[::-1])
        ax[i, 2].errorbar(MJDc[f], MAGc[f]+np.average(MAG[f])-np.average(MAGc[f]), yerr=ERRc[f], fmt='g.', markersize=8, elinewidth=1.0, capsize=0, label=label2)

    ax[3, 0].set_xticklabels(ax[3, 0].get_xticks(), rotation=70, ha='right')
    ax[3, 1].set_xticklabels(ax[3, 0].get_xticks(), rotation=70, ha='right')
    ax[3, 2].set_xticklabels(ax[3, 0].get_xticks(), rotation=70, ha='right')

    # fig.savefig(os.path.join(dir_out, 'filters.eps'), bbox_inches='tight', pad_inches=0.05)
    plt.tight_layout()
    pdf.savefig()
    plt.close('all')


    fig_width_pt = 1*512.1496              # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27              # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0     # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt # width in inches
    fig_height = fig_width*golden_mean     # height in inches
    fig_size =  [fig_width,fig_height*1.2]
    params = {'backend': 'ps',
              'font.family':'serif',
              'axes.labelsize': 18,
              'font.size': 18,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'text.usetex': True,
              'text.latex.preamble':[r'\usepackage{txfonts}'],
              'ps.usedistiller': 'xpdf',
              'figure.figsize': fig_size}
    fs = 18
    for i, f in enumerate(filters):
        plt.rcdefaults()
        plt.rcParams.update(params)
        plt.close('all')
        fig, ax = plt.subplots()
        ax.set_xlabel('MJD')
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.errorbar(MJD[f], MAG[f], yerr=ERR[f], color=filter_colors[i], fmt='.', markersize=8, elinewidth=1.0, capsize=0, label=label1)
        ax.set_ylabel(r'$I_{\rm C}$ [mag]')
        ymin = np.average(MAG[f])-dy/2.
        ymax = np.average(MAG[f])+dy/2.
        ax.set_ylim((ymin, ymax))
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.errorbar(MJDc[f], MAGc[f]+np.average(MAG[f])-np.average(MAGc[f]), yerr=ERRc[f], color='black', fmt='.', markersize=8, elinewidth=1.0, capsize=0, label=label2)
        plt.tight_layout()
        # fig.savefig(os.path.join(dir_out, 'filter_{}.eps'.format(filter_colors[i])), bbox_inches='tight', pad_inches=0.05)
        fig.savefig(os.path.join(dir_out, 'filter_{}.jpg'.format(f)), bbox_inches='tight', pad_inches=0.05)
        plt.close('all')


def plot_colors(f, MJD, MAG, ERR, MJDc, MAGc, ERRc, dir_out, pdf):

    fig, ax = plt.subplots(6, 3, sharex=True, figsize=(12,15))
    fig.subplots_adjust(wspace=0.3)

    ax[0, -1].set_xlabel('MJD')
    ax[0, -2].set_xlabel('MJD')
    ax[0, -3].set_xlabel('MJD')
    for i in range(Npairs):
        q = 0
        ax[i, q].yaxis.set_major_locator(MultipleLocator(0.01))
        ax[i, q].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i, q].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[i, q].xaxis.set_minor_locator(MultipleLocator(5))
        col = make_color_dict(f[i][0], f[i][1], MJD, MAG, ERR)
        col_name = '{}-{}'.format(f[i][0], f[i][1])
        ax[i, q].errorbar(col['MJD'], col['COL'], yerr=col['COLe'], xerr=col['MJDe'], fmt='.', color='black', markersize=8, elinewidth=1.0, capsize=0)
        ax[i, q].set_ylabel('{} [mag]'.format(col_name))

        q = 1
        ax[i, q].yaxis.set_major_locator(MultipleLocator(0.01))
        ax[i, q].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i, q].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[i, q].xaxis.set_minor_locator(MultipleLocator(5))
        colc = make_color_dict(f[i][0], f[i][1], MJDc, MAGc, ERRc)
        col_name = '{}-{}'.format(f[i][0], f[i][1])
        ax[i, q].errorbar(colc['MJD'], colc['COL'], yerr=colc['COLe'], xerr=colc['MJDe'], fmt='.', color='green', markersize=8, elinewidth=1.0, capsize=0)
        ax[i, q].set_ylabel('{} [mag]'.format(col_name))

        q = 2
        ax[i, q].yaxis.set_major_locator(MultipleLocator(0.01))
        ax[i, q].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[i, q].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[i, q].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i, q].errorbar(col['MJD'], col['COL'], yerr=col['COLe'], xerr=col['MJDe'], fmt='.', color='black', markersize=8, elinewidth=1.0, capsize=0)
        ax[i, q].errorbar(colc['MJD'], colc['COL']-np.average(colc['COL'])+np.average(col['COL']), yerr=colc['COLe'], xerr=colc['MJDe'], fmt='.', color='green', markersize=8, elinewidth=1.0, capsize=0)
        ax[i, q].set_ylabel('{} [mag]'.format(col_name))

    ax[-1, 0].set_xticklabels(ax[-1, 0].get_xticks(), rotation=70, ha='right')
    ax[-1, 1].set_xticklabels(ax[-1, 1].get_xticks(), rotation=70, ha='right')
    ax[-1, 2].set_xticklabels(ax[-1, 2].get_xticks(), rotation=70, ha='right')

    # fig.savefig(os.path.join(dir_out, 'colors.eps'), bbox_inches='tight', pad_inches=0.05)
    plt.tight_layout()
    fig.savefig(os.path.join(dir_out, 'colors.jpg'), bbox_inches='tight', pad_inches=0.05)
    # plt.suptitle(title)
    plt.tight_layout()
    pdf.savefig()
    plt.close('all')


def plot_colors_together(f, MJD, MAG, ERR, label1, MJDc, MAGc, ERRc, label2, dir_out, pdf):
    plt.close('all')
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
        # plot_single_color(col, '{} [mag]'.format(col_name), dir_out, '{}.eps'.format(col_name))

        ax[i, j].errorbar(col['MJD'], col['COL'], yerr=col['COLe'], xerr=col['MJDe'], fmt='b.', markersize=8, elinewidth=1.0, capsize=0, label=label1)
        ax[i, j].set_ylabel('{} [mag]'.format(col_name))

    for k in range(len(f)):
        if k < 3:
            i = k
            j = 0
        else:
            i = k-3
            j = 1
        col = make_color_dict(f[k][0], f[k][1], MJDc, MAGc, ERRc)
        ax[i, j].errorbar(col['MJD'], col['COL'], yerr=col['COLe'], xerr=col['MJDe'], fmt='g.', markersize=8, elinewidth=1.0, capsize=0, label=label1)
        ax[i, j].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

    # fig.savefig(os.path.join(dir_out, 'colors.eps'), bbox_inches='tight', pad_inches=0.05)
    plt.tight_layout()
    pdf.savefig()
    plt.close('all')


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


def plot_mjd_cycles(ff, MJD, MAG, ERR,  MJDc, MAGc, ERRc, dir_out, pdf, plot_errors=True):

    for iii in range(Npairs):
        col = make_color_dict(ff[iii][0], ff[iii][1], MJD, MAG, ERR)
        col_name = '{}-{}'.format(ff[iii][0], ff[iii][1])
        x_cyc, y_cyc, yerr_cyc = make_cycles(col['MJD'], col['COL'], col['COLe'])
        col = make_color_dict(ff[iii][0], ff[iii][1], MJDc, MAGc, ERRc)
        x_cyc1, y_cyc1, yerr_cyc1 = make_cycles(col['MJD'], col['COL'], col['COLe'])

        def tick_function(tick_list):
            return [Time(t, format='mjd', scale='utc' ).datetime.date() for t in tick_list]

        xmin = min([min(x) for x in x_cyc])
        xmax = max([max(x) for x in x_cyc])
        ymin = min([min(np.asarray(y)-np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])
        ymax = max([max(np.asarray(y)+np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])

        plt.close('all')
        f, ax = plt.subplots(1, 2, figsize=(2*7,1*5.5))

        colormap = eval('plt.cm.'+param['colormap_cycles'])
        col_range = param['colormap_cycles_range']
        ax[0].set_color_cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
        marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
        for (x, y, yerr) in zip(x_cyc, y_cyc, yerr_cyc):
            if not plot_errors:
                yerr=np.array(yerr)*0.
            ax[0].errorbar(x, y, yerr=yerr, fmt='.', alpha=0.8, marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                           markeredgewidth=0.00)
        ax[0].set_xlabel(r'MJD')
        ax[0].set_ylabel(r'{} [mag]'.format(col_name))
        ax[0].set_xlim(xmin-6, xmax+6)
        ax[0].set_ylim(ymin*(1-0.001), ymax*(1+0.001))
        # ax[0].set_ylim(ax[0].get_ylim()[::-1])
        ax[0].yaxis.set_minor_locator(AutoMinorLocator())
        ax[0].grid()
        ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        axtop = ax[0].twiny()
        axtop.set_xticks(ax[0].get_xticks())
        axtop.set_xbound(ax[0].get_xbound())
        axtop.set_xticklabels(tick_function(ax[0].get_xticks()), rotation=70, ha='left')

        ax[1].set_color_cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
        marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
        k = True
        for (x, y, yerr) in zip(x_cyc1, y_cyc1, yerr_cyc1):
            if not plot_errors:
                yerr=np.array(yerr)*0.
            if k:
                ax[1].errorbar(x, y, yerr=yerr, fmt='.', alpha=0.8, marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                               markeredgewidth=0.00, label='Comp. star 1')
                k = False
            else:
                ax[1].errorbar(x, y, yerr=yerr, fmt='.', alpha=0.8, marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                               markeredgewidth=0.00)
        ax[1].set_xlabel(r'MJD')
        ax[1].set_ylabel(r'{} [mag]'.format(col_name))
        ax[1].set_xlim(xmin-6, xmax+6)
        # ax[1].set_ylim(ax[1].get_ylim()[::-1])
        ax[1].yaxis.set_minor_locator(AutoMinorLocator())
        ax[1].grid()
        ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)
        axtop = ax[1].twiny()
        axtop.set_xticks(ax[1].get_xticks())
        axtop.set_xbound(ax[1].get_xbound())
        axtop.set_xticklabels(tick_function(ax[1].get_xticks()), rotation=70, ha='left')

        plt.tight_layout()
        pdf.savefig()
        plt.close('all')
    # f.savefig(o, bbox_inches='tight', pad_inches=0.05)
    # auto_crop_img(o)


def plot_phase_cycles(ff, MJD, MAG, ERR,  MJDc, MAGc, ERRc, dir_out, pdf, plot_errors=True):

    for iii in range(Npairs):
        col = make_color_dict(ff[iii][0], ff[iii][1], MJD, MAG, ERR)
        col_name = '{}-{}'.format(ff[iii][0], ff[iii][1])
        x_cyc, y_cyc, yerr_cyc = make_cycles(col['MJD'], col['COL'], col['COLe'])
        col = make_color_dict(ff[iii][0], ff[iii][1], MJDc, MAGc, ERRc)
        x_cyc1, y_cyc1, yerr_cyc1 = make_cycles(col['MJD'], col['COL'], col['COLe'])

        ymin = min([min(np.asarray(y)-np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])
        ymax = max([max(np.asarray(y)+np.asarray(dy)) for (y, dy) in zip(y_cyc, yerr_cyc)])

        # ALL CYCLES TOGETHER
        plt.close('all')
        f, ax = plt.subplots(1, 2, figsize=(2*7,1*5))

        colormap = eval('plt.cm.'+param['colormap_cycles'])
        col_range = param['colormap_cycles_range']

        ax[0].set_color_cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
        marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
        for (x, y, yerr) in zip(x_cyc, y_cyc, yerr_cyc):
            xp, yp, yerrp = compute_orbital_phase(np.asarray(x), np.asarray(y), np.asarray(yerr))
            if not plot_errors:
                yerrp=np.array(yerrp)*0.
            ax[0].errorbar(xp, yp, yerr=yerrp, fmt='.', alpha=0.8, marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                           markeredgewidth=0.00)
        ax[0].set_xlabel(r'PHASE')
        ax[0].set_ylabel(r'{} [mag]'.format(col_name))
        ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator())
        ax[0].grid()
        ax[0].set_xlim(0, 2)
        ax[0].set_ylim(ymin*(1-0.001), ymax*(1+0.001))
        ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))

        ax[1].set_color_cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
        marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
        k = True
        for (x, y, yerr) in zip(x_cyc1, y_cyc1, yerr_cyc1):
            xp, yp, yerrp = compute_orbital_phase(np.asarray(x), np.asarray(y), np.asarray(yerr))
            if not plot_errors:
                yerrp=np.array(yerrp)*0.
            if k:
                ax[1].errorbar(xp, yp, yerr=yerrp, fmt='.',  alpha=0.8, marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                               markeredgewidth=0.00, label='Comp. star 1')
                k = False
            else:
                ax[1].errorbar(xp, yp, yerr=yerrp, fmt='.',  alpha=0.8, marker=marker.next(), markersize=8, elinewidth=1.0, capsize=0,
                               markeredgewidth=0.00)
        ax[1].set_xlabel(r'PHASE')
        ax[1].set_ylabel(r'{} [mag]'.format(col_name))
        ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator())
        ax[1].grid()
        ax[1].set_xlim(0, 2)
        ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

        plt.tight_layout()
        pdf.savefig()
        plt.close('all')


def compute_orbital_phase(mjd, mag, merr):
    jd = mjd+2400000.5
    jd0 = param['JD0']
    p = param['period']
    phase = np.asarray([(j-jd0)/p-np.int((j-jd0)/p) for j in jd])
    return np.append(phase, phase+1), np.append(mag, mag), np.append(merr, merr)


def zero_mag(dir_base, suffix,):
    dir_in = os.path.join(dir_base, 'V', 'data/S0_{}_MJD_MAG_ERR-{}-nightly_average.dat'.format(suffix, 'V'))
    MJD, MAG, ERR, _nframes = np.loadtxt(dir_in, unpack=True, delimiter=' ')
    dm = np.average(MAG)-param['zero_magnitude']
    return dm


def main():
    dir_base = param['ref_star_file_out']
    dir_out = os.path.join(dir_base, 'colors')
    make_dir(dir_out)
    dm = zero_mag(dir_base, 'target')
    MJD, MAG, ERR = read_data(dir_base, 'target', dm)
    MJDc, MAGc, ERRc = read_data(dir_base, 'compa1', dm)
    # filter_pairs = [('B', 'V'), ('B', 'R'), ('B', 'I'), ('V', 'R'), ('V', 'I'), ('R', 'I')]

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(os.path.join(dir_out, 'colors.pdf')) as pdf:
        plot_filters(MJD, MAG, ERR, 'MWC 656', MJDc, MAGc, ERRc, 'Comparison star', dir_out, pdf)
        plot_colors(filter_pairs, MJD, MAG, ERR, MJDc, MAGc, ERRc, dir_out, pdf)
        plot_mjd_cycles(filter_pairs, MJD, MAG, ERR, MJDc, MAGc, ERRc, dir_out, pdf)
        plot_phase_cycles(filter_pairs, MJD, MAG, ERR, MJDc, MAGc, ERRc, dir_out, pdf)
        # plot_colors(filter_pairs, MJDc, MAGc, ERRc, 'Comparison star', dir_out, pdf)
        # plot_colors_together(filter_pairs, MJD, MAG, ERR, 'MWC 656', MJDc, MAGc, ERRc, 'Comparison star', dir_out, pdf)



if __name__ == '__main__':
    try:
        main()
        print '\nNormal termination.'
    except Exception, err:
        sys.stderr.write('\nERROR: %s\n' % str(err))
