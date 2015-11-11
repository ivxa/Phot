# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator
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
    mjd_list, mag_list, std_list, night_list = np.loadtxt(i + 'data/S{}_MJD_MAG_ERR-{}-all_frames.dat'
                                              .format(fname, param['field_name']), unpack=True, delimiter=' ')
    mjd, nightly_avg_mag, nightly_std_mag, _night_list = np.loadtxt(i + 'data/S{}_MJD_MAG_ERR-{}-nightly_average.dat'
                                                       .format(fname, param['field_name']), unpack=True, delimiter=' ')
    return mjd_list, mag_list, std_list, mjd, nightly_avg_mag, nightly_std_mag, night_list


def nightly_plots(o, suff, fn, tn, nstars, pdf):

    def tick_function(tick_list):
        return ['{:.0f}'.format((Time(t, format='mjd', scale='utc').datetime-Time(tick_list[0], format='mjd', scale='utc').datetime).total_seconds()/60.) for t in tick_list]

    _mjd_list, _mag_list, _std_list, _night_list = np.loadtxt(o + 'data/S0_target_MJD_MAG_ERR-{}-nightly_average.dat'.format(param['field_name']), unpack=True, delimiter=' ')
    mjd_list, mag_list, std_list, night_list = np.loadtxt(o + 'data/S0_target_MJD_MAG_ERR-{}-all_frames.dat'.format(param['field_name']), unpack=True, delimiter=' ')
    mjd_list1, mag_list1, std_list1, night_list1 = np.loadtxt(o + 'data/S0_compa1_MJD_MAG_ERR-{}-all_frames.dat'.format(param['field_name']), unpack=True, delimiter=' ')
    mjd_list2, mag_list2, std_list2, night_list2 = np.loadtxt(o + 'data/S0_compa2_MJD_MAG_ERR-{}-all_frames.dat'.format(param['field_name']), unpack=True, delimiter=' ')
    nnights = int(max(_night_list))

    nights_per_page = 25.
    npages = np.int(np.ceil(nnights/np.float(nights_per_page)))
    for page in range(npages):
        plt.close('all')
        # fig, ax = plt.subplots(nnights, 4, sharex=False, figsize=(4*4.5, nnights*4))
        fig, ax = plt.subplots(int(nights_per_page), 4, sharex=False, figsize=(4*4.5, nights_per_page*4))

        alpha_plots = 0.8
        n0 = -1
        for n in _night_list[int(page)*int(nights_per_page):int(page)*int(nights_per_page)+int(nights_per_page)]:
            n = int(n)-1
            n1 = n+1
            n0 += 1
            ax[n0, 0].errorbar(mjd_list[night_list == n1], mag_list[night_list == n1], yerr=std_list[night_list == n1], fmt='b.', markersize=7, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='{}'.format(tn))
            ax[n0, 0].set_ylabel('$m$ [mag]')
            ax[n0, 0].set_xlabel('MJD')
            # ax[n0, 0].yaxis.set_minor_locator(MultipleLocator(0.005))
            ax[n0, 0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
            ax[n0, 0].set_title('Night {} of {}\nStart: {}\nEnd: {}'.format(n1, nnights, Time(mjd_list[night_list == n1][0], format='mjd', scale='utc' ).datetime, Time(mjd_list[night_list == n1][-1], format='mjd', scale='utc' ).datetime), y=1.2)
            axtop = ax[n0, 0].twiny()
            axtop.set_xticks(ax[n0, 0].get_xticks())
            axtop.set_xbound(ax[n0, 0].get_xbound())
            axtop.set_xticklabels(tick_function(ax[n0, 0].get_xticks()))#, rotation=70, ha='left')
            axtop.set_xlabel('Elapsed time [minutes]')
            ax[n0, 0].set_ylim(ax[n0, 0].get_ylim()[::-1])
            ax[n0, 0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
            ax[n0, 0].grid()
            ax[n0, 0].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

            ax[n0, 1].errorbar(mjd_list1[night_list1 == n1], mag_list1[night_list1 == n1], yerr=std_list1[night_list1 == n1], fmt='g.', markersize=7, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='Comp. star 1')
            ax[n0, 1].set_ylabel('$m$ [mag]')
            ax[n0, 1].set_xlabel('MJD')
            # ax[n0, 1].yaxis.set_minor_locator(MultipleLocator(0.005))
            ax[n0, 1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
            axtop = ax[n0, 1].twiny()
            axtop.set_xticks(ax[n0, 1].get_xticks())
            axtop.set_xbound(ax[n0, 1].get_xbound())
            axtop.set_xticklabels(tick_function(ax[n0, 1].get_xticks()))#, rotation=70, ha='left')
            axtop.set_xlabel('Elapsed time [minutes]')
            ax[n0, 1].set_ylim(ax[n0, 1].get_ylim()[::-1])
            ax[n0, 1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
            ax[n0, 1].grid()
            ax[n0, 1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

            ax[n0, 2].errorbar(mjd_list2[night_list2 == n1], mag_list2[night_list2 == n1], yerr=std_list2[night_list2 == n1], fmt='gs', markersize=5, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='Comp. star 2')
            ax[n0, 2].set_ylabel('$m$ [mag]')
            ax[n0, 2].set_xlabel('MJD')
            # ax[n0, 2].yaxis.set_minor_locator(MultipleLocator(0.005))
            ax[n0, 2].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
            axtop = ax[n0, 2].twiny()
            axtop.set_xticks(ax[n0, 2].get_xticks())
            axtop.set_xbound(ax[n0, 2].get_xbound())
            axtop.set_xticklabels(tick_function(ax[n0, 2].get_xticks()))#, rotation=70, ha='left')
            axtop.set_xlabel('Elapsed time [minutes]')
            ax[n0, 2].set_ylim(ax[n0, 2].get_ylim()[::-1])
            ax[n0, 2].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
            ax[n0, 2].grid()
            ax[n0, 2].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

            offset_value = np.average(mag_list1[night_list1 == n1])-np.average(mag_list[night_list == n1])
            ax[n0, 3].errorbar(mjd_list[night_list == n1], mag_list[night_list == n1], yerr=std_list[night_list == n1], fmt='b.', markersize=7, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='{}'.format(tn))
            ax[n0, 3].errorbar(mjd_list1[night_list1 == n1], mag_list1[night_list1 == n1]-offset_value, yerr=std_list1[night_list1 == n1], fmt='g.', markersize=7, alpha=0.5, elinewidth=1.0, capsize=0, label='Comp. star 1 (offset)'.format(tn))
            ax[n0, 3].set_ylabel('$m$ [mag]')
            ax[n0, 3].set_xlabel('MJD')
            # ax[n0, 3].yaxis.set_minor_locator(MultipleLocator(0.005))
            ax[n0, 3].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
            axtop = ax[n0, 3].twiny()
            axtop.set_xticks(ax[n0, 3].get_xticks())
            axtop.set_xbound(ax[n0, 3].get_xbound())
            axtop.set_xticklabels(tick_function(ax[n0, 3].get_xticks()))#, rotation=70, ha='left')
            axtop.set_xlabel('Elapsed time [minutes]')
            ax[n0, 3].set_ylim(ax[n0, 3].get_ylim()[::-1])
            ax[n0, 3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
            ax[n0, 3].grid()
            # ax[n0, 3].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1, bbox_to_anchor=(0., 1.3, 1., .102))

        plt.tight_layout()
        pdf.savefig()
        plt.close('all')


def mega_plot(o, suff, fn, tn, nstars, pdf):
    if suff == 'average':
        tname = 'Nightly averaged'
        alpha_plots = 0.5
        alpha_plots2 = 0.7
    else:
        tname = ''
        alpha_plots = 0.3
        alpha_plots2 = 0.5

    fname = os.path.join(o, 'multi_night_LC/megaplot-{}-{}.pdf'.format(fn, suff))

    plt.close('all')
    fig, ax = plt.subplots(9+1, nstars, sharex=False, figsize=(nstars*4.5, 14*2.5))

    RA = []
    DEC = []
    for j in range(nstars):
        info = np.loadtxt(param['output_path']+'data/S{}_info'.format(j))
        RA.append(float(info[1]))
        DEC.append(float(info[2]))

    for j in range(nstars):
        mjd_list_t, mag_list_t, std_list_t, mjd_t, nightly_avg_mag_t, nightly_std_mag_t, night_list = load_data(o, '{}_target'.format(str(j)))
        mjd_list_c, mag_list_c, std_list_c, mjd_c, nightly_avg_mag_c, nightly_std_mag_c, night_list = load_data(o, '{}_refere'.format(str(j)))
        mjd_list_c1, mag_list_c1, std_list_c1, mjd_c1, nightly_avg_mag_c1, nightly_std_mag_c1, night_list = load_data(o, '{}_compa1'.format(str(j)))
        mjd_list_c2, mag_list_c2, std_list_c2, mjd_c2, nightly_avg_mag_c2, nightly_std_mag_c2, night_list = load_data(o, '{}_compa2'.format(str(j)))
        avg_mag_all, std_mag_all, flags = np.loadtxt(o+'std_multi_night_plots/S{}_std_{}_multi_night_02_qc-diff.dat'.format(str(j), fn), unpack=True, delimiter=' ')

        if suff != 'average':
            mjd_t = mjd_list_t
            nightly_avg_mag_t = mag_list_t
            nightly_std_mag_t = std_list_t

            mjd_c = mjd_list_c
            nightly_avg_mag_c = mag_list_c
            nightly_std_mag_c = std_list_c

            mjd_c1 = mjd_list_c1
            nightly_avg_mag_c1 = mag_list_c1
            nightly_std_mag_c1 = std_list_c1

            mjd_c2 = mjd_list_c2
            nightly_avg_mag_c2 = mag_list_c2
            nightly_std_mag_c2 = std_list_c2

        target_mean = np.average(nightly_avg_mag_t)
        target_std1 = np.std(nightly_avg_mag_t)
        target_std2 = np.average(nightly_std_mag_t)

        compar1_mean = np.average(nightly_avg_mag_c1)
        compar1_std1 = np.std(nightly_avg_mag_c1)
        compar1_std2 = np.average(nightly_std_mag_c1)

        compar2_mean = np.average(nightly_avg_mag_c2)
        compar2_std1 = np.std(nightly_avg_mag_c2)
        compar2_std2 = np.average(nightly_std_mag_c2)

        compar_mean = np.average(nightly_avg_mag_c)
        compar_std1 = np.std(nightly_avg_mag_c)
        compar_std2 = np.average(nightly_std_mag_c)

        info = np.loadtxt(param['output_path']+'data/S{}_info'.format(j))
        ra = float(info[1])
        dec = float(info[2])
        w = float(info[3])
        d = float(info[0])

        info1 = np.loadtxt(param['output_path']+'data/S{}_comp_info'.format(1))
        ra1 = float(info1[1])
        dec1 = float(info1[2])
        d1 = float(info1[0])

        info2 = np.loadtxt(param['output_path']+'data/S{}_comp_info'.format(2))
        ra2 = float(info2[1])
        dec2 = float(info2[2])
        d2 = float(info2[0])

        t = Time(mjd_c.max(), format='mjd', scale='utc' )
        last_obs = 'Last observation: {}'.format(t.datetime)

        def tick_function(tick_list):
            return [Time(t, format='mjd', scale='utc' ).datetime.date() for t in tick_list]

        pha_t, mag_pha_t, mag_err_pha_t = compute_orbital_phase(mjd_t, nightly_avg_mag_t, nightly_std_mag_t)
        ax[0, j].errorbar(pha_t, mag_pha_t, fmt='b.', alpha=alpha_plots, markersize=7, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[0, j].set_title('(Corrected using the reference star set {})\n{} LC of {}\nMean magnitude: {:.2f}\nStd of the nightly mean: {:.3f}\nMean of the nightly std: {:.3f}'.format(str(j), tname, tn, target_mean, target_std1, target_std2))
        ax[0, j].set_xlim((0,2))
        ax[0, 0].set_ylabel('$m$ [mag]')
        ax[0, j].set_xlabel('PHASE (P = {:.3f} d)'.format(param['period']))
        # ax[0, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[0, j].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[0, j].set_ylim(ax[0, j].get_ylim()[::-1])
        ax[0, j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[0, j].grid()

        ax[1, j].errorbar(mjd_t, nightly_avg_mag_t, fmt='b.', markersize=7, alpha=alpha_plots2, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[1, 0].set_ylabel('$m$ [mag]')
        ax[1, j].set_xlabel('MJD')
        # ax[1, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[1, j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        axtop = ax[1, j].twiny()
        axtop.set_xticks(ax[1,j].get_xticks())
        axtop.set_xbound(ax[1,j].get_xbound())
        axtop.set_xticklabels(tick_function(ax[1,j].get_xticks()), rotation=70, ha='left')
        ax[1, j].set_ylim(ax[1, j].get_ylim()[::-1])
        ax[1, j].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[1, j].grid()

        pha_t, mag_pha_t, mag_err_pha_t = compute_orbital_phase(mjd_t, nightly_avg_mag_t, nightly_std_mag_t)
        ax[2, j].errorbar(pha_t, mag_pha_t, yerr=mag_err_pha_t, fmt='b.', alpha=alpha_plots, markersize=7, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        if suff == 'average':
            ax[2, j].set_title('{} LC of {} with error bars\n(computed as the std/sqrt(n) of the nightly images)'.format(tname, tn))
        else:
            ax[2, j].set_title('{} LC of {} with error bars\n(computed as the std of the nightly images)'.format(tname, tn))
        ax[2, j].set_xlim((0,2))
        ax[2, 0].set_ylabel('$m$ [mag]')
        ax[2, j].set_xlabel('PHASE (P = {:.3f} d)'.format(param['period']))
        # ax[2, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[2, j].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax[2, j].set_ylim(ax[2, j].get_ylim()[::-1])
        ax[2, j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[2, j].grid()

        ax[3, j].errorbar(mjd_t, nightly_avg_mag_t, yerr=nightly_std_mag_t, fmt='b.', markersize=7, alpha=alpha_plots2, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[3, 0].set_ylabel('$m$ [mag]')
        ax[3, j].set_xlabel('MJD')
        # ax[3, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[3, j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        axtop = ax[3, j].twiny()
        axtop.set_xticks(ax[3,j].get_xticks())
        axtop.set_xbound(ax[3,j].get_xbound())
        axtop.set_xticklabels(tick_function(ax[1,j].get_xticks()), rotation=70, ha='left')
        ax[3, j].set_ylim(ax[3, j].get_ylim()[::-1])
        ax[3, j].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[3, j].grid()

        ax[4, j].errorbar(mjd_c1, nightly_avg_mag_c1, fmt='g.', markersize=5, alpha=alpha_plots2, elinewidth=1.0, capsize=0, label='Comparison star {} ($\sigma$ = {:.4f})'.format(tname, 1, np.std(nightly_avg_mag_c1)))
        ax[4, j].set_title('{} LC of comp. star {}\nMean magnitude: {:.2f}\nStd of the nightly mean: {:.3f}\nMean of the nightly std: {:.3f}\nRA {:.3f} deg, DEC {:.3f} deg\nDistance with respect to {}: {:.2f} arcmin'.format(tname, str(1), compar1_mean, compar1_std1, compar1_std2, ra1, dec1, tn, d1))
        ax[4, 0].set_ylabel('$m$ [mag]')
        ax[4, j].set_xlabel('MJD')
        # ax[4, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[4, j].set_ylim(ax[4, j].get_ylim()[::-1])
        ax[4, j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[4, j].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[4, j].grid()

        ax[5, j].errorbar(mjd_c2, nightly_avg_mag_c2, fmt='gs', markersize=3, alpha=alpha_plots2, elinewidth=1.0, capsize=0, markeredgewidth=0, label='Comparison star {} ($\sigma$ = {:.4f})'.format(2, np.std(nightly_avg_mag_c2)))
        ax[5, j].set_title('{} LC of comp. star {}\nMean magnitude: {:.2f}\nStd of the nightly mean: {:.3f}\nMean of the nightly std: {:.3f}\nRA {:.3f} deg, DEC {:.3f} deg\nDistance with respect to {}: {:.2f} arcmin'.format(tname, str(2), compar2_mean, compar2_std1, compar2_std2, ra2, dec2, tn, d2))
        ax[5, 0].set_ylabel('$m$ [mag]')
        ax[5, j].set_xlabel('MJD')
        # ax[5, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[5, j].set_ylim(ax[5, j].get_ylim()[::-1])
        ax[5,j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[5, j].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[5, j].grid()

        offset_value1 = np.average(nightly_avg_mag_c1)-np.average(nightly_avg_mag_t)
        offset_value2 = np.average(nightly_avg_mag_c2)-np.average(nightly_avg_mag_t)
        ax[6, j].errorbar(mjd_t, nightly_avg_mag_t, fmt='b.', markersize=7, alpha=alpha_plots2, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[6, j].errorbar(mjd_c1, nightly_avg_mag_c1-offset_value1, fmt='g.', markersize=5, elinewidth=1.0, alpha=0.5, capsize=0, label='Comparison star {} (offset)'.format(1))
        ax[6, j].set_title('{} LCs of {} and comp. star {}'.format(tname, tn, 1))
        ax[6, 0].set_ylabel('$m$ [mag]')
        ax[6, j].set_xlabel('MJD')
        # ax[6, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[6, j].set_ylim(ax[6, j].get_ylim()[::-1])
        ax[6,j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[6, j].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[6, j].grid()

        ax[7, j].errorbar(mjd_c, nightly_avg_mag_c, fmt='r.', markersize=5, alpha=alpha_plots2, elinewidth=1.0, capsize=0, label='Excluded reference star {} ($\sigma$ = {:.4f})'.format(j, np.std(nightly_avg_mag_c)))
        ax[7, j].set_title('{} LC of the excluded ref. star {}\nMean magnitude: {:.2f}\nStd of the nightly mean: {:.3f}\nMean of the nightly std: {:.3f}\nCorrection weight (when included): {:.2e}\nRA {:.3f} deg, DEC {:.3f} deg\nDistance with respect to {}: {:.2f} arcmin'.format(tname, str(j), compar_mean, compar_std1, compar_std2, w, ra, dec, tn, d))
        ax[7, 0].set_ylabel('$m$ [mag]')
        ax[7, j].set_xlabel('MJD')
        # ax[7, j].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[7, j].set_ylim(ax[7, j].get_ylim()[::-1])
        ax[7, j].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[7, j].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[7, j].grid()

        ax[8, j].plot(avg_mag_all[flags==0], std_mag_all[flags==0], 'k.', alpha=0.5, label='Field stars', ms=10)
        ax[8, j].plot(avg_mag_all[flags==3], std_mag_all[flags==3], 'y.', label='Ref. stars', ms=10)
        ax[8, j].plot(avg_mag_all[flags==4], std_mag_all[flags==4], 'g.', label='Comp. star 1', ms=10)
        ax[8, j].plot(avg_mag_all[flags==5], std_mag_all[flags==5], 'gs', label='Comp. star 2', ms=6, markeredgewidth=0,)
        ax[8, j].plot(avg_mag_all[flags==2], std_mag_all[flags==2], 'r.', label='Excluded ref. star {}'.format(j), ms=10)
        ax[8, j].plot(avg_mag_all[flags==1], std_mag_all[flags==1], 'b.', label='{}'.format(tn), ms=10)
        ax[8, j].set_yscale('log')
        ax[8, j].set_xlabel(r'$\overline{m}$ [mag]')
        ax[8, j].set_ylabel(r'$\sigma_{m}$ [mag]')
        ax[8, j].set_title('Nightly standard deviation', y=1.40)
        ax[8, j].legend(bbox_to_anchor=(0., 1.3, 1., .102), loc=9, fancybox=True, framealpha=0.5, numpoints=1, ncol=2)
        ax[8, j].grid()

        ax[9, j].plot(RA, DEC, 'y.', label='Ref. stars', ms=10)
        ax[9, j].plot(ra, dec, 'r.', label='Excluded ref. star', ms=10)
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        c = SkyCoord(param['ra'], param['dec'], unit=(u.hour, u.deg))
        ax[9, j].plot(c.ra.deg, c.dec.deg, 'b.', label='{}'.format(tn), ms=10)
        ax[9, j].plot(ra1, dec1, 'g.', label='Comp. star 1', ms=10)
        ax[9, j].plot(ra2, dec2, 'gs', label='Comp. star 2', markeredgewidth=0, ms=6)
        ax[9, j].set_xlabel('RA (deg)')
        ax[9, j].set_ylabel('DEC (deg)')
        ax[9, j].set_title('Equatorial map', y=1.4)
        ax[9, j].legend(bbox_to_anchor=(0., 1.3, 1., .102), loc=9, fancybox=True, framealpha=0.5, numpoints=1, ncol=2)
        ax[9, j].yaxis.set_minor_locator(MultipleLocator(0.05))
        ax[9, j].xaxis.set_minor_locator(MultipleLocator(0.05))
        ax[9, j].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[9, j].set_xticklabels(ax[9, j].get_xticks(), rotation=70, ha='right')
        ax[9, j].grid()

    plt.tight_layout()
    pdf.savefig()
    plt.close('all')
    # fig.savefig(fname, bbox_inches='tight', pad_inches=0.05)
    # auto_crop_img(fname)


def plot_mjd(x, y, yerr, x1, y1, yerr1, o, pdf):
    def tick_function(tick_list):
        return [Time(t, format='mjd', scale='utc' ).datetime.date() for t in tick_list]

    ymin = min(np.asarray(y)-np.asarray(yerr))
    ymax = max(np.asarray(y)+np.asarray(yerr))

    plt.close('all')
    f, ax = plt.subplots(1, 2, figsize=(2*7,1*5.5))
    ax[0].errorbar(x, y, yerr=yerr, fmt='k.', markersize=8, alpha=0.7, elinewidth=1.0, capsize=0)
    ax[0].set_xlabel(r'MJD')
    ax[0].set_ylabel(r'$m$ [mag]')
    ax[0].set_xlim((min(x)-6, max(x)+6))
    ax[0].set_ylim((ymin*(1-0.001), ymax*(1+0.001)))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].grid()
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    axtop = ax[0].twiny()
    axtop.set_xticks(ax[0].get_xticks())
    axtop.set_xbound(ax[0].get_xbound())
    axtop.set_xticklabels(tick_function(ax[0].get_xticks()), rotation=70, ha='left')


    ax[1].errorbar(x1, y1, yerr=yerr1, fmt='k.', markersize=8, alpha=0.7, elinewidth=1.0, capsize=0, label='Comp. star')
    ax[1].set_xlabel(r'MJD')
    ax[1].set_ylabel(r'$m$ [mag]')
    ax[1].set_xlim((min(x)-6, max(x)+6))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].grid()
    ax[1].set_ylim(ax[1].get_ylim()[::-1])
    ax[1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    axtop = ax[1].twiny()
    axtop.set_xticks(ax[1].get_xticks())
    axtop.set_xbound(ax[1].get_xbound())
    axtop.set_xticklabels(tick_function(ax[1].get_xticks()), rotation=70, ha='left')


    plt.tight_layout()
    pdf.savefig()
    plt.close('all')
    # f.savefig(o, bbox_inches='tight', pad_inches=0.05)
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


def plot_phase(i, i1, o, pdf, plot_errors=True):
    x, y, yerr = i
    x1, y1, yerr1 = i1
    if not plot_errors:
        yerr=yerr*0.
        yerr1=yerr*0.
    ymin = min(np.asarray(y)-np.asarray(yerr))
    ymax = max(np.asarray(y)+np.asarray(yerr))

    plt.close('all')
    f, ax = plt.subplots(1, 2, figsize=(2*7,1*5))

    ax[0].errorbar(x, y, yerr=yerr, fmt='k.', markersize=8, alpha=0.8, elinewidth=1.0, capsize=0)
    ax[0].set_xlabel(r'PHASE')
    ax[0].set_ylabel(r'$m$ [mag]')
    ax[0].set_xlim(0, 2)
    ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].grid()
    ax[0].set_ylim((ymin*(1-0.001), ymax*(1+0.001)))
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))

    ax[1].errorbar(x1, y1, yerr=yerr1, fmt='k.', markersize=8, alpha=0.8, elinewidth=1.0, capsize=0, label='Comp. star 1')
    ax[1].set_xlabel(r'PHASE')
    ax[1].set_ylabel(r'$m$ [mag]')
    ax[1].set_xlim(0, 2)
    ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].grid()
    ax[1].set_ylim(ax[1].get_ylim()[::-1])
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    ax[1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

    plt.tight_layout()
    pdf.savefig()
    plt.close('all')
    # f.savefig(o, bbox_inches='tight', pad_inches=0.05)
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


def plot_mjd_cycles(x_cyc, y_cyc, yerr_cyc, x_cyc1, y_cyc1, yerr_cyc1, o, pdf, plot_errors=True):
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
    ax[0].set_ylabel(r'$m$ [mag]')
    ax[0].set_xlim(xmin-6, xmax+6)
    ax[0].set_ylim(ymin*(1-0.001), ymax*(1+0.001))
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
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
    ax[1].set_ylabel(r'$m$ [mag]')
    ax[1].set_xlim(xmin-6, xmax+6)
    ax[1].set_ylim(ax[1].get_ylim()[::-1])
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


def plot_phase_cycles(x_cyc, y_cyc, yerr_cyc, x_cyc1, y_cyc1, yerr_cyc1, o, pdf, plot_errors=True):
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
    ax[0].set_ylabel(r'$m$ [mag]')
    ax[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator())
    ax[0].grid()
    ax[0].set_xlim(0, 2)
    ax[0].set_ylim(ymin*(1-0.001), ymax*(1+0.001))
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
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
    ax[1].set_ylabel(r'$m$ [mag]')
    ax[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator())
    ax[1].grid()
    ax[1].set_xlim(0, 2)
    ax[1].set_ylim(ax[1].get_ylim()[::-1])
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    ax[1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

    plt.tight_layout()
    pdf.savefig()
    plt.close('all')
    # f.savefig(o, bbox_inches='tight', pad_inches=0.05)
    # auto_crop_img(o)

    # CYCLE SUBPLOTS
    k =0
    if len(x_cyc) == 1:
        k = 1
    fs = (2*7, (len(x_cyc)+k)*4.6)

    plt.close('all')
    f2, ax2 = plt.subplots(len(x_cyc)+k, 2, sharex=False, sharey=False, figsize=fs)
    colormap = eval('plt.cm.'+param['colormap_cycles'])
    col_range = param['colormap_cycles_range']
    # ax2.set_color_cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])

    colorc = itertools.cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
    marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
    for (k, (x, y, yerr)) in enumerate(zip(x_cyc, y_cyc, yerr_cyc)):
        xp, yp, yerrp = compute_orbital_phase(np.asarray(x), np.asarray(y), np.asarray(yerr))
        if not plot_errors:
            yerrp=np.array(yerrp)*0.
        ax2[k, 0].errorbar(xp, yp, yerr=yerrp, fmt='.', marker=marker.next(), color=colorc.next(), markersize=8, elinewidth=1.0, capsize=0,
                           markeredgewidth=0.00)
        tmin = Time(np.asarray(x).min(), format='mjd', scale='utc').datetime.date()
        tmax = Time(np.asarray(x).max(), format='mjd', scale='utc').datetime.date()
        ax2[k, 0].set_title('Cycle number {} of {}\nFrom {} until {}'.format(str(k+1), len(x_cyc), tmin, tmax))
        ax2[k, 0].set_xlabel(r'PHASE')
        ax2[k, 0].set_ylabel(r'$m$ [mag]')
        ax2[k, 0].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax2[k, 0].yaxis.set_minor_locator(AutoMinorLocator())
        ax2[k, 0].grid()
        ax2[k, 0].set_xlim(0, 2)
        ax2[k, 0].set_ylim(ymin*(1-0.001), ymax*(1+0.001))
        ax2[k, 0].set_ylim(ax2[k, 0].get_ylim()[::-1])
        ax2[k, 0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    if len(x_cyc) == 1:
        ax2[1, 0].axis('off')

    colorc = itertools.cycle([colormap(i) for i in np.linspace(col_range[0], col_range[1], len(x_cyc))])
    marker = itertools.cycle(('^', 's', 'o', 'p', 'v', '<', 'H', '*', 'h', '<', '>', 'D', 'd', '4'))
    for (k, (x, y, yerr)) in enumerate(zip(x_cyc1, y_cyc1, yerr_cyc1)):
        xp, yp, yerrp = compute_orbital_phase(np.asarray(x), np.asarray(y), np.asarray(yerr))
        if not plot_errors:
            yerrp=np.array(yerrp)*0.
        ax2[k, 1].errorbar(xp, yp, yerr=yerrp, fmt='.', marker=marker.next(), color=colorc.next(), markersize=8, elinewidth=1.0, capsize=0,
                           markeredgewidth=0.00, label='Comp. star 1')
        tmin = Time(np.asarray(x).min(), format='mjd', scale='utc').datetime.date()
        tmax = Time(np.asarray(x).max(), format='mjd', scale='utc').datetime.date()
        ax2[k, 1].set_title('Cycle number {} of {}\nFrom {} until {}'.format(str(k+1), len(x_cyc), tmin, tmax))
        ax2[k, 1].set_xlabel(r'PHASE')
        ax2[k, 1].set_ylabel(r'$m$ [mag]')
        ax2[k, 1].xaxis.set_minor_locator(MultipleLocator(0.1))
        ax2[k, 1].yaxis.set_minor_locator(AutoMinorLocator())
        ax2[k, 1].grid()
        ax2[k, 1].set_xlim(0, 2)
        ax2[k, 1].set_ylim(ax2[k, 1].get_ylim()[::-1])
        ax2[k, 1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax2[k, 1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)
    if len(x_cyc) == 1:
        ax2[1, 1].axis('off')

    plt.tight_layout()
    pdf.savefig()
    plt.close('all')


def make_plots():
    fig_width_pt = 1*512.1496              # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27              # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0     # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt # width in inches
    fig_height = fig_width*golden_mean     # height in inches
    fig_size =  [fig_width,fig_height]
    params = {'backend': 'pdf',
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

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('{}data/mega_plot_{}.pdf'.format(output_path, field_name)) as pdf:
        mjd_list, mag_list, std_list, mjd, nightly_avg_mag, nightly_std_mag, night_list = load_data(output_path, '{}_target'.format('0'))
        mjd_list1, mag_list1, std_list1, mjd1, nightly_avg_mag1, nightly_std_mag1, night_list1 = load_data(output_path, '{}_compa1'.format('0'))

        if param['field_name'] == 'lsi61303':
            try:
                from plots_lsi61303 import make_plots_lsi61303
                make_plots_lsi61303(pdf)
            except:
                pass

        # WITH ERROR BARS
        # MJD (averaged)
        plot_mjd(mjd, nightly_avg_mag, nightly_std_mag, mjd1, nightly_avg_mag1, nightly_std_mag1, param['output_path']
                 + 'multi_night_LC/MJD-{}-target-nightly_average.eps'.format(param['field_name']), pdf)
        # PHASE (averaged)
        plot_phase(compute_orbital_phase(mjd, nightly_avg_mag, nightly_std_mag), compute_orbital_phase(mjd1, nightly_avg_mag1, nightly_std_mag1), param['output_path']
                   + 'multi_night_LC/PHA-{}-target-nightly_average.eps'.format(param['field_name']), pdf)
        # Cycle coloured LCs (averaged)
        if param['disable_plots_cycles'] == 0:
            mjd_cyc, mag_cyc, merr_cyc = make_cycles(mjd, nightly_avg_mag, nightly_std_mag)
            mjd_cyc1, mag_cyc1, merr_cyc1 = make_cycles(mjd1, nightly_avg_mag1, nightly_std_mag1)
            # MJD (averaged)
            plot_mjd_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                            + 'multi_night_LC/MJD-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf)
            # PHASE (averaged)
            plot_phase_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                              + 'multi_night_LC/PHA-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf)
        # MJD (not averaged)
        plot_mjd(mjd_list, mag_list, std_list, mjd_list1, mag_list1, std_list1, param['output_path']
                 + 'multi_night_LC/MJD-{}-target-all_frames.eps'.format(param['field_name']), pdf)
        # PHASE (not averaged)
        plot_phase(compute_orbital_phase(mjd_list, mag_list, std_list), compute_orbital_phase(mjd_list1, mag_list1, std_list1), param['output_path']
                   + 'multi_night_LC/PHA-{}-target-all_frames.eps'.format(param['field_name']), pdf)
        np.savetxt(param['output_path']+'data/'+'PHA_MAG_ERR-{}-nightly_average.dat'.format(param['field_name']),
                   np.transpose(compute_orbital_phase_mid(mjd, nightly_avg_mag, nightly_std_mag)), delimiter=' ')
        # Cycle coloured LCs (not averaged)
        if param['disable_plots_cycles'] == 0:
            mjd_cyc, mag_cyc, merr_cyc = make_cycles(mjd_list, mag_list, std_list)
            mjd_cyc1, mag_cyc1, merr_cyc1 = make_cycles(mjd_list1, mag_list1, std_list1)
            # MJD (not averaged)
            plot_mjd_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                            + 'multi_night_LC/MJD-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf)
            # PHASE (not averaged)
            plot_phase_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                              + 'multi_night_LC/PHA-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf)

        # WITHOUT ERROR BARS
        # MJD (averaged)
        plot_mjd(mjd, nightly_avg_mag, nightly_std_mag*0., mjd1, nightly_avg_mag1, nightly_std_mag1*0.,param['output_path']
                 + 'multi_night_LC/MJD-{}-target-nightly_average.eps'.format(param['field_name']), pdf)
        # PHASE (averaged)
        plot_phase(compute_orbital_phase(mjd, nightly_avg_mag, nightly_std_mag), compute_orbital_phase(mjd1, nightly_avg_mag1, nightly_std_mag1), param['output_path']
                   + 'multi_night_LC/PHA-{}-target-nightly_average.eps'.format(param['field_name']), pdf, False)
        # Cycle coloured LCs (averaged)
        if param['disable_plots_cycles'] == 0:
            mjd_cyc, mag_cyc, merr_cyc = make_cycles(mjd, nightly_avg_mag, nightly_std_mag)
            mjd_cyc1, mag_cyc1, merr_cyc1 = make_cycles(mjd1, nightly_avg_mag1, nightly_std_mag1)
            # MJD (averaged)
            plot_mjd_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                            + 'multi_night_LC/MJD-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf, False)
            # PHASE (averaged)
            plot_phase_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                              + 'multi_night_LC/PHA-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf, False)
        # MJD (not averaged)
        plot_mjd(mjd_list, mag_list, std_list*0., mjd_list1, mag_list1, std_list1*0., param['output_path']
                 + 'multi_night_LC/MJD-{}-target-all_frames.eps'.format(param['field_name']), pdf)
        # PHASE (not averaged)
        plot_phase(compute_orbital_phase(mjd_list, mag_list, std_list), compute_orbital_phase(mjd_list1, mag_list1, std_list1), param['output_path']
                   + 'multi_night_LC/PHA-{}-target-all_frames.eps'.format(param['field_name']), pdf, False)
        # Cycle coloured LCs (not averaged)
        if param['disable_plots_cycles'] == 0:
            mjd_cyc, mag_cyc, merr_cyc = make_cycles(mjd_list, mag_list, std_list)
            mjd_cyc1, mag_cyc1, merr_cyc1 = make_cycles(mjd_list1, mag_list1, std_list1)
            # MJD (not averaged)
            plot_mjd_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                            + 'multi_night_LC/MJD-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf, False)
            # PHASE (not averaged)
            plot_phase_cycles(mjd_cyc, mag_cyc, merr_cyc, mjd_cyc1, mag_cyc1, merr_cyc1, param['output_path']
                              + 'multi_night_LC/PHA-{}-target-nightly_average_cycles.eps'.format(param['field_name']), pdf, False)

        mega_plot(output_path, 'average', field_name, title_name, nstars, pdf)
        mega_plot(output_path, 'single', field_name, title_name, nstars, pdf)
        if param['disable_plots_nightly'] == 0:
            nightly_plots(output_path, 'single', field_name, title_name, nstars, pdf)


if __name__ == '__main__':
    # Testing / plotting from data
    make_plots()
    print('DONE')
