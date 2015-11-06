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

    plt.close('all')
    fig, ax = plt.subplots(nnights, 4, sharex=False, figsize=(4*4.5, nnights*4)) # 1.5 l'he baixat

    alpha_plots = 0.8
    for n in _night_list:
        n = int(n)-1
        n1 = n+1
        ax[n, 0].errorbar(mjd_list[night_list == n1], mag_list[night_list == n1], yerr=std_list[night_list == n1], fmt='b.', markersize=7, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[n, 0].set_ylabel('$m$ [mag]')
        ax[n, 0].set_xlabel('MJD')
        # ax[n, 0].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[n, 0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        ax[n, 0].set_title('Night {} of {}\nStart: {}\nEnd: {}'.format(n1, nnights, Time(mjd_list[night_list == n1][0], format='mjd', scale='utc' ).datetime, Time(mjd_list[night_list == n1][-1], format='mjd', scale='utc' ).datetime), y=1.2)
        axtop = ax[n, 0].twiny()
        axtop.set_xticks(ax[n, 0].get_xticks())
        axtop.set_xbound(ax[n, 0].get_xbound())
        axtop.set_xticklabels(tick_function(ax[n, 0].get_xticks()))#, rotation=70, ha='left')
        axtop.set_xlabel('Elapsed time [minutes]')
        ax[n, 0].set_ylim(ax[n, 0].get_ylim()[::-1])
        ax[n, 0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[n, 0].grid()
        ax[n, 0].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

        ax[n, 1].errorbar(mjd_list1[night_list1 == n1], mag_list1[night_list1 == n1], yerr=std_list1[night_list1 == n1], fmt='g.', markersize=7, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='Comp. star 1')
        ax[n, 1].set_ylabel('$m$ [mag]')
        ax[n, 1].set_xlabel('MJD')
        # ax[n, 1].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[n, 1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        axtop = ax[n, 1].twiny()
        axtop.set_xticks(ax[n, 1].get_xticks())
        axtop.set_xbound(ax[n, 1].get_xbound())
        axtop.set_xticklabels(tick_function(ax[n, 1].get_xticks()))#, rotation=70, ha='left')
        axtop.set_xlabel('Elapsed time [minutes]')
        ax[n, 1].set_ylim(ax[n, 1].get_ylim()[::-1])
        ax[n, 1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[n, 1].grid()
        ax[n, 1].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

        ax[n, 2].errorbar(mjd_list2[night_list2 == n1], mag_list2[night_list2 == n1], yerr=std_list2[night_list2 == n1], fmt='gs', markersize=5, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='Comp. star 2')
        ax[n, 2].set_ylabel('$m$ [mag]')
        ax[n, 2].set_xlabel('MJD')
        # ax[n, 2].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[n, 2].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        axtop = ax[n, 2].twiny()
        axtop.set_xticks(ax[n, 2].get_xticks())
        axtop.set_xbound(ax[n, 2].get_xbound())
        axtop.set_xticklabels(tick_function(ax[n, 2].get_xticks()))#, rotation=70, ha='left')
        axtop.set_xlabel('Elapsed time [minutes]')
        ax[n, 2].set_ylim(ax[n, 2].get_ylim()[::-1])
        ax[n, 2].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[n, 2].grid()
        ax[n, 2].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1)

        offset_value = np.average(mag_list1[night_list1 == n1])-np.average(mag_list[night_list == n1])
        ax[n, 3].errorbar(mjd_list[night_list == n1], mag_list[night_list == n1], yerr=std_list[night_list == n1], fmt='b.', markersize=7, alpha=alpha_plots, elinewidth=1.0, capsize=0, label='{}'.format(tn))
        ax[n, 3].errorbar(mjd_list1[night_list1 == n1], mag_list1[night_list1 == n1]-offset_value, yerr=std_list1[night_list1 == n1], fmt='g.', markersize=7, alpha=0.5, elinewidth=1.0, capsize=0, label='Comp. star 1 (offset)'.format(tn))
        ax[n, 3].set_ylabel('$m$ [mag]')
        ax[n, 3].set_xlabel('MJD')
        # ax[n, 3].yaxis.set_minor_locator(MultipleLocator(0.005))
        ax[n, 3].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        axtop = ax[n, 3].twiny()
        axtop.set_xticks(ax[n, 3].get_xticks())
        axtop.set_xbound(ax[n, 3].get_xbound())
        axtop.set_xticklabels(tick_function(ax[n, 3].get_xticks()))#, rotation=70, ha='left')
        axtop.set_xlabel('Elapsed time [minutes]')
        ax[n, 3].set_ylim(ax[n, 3].get_ylim()[::-1])
        ax[n, 3].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=True))
        ax[n, 3].grid()
        # ax[n, 3].legend(loc='upper right', fancybox=True, framealpha=0.5, numpoints=1, bbox_to_anchor=(0., 1.3, 1., .102))

    plt.tight_layout()
    pdf.savefig()
    plt.close()


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
    plt.close()
    # fig.savefig(fname, bbox_inches='tight', pad_inches=0.05)
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

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages('{}data/mega_plot_{}.pdf'.format(output_path, field_name)) as pdf:
        mega_plot(output_path, 'average', field_name, title_name, nstars, pdf)
        mega_plot(output_path, 'single', field_name, title_name, nstars, pdf)
        nightly_plots(output_path, 'single', field_name, title_name, nstars, pdf)

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
    # #

if __name__ == '__main__':
    # Testing / plotting from data
    make_plots()
    print('DONE')
