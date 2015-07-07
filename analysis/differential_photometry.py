# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import sys
param = {}
execfile(sys.argv[1])


def find_target(ra, dec, ra0, dec0, testing=0):
    tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
    cat = SkyCoord(ra*u.degree, dec*u.degree)
    ind, sep2d, dist3d = tar.match_to_catalog_sky(cat)
    if sep2d.arcsec > param['astrometric_tolerance']:
        raise RuntimeError("Target not found")
    if testing == 1:
        ra = np.delete(ra, ind)
        dec = np.delete(dec, ind)
        ind = find_target(ra, dec, ra0, dec0, testing=0)
    return ind


def distance_selection(ra, dec, ra0, dec0, bool_sel, testing=1):
    dmax = param['dmax']
    tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
    cat = SkyCoord(ra*u.degree, dec*u.degree)
    sep2d = tar.separation(cat)

    if testing == 1:
        print '  Before DISTANCE selection: (all distances)'
        print '  - Minimum separation: {:.2f}'.format(min(sep2d.deg[bool_sel]))
        print '  - Maximum separation: {:.2f}'.format(max(sep2d.deg[bool_sel]))
        print '  - Number of stars: {}'.format(len(sep2d.deg[bool_sel]))

    bool_sel[sep2d.deg > dmax] = False
    if testing == 1:
        print '\n  After DISTANCE selection: (dmax = {})'.format(dmax)
        print '  - Minimum separation: {:.2f}'.format(min(sep2d.deg[bool_sel]))
        print '  - Maximum separation: {:.2f}'.format(max(sep2d.deg[bool_sel]))
        print '  - Number of stars: {}\n'.format(len(sep2d.deg[bool_sel]))
    return bool_sel


def magnitude_selection(cat_mag, ind, mmin, mmax, bool_sel, testing=1):
    avg_mag = np.asarray([np.average(cat_mag[:, k]) for k in xrange(len(cat_mag[0, :]))])
    avg_tar = avg_mag[ind]

    if testing == 1:
        print '  Before MAGNITUDE selection: (all magnitudes)'
        print '  - Minimum magnitude value: {:.2f}'.format(min(avg_mag[bool_sel]))
        print '  - Maximum magnitude value: {:.2f}'.format(max(avg_mag[bool_sel]))
        print '  - Number of stars: {}'.format(len(avg_mag[bool_sel]))

    bool_sel[avg_mag > avg_tar+mmax] = False
    bool_sel[avg_mag < avg_tar-mmin] = False
    if testing == 1:
        print '\n  After MAGNITUDE selection such as m in [mtar-mmin, mtar+mmax]:'
        print '  - Maximum magnitude difference: (mmin, mmax) = ({}, {})'.format(mmin, mmax)
        print '  - Minimum magnitude value: {:.2f}'.format(min(avg_mag[bool_sel]))
        print '  - Target mean magnitude: {:.2f}'.format(avg_tar)
        print '  - Maximum magnitude value: {:.2f}'.format(max(avg_mag[bool_sel]))
        print '  - Number of stars: {}'.format(len(avg_mag[bool_sel]))
    return bool_sel


def correct_magnitudes(cm, m0, ind_ref):
    ns = len(ind_ref)
    for (i, m) in enumerate(cm):
        # Number of frames
        nf = len(m[:, 0])

        # Weights
        w = np.asarray([1./np.std(m[:, ind_ref[k]], axis=0)**2 for k in xrange(ns)])

        # Computing the correcting factor for each frame using the average variation of ns stars
        dm = np.average([m[:, ind_ref[k]]-m0[ind_ref[k]] for k in xrange(ns)], axis=0, weights=w)

        # Applying the correcting factor to each frame
        cm[i] = np.array([m[j, :]-dm[j] for j in xrange(nf)])
    return cm


def compute_std(cat_mag_corrected, it, ind, ns, ind_ref):
    nstars = len(cat_mag_corrected[0][0, :])
    avg_nightly_m = np.asarray([[np.average(mag[:, i]) for i in xrange(nstars)] for mag in cat_mag_corrected])
    std_m = np.asarray([np.std(avg_nightly_m[:, i]) for i in xrange(nstars)])

    # WRONG, std_m2 does not account for night to night variations
    # std_m2 = np.average([np.std(mag[:, :], axis=0) for mag in cat_mag_corrected], axis=0)

    print '\n  After correcting loop {}:'.format(it)
    print '  - Number of stars used for correction {} of {} stars'.format(ns, nstars)
    print '  - Labeled with (the first 10 of them): {}'.format(ind_ref[0:10])
    print '  - Multi night target STD of the averaged mag is of {:.4f}'.format(std_m[ind])
    print '  - Target MEAN magnitude is of {:.3f}'.format(np.average(avg_nightly_m[:, ind]))
    return avg_nightly_m, std_m


def extinction_correction(cat_mag, ind, bool_sel, nsel):
    # First correction using all the available stars
    mag0 = cat_mag[0][0, :]
    ind_ref = range(len(mag0))
    cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref)

    # Many corrections
    it = 0
    while True:
        it += 1
        ind_ref_aux = np.copy(ind_ref)

        # Compute the standard deviation of the nightly averages
        avg_m, std_m = compute_std(cat_mag_corrected, it, ind, len(ind_ref), ind_ref)

        # Remove the target from the comparison star candidates
        std_m[ind] = 9999999

        # Remove less suitable comparison stars based on distance and magnitude criteria
        std_m[np.where(bool_sel == False)] = 9999999

        # Apply correction only using the selected stars
        ind_ref = std_m.argsort()[0:nsel]

        print '\nStar label {}, std: {}'.format(ind_ref[0], std_m[ind_ref[0]])
        print 'Star label {}, std: {}'.format(ind_ref[1], std_m[ind_ref[1]])
        print 'Star label {}, std: {}'.format(ind_ref[2], std_m[ind_ref[2]])
        print 'Star label {}, std: {}'.format(ind_ref[3], std_m[ind_ref[3]])

        if np.array_equal(np.sort(ind_ref), np.sort(ind_ref_aux)):
            break
        mag0 = cat_mag[0][0, :]
        cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref)
    return cat_mag_corrected, ind_ref


def ref_stars_info(ind_ref, mag, ra, dec, ra0, dec0):
    tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
    cat = SkyCoord(ra[ind_ref]*u.degree, dec[ind_ref]*u.degree)
    d = tar.separation(cat)
    m = [np.average(mag[:, k]) for k in ind_ref]
    r = [ra[k] for k in ind_ref]
    dc = [dec[k] for k in ind_ref]
    print '\n  Number of reference stars: {}'.format(len(ind_ref))
    for i, ir in enumerate(ind_ref):
        print '  - The reference star {} labeled with {} is\n' \
              '    placed at a DISTANCE of {:.2f} with respect the target\n' \
              '    and its MAGNITUDE is {:.2f} mag (RA {:.4f}, DEC {:.4f}) '.format(i+1, ir, d[i], m[i], r[i], dc[i])


def extinction_correction_ref(cat_mag, ind, ir, ind_ref, testing=1):
    nref = len(ind_ref)
    ir_label = ind_ref[ir]
    ind_ref = np.delete(ind_ref, ir)
    mag0 = cat_mag[0][0, :]
    cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref)
    if testing == 1:
        nstars = len(cat_mag_corrected[0][0, :])
        avg_nightly_m = np.asarray([[np.average(mag[:, i]) for i in xrange(nstars)] for mag in cat_mag_corrected])
        std_m = np.asarray([np.std(avg_nightly_m[:, k]) for k in xrange(nstars)])
        print '\n  After removing only the {} reference star of {} labeled with {}:'.format(ir+1, nref, ir_label)
        print '  - Multi night target STD of the averaged mag is of {:.4f}'.format(std_m[ind])
        print '  - Target MEAN magnitude is of {:.3f}'.format(np.average(avg_nightly_m[:, ind]))
    return cat_mag_corrected


def plot_self_corrected(cat_mag_corrected, cat_mjd, ind, ind_ref, o):
    nstars = len(cat_mag_corrected[0][0, :])
    avg_nightly_mag = np.asarray([[np.average(mag[:, i]) for i in xrange(nstars)] for mag in cat_mag_corrected])
    std_nightly_mag = np.asarray([[np.std(mag[:, i]) for i in xrange(nstars)] for mag in cat_mag_corrected])
    avg_mjd = np.asarray([np.average(mjd) for mjd in cat_mjd])

    plt.rcdefaults()
    f, ax = plt.subplots(len(ind_ref)+1, 1, figsize=(3, 3*(len(ind_ref)+1)), sharex=True)
    # f.subplots_adjust(hspace=1)
    ax[0].set_title('Target'.format(i))
    ax[0].errorbar(avg_mjd, avg_nightly_mag[:, ind], yerr=std_nightly_mag[:, ind], fmt='k.', markersize=8,
                   elinewidth=1.0, capsize=0)
    ax[0].set_xlabel(r'MJD')
    ax[0].set_ylabel(r'$m$ (mag)')
    ax[0].set_xlim((min(avg_mjd)*(1-0.00005), max(avg_mjd)*(1+0.00005)))
    ax[0].set_ylim((min(avg_nightly_mag[:, ind])*(1-0.005), max(avg_nightly_mag[:, ind])*(1+0.005)))
    ax[0].set_ylim(ax[0].get_ylim()[::-1])
    # from matplotlib.ticker import MultipleLocator
    # ax[0].xaxis.set_minor_locator(MultipleLocator(0.5))
    for (i, indi) in enumerate(ind_ref):
        ax[i+1].set_title('Comparison star {} ({})'.format(i+1, indi))
        ax[i+1].errorbar(avg_mjd, avg_nightly_mag[:, indi], yerr=std_nightly_mag[:, indi], fmt='k.', markersize=8,
                         elinewidth=1.0, capsize=0)
        ax[i+1].set_ylabel(r'$m$ (mag)')
        ax[i+1].set_ylim((min(avg_nightly_mag[:, indi])*(1-0.005), max(avg_nightly_mag[:, indi])*(1+0.005)))
        ax[i+1].set_ylim(ax[i+1].get_ylim()[::-1])
    f.savefig(o, bbox_inches='tight', pad_inches=0.05)
    plt.close(f)


def plot_not_self_corrected(cat_mag, cat_mag_corrected, cat_mjd, ind, ind_ref, o):

    def set_figure():
        plt.rcdefaults()
        return plt.subplots(len(ind_ref[0:param['nsel_plots']])+1, 1,
                            figsize=(3, 3*(len(ind_ref[0:param['nsel_plots']])+1)), sharex=True)

    def plot(ii, iind, title):
        ax[ii].set_title(title)
        if param['field_name'] == 'lsi61303':
            std = np.std(y[:, iind])
            avg_set1 = np.average([yi for i, yi in enumerate(y[:, iind]) if x[i] < 56400])
            avg_set2 = np.average([yi for i, yi in enumerate(y[:, iind]) if 56400 <= x[i] < 56800])
            avg_set3 = np.average([yi for i, yi in enumerate(y[:, iind]) if 56800 < x[i]])
            ax[ii].errorbar(x, y[:, iind], yerr=dy[:, iind], fmt='k.', markersize=8, elinewidth=1.0, capsize=0,
                            label='std = {:.3f}; m1 = {:.3f}, m2 = {:.3f}, m3 = {:.3f}'.format(std, avg_set1, avg_set2, avg_set3))
            ax[ii].legend()
        else:
            ax[ii].errorbar(x, y[:, iind], yerr=dy[:, iind], fmt='k.', markersize=8, elinewidth=1.0, capsize=0)
        ax[ii].set_xlabel(r'MJD')
        ax[ii].set_ylabel(r'$m$ (mag)')
        ax[ii].set_xlim((min(x)*(1-0.00005), max(x)*(1+0.00005)))
        ax[ii].set_ylim((min(y[:, iind])*(1-0.005), max(y[:, iind])*(1+0.005)))
        ax[ii].set_ylim(ax[ii].get_ylim()[::-1])

    def compute_x_y_dy():
        nstars = len(cat_mag_corrected[0][0, :])
        avg_mjd = np.asarray([np.average(mjd) for mjd in cat_mjd])
        avg_nightly_mag = np.asarray([[np.average(mag[:, i]) for i in xrange(nstars)] for mag in cat_mag_corrected])
        std_nightly_mag = np.asarray([[np.std(mag[:, i]) for i in xrange(nstars)] for mag in cat_mag_corrected])
        return avg_mjd, avg_nightly_mag, std_nightly_mag

    x, y, dy = compute_x_y_dy()
    f, ax = set_figure()
    plot(0, ind, 'Target')
    for ir in xrange(len(ind_ref[0:param['nsel_plots']])):
        cat_mag_corrected = extinction_correction_ref(cat_mag[:], ind, ir, np.copy(ind_ref))
        x, y, dy = compute_x_y_dy()
        plot(ir+1, ind_ref[ir], 'Comparison star {} ({})'.format(ir+1, ind_ref[ir]))
    f.savefig(o+'/MJD-'+param['field_name'] + '-ref_stars_not_self_corrected.eps', bbox_inches='tight', pad_inches=0.05)
    plt.close(f)


def compute_differential_photometry(cat_ra, cat_dec, cat_mag, cat_mjd, o):
    ind = find_target(cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    bool_sel = np.ones(len(cat_ra[0][0, :]), dtype=bool)
    bool_sel = distance_selection(cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'], bool_sel)
    bool_sel = magnitude_selection(cat_mag[0][:, :], ind, param['mmin'], param['mmax'], bool_sel)
    cat_mag_corrected, ind_ref = extinction_correction(cat_mag[:], ind, bool_sel, param['nsel'])
    ref_stars_info(ind_ref, cat_mag_corrected[0][:, :], cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    plot_self_corrected(cat_mag_corrected, cat_mjd, ind, ind_ref, o+'/MJD-'+param['field_name']+'-ref_stars_self_corrected.eps')
    plot_not_self_corrected(cat_mag[:], cat_mag_corrected[:], cat_mjd, ind, ind_ref, o)
    return cat_mag_corrected, ind, ind_ref


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')
