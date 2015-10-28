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

def find_stars(ra, dec, stars, testing=0):
    ind_list = []
    for star in stars:
        ra0 = star[0]
        dec0 = star[1]
        tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
        cat = SkyCoord(ra*u.degree, dec*u.degree)
        ind, sep2d, dist3d = tar.match_to_catalog_sky(cat)
        ind_list.append(int(ind))
        if sep2d.arcsec > param['astrometric_tolerance']:
            raise RuntimeError("Reference star not found")
        if testing == 1:
            ra = np.delete(ra, ind)
            dec = np.delete(dec, ind)
            ind = find_target(ra, dec, ra0, dec0, testing=0)
        # print 'Index value: {}'.format(ind)
    # print ind_list
    assert len(stars) == len(ind_list), 'Could not find reference stars'
    return ind_list


def distance_selection(ra, dec, ra0, dec0, dmax, bool_sel, testing=1):

    tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
    cat = SkyCoord(ra*u.degree, dec*u.degree)
    sep2d = tar.separation(cat)

    if testing == 1:
        print '\n  Before DISTANCE selection: (all distances)'
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


def magnitude_selection(mag, mag_target, sel, mmin, mmax, testing =1):

    if testing == 1:
        print '\n  Before MAGNITUDE selection: (all magnitudes)'
        print '  - Minimum magnitude value: {:.2f}'.format(min(mag[sel]))
        print '  - Maximum magnitude value: {:.2f}'.format(max(mag[sel]))
        print '  - Number of stars: {}'.format(len(mag[sel]))

    sel[mag > mag_target+mmax] = False
    sel[mag < mag_target-mmin] = False

    if testing == 1:
        print '\n  After MAGNITUDE selection such as m in [mtar-mmin, mtar+mmax]:'
        print '  - Maximum magnitude difference: (mmin, mmax) = ({}, {})'.format(mmin, mmax)
        print '  - Minimum magnitude value: {:.2f}'.format(min(mag[sel]))
        print '  - Target mean magnitude: {:.2f}'.format(mag_target)
        print '  - Maximum magnitude value: {:.2f}'.format(max(mag[sel]))
        print '  - Number of stars: {}'.format(len(mag[sel]))
    return sel


def correct_magnitudes(cm, m0, ind_ref, w, testing=0):
    ns = len(ind_ref)
    for (i, m) in enumerate(cm):
        # Number of frames
        nf = len(m[:, 0])

        # Nightly weights instead of long-term weights given by w parameter
        #w = np.asarray([1./np.std(m[:, ind_ref[k]], axis=0)**2 for k in xrange(ns)])

        # Computing the correcting factor for each frame using the average variation of ns stars
        #dm = np.average([m[:, ind_ref[k]]-m0[ind_ref[k]] for k in xrange(ns)], axis=0, weights=w)
        dm = np.average([m[:, k]-m0[k] for k in ind_ref], axis=0, weights=w)

        # Applying the correcting factor to each frame
        cm[i] = np.array([m[j, :]-dm[j] for j in xrange(nf)])

    if testing == 1:
        ii = w.argsort()[::-1]
        ind_ref = ind_ref[ii]
        avg_nightly_m = np.asarray([[np.average(mag[:, i]) for i in ind_ref] for mag in cm])
        AVG_m = np.asarray([np.average(avg_nightly_m[:, i]) for i in xrange(len(ind_ref))])
        print '\nWeights:\n',w
        print 'Average magnitude:\n',AVG_m
        print 'Reference stars:\n', ind_ref
        print '(is it ok compared with the RMS plot?)'

    return cm


def compute_std(cat_mag_corrected, it, ind, ns, ind_ref):
    nstars = len(cat_mag_corrected[0][0, :])
    avg_nightly_m = np.asarray([[np.average(mag[:, i]) for i in xrange(nstars)] for mag in cat_mag_corrected])
    std_m = np.asarray([np.std(avg_nightly_m[:, i]) for i in xrange(nstars)])
    AVG_m = np.asarray([np.average(avg_nightly_m[:, i]) for i in xrange(nstars)])

    # WRONG, std_m2 does not account for night to night variations
    # std_m2 = np.average([np.std(mag[:, :], axis=0) for mag in cat_mag_corrected], axis=0)

    print '\n  After correcting loop {}:'.format(it)
    print '  - Number of stars used for correction {} of {} stars'.format(ns, nstars)
    print '  - Labeled with: {}'.format(ind_ref)
    print '  - And std of {}'.format(std_m[ind_ref])
    print '  - Multi night target STD of the averaged mag is of {:.4f}'.format(std_m[ind])
    print '  - Target MEAN magnitude is of {:.3f}\n'.format(np.average(avg_nightly_m[:, ind]))
    return avg_nightly_m, AVG_m, std_m


def residual(params, x, data=None):
    parvals = params.valuesdict()
    model = np.abs(parvals['a'])*np.ones_like(x) + np.abs(parvals['b'])*10**(parvals['c']*x/2.5) +np.abs(parvals['d'])*10**(parvals['e']*x/2.5)
    if data is None:
        return model
    return (model - data)


def parab_select(mag, std, sel, it):
    std_aux = std[:]

    from lmfit import minimize, Parameters, Parameter, report_fit

    params = Parameters()
    params.add('a', value= 0.003)#, min=0.003, max = 0.015)
    params.add('b', value= 1e-12)#, min =1.0e-12, max=1e-7)
    params.add('c', value= 0.5)#,  min =0.3, max =0.7)
    params.add('d', value= 1e-9)#, min =1e-11, max=1e-7)
    params.add('e', value= 1.2)#,  min =0.8, max =1.6)

    out = minimize(residual, params, args=(mag[std < 999999], std[std < 999999]))
    res = residual(params, mag, std)
    sel[abs(res)>3.0*np.std(residual(params, mag[std < 999999], std[std < 999999]))] = False

    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(mag[sel], std[sel], fmt='ok', mec='k', markersize=4, label='Selected stars')
    ax.errorbar(mag[~sel], std_aux[~sel],fmt='or', mec='r', markersize=4, label='Discarded stars')
    mag_fit = np.linspace(np.sort(mag)[0], np.sort(mag)[-1], 500)
    ax.errorbar(mag_fit, residual(params, mag_fit), label='Fitted error',fmt='-b')
    ax.set_xlabel(r'$\overline{m}$ (mag)')
    ax.set_ylabel(r'$\sigma$')
    ax.set_ylim((std[std<9999].min(), std[std<9999].max()))
    ax.set_yscale('log')
    fig.savefig(param['output_path'] + '/RMSvsMAG/ref_stars_selection_{}.eps'.format(it), bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return sel


def extinction_correction(cat_mag, ind, ind_ref, ra, dec, ra0, dec0,):

    # Boolean array for labeling the preselected reference stars
    sel = np.zeros_like(ra, dtype=bool)
    sel[ind_ref] = True

    # First correction equal weights
    mag0 = cat_mag[0][0, :]
    w = np.ones_like(ind_ref)
    cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref, w)
    avg_m, AVG_m, std_m = compute_std(cat_mag_corrected, 1, ind, len(ind_ref), ind_ref)

    # Second correction weighted with the std
    # mag0 = cat_mag[0][0, :]
    # w = 1./std_m[ind_ref]**2.
    # cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref, w)
    # avg_m, AVG_m, std_m = compute_std(cat_mag_corrected, 2, ind, len(ind_ref), ind_ref)

    print '------'
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar(AVG_m[np.where(sel == False)], std_m[np.where(sel == False)], fmt='or', mec='r', markersize=4)
    ax.errorbar(AVG_m[np.where(sel == True)], std_m[np.where(sel == True)], fmt='ok', mec='g', markersize=4)
    ax.errorbar(AVG_m[ind], std_m[ind], fmt='*b', mec='b', markersize=8, linewidth=0,)
    ax.set_xlabel(r'$\overline{m}$ (mag)')
    ax.set_ylabel(r'$\sigma$')
    ax.set_yscale('log')
    # ylim1max = std_m_aux[std_m_aux<99999].max()
    # ylim2max = std_m[std_m<99999].max()
    # ylim1min = std_m_aux.min()
    # ylim2min = std_m.min()
    # ax.set_ylim((min(ylim1min, ylim2min), max(ylim1max,ylim2max)))
    fig.savefig(param['output_path'] + '/RMSvsMAG/RMSplot.eps', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    return cat_mag_corrected, ind_ref, 1./std_m[ind_ref]**2.


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


def extinction_correction_ref(cat_mag, ind, ir, ind_ref, w, testing=1):
    nref = len(ind_ref)
    ir_label = ind_ref[ir]
    ind_ref = np.delete(ind_ref, ir)
    mag0 = cat_mag[0][0, :]
    w = np.delete(w, ir)

    cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref, w, 0)
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


def plot_not_self_corrected(cat_mag, cat_mag_corrected, cat_mjd, ind, ind_ref, o, w):

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
        cat_mag_corrected = extinction_correction_ref(cat_mag[:], ind, ir, np.copy(ind_ref), w)
        x, y, dy = compute_x_y_dy()
        plot(ir+1, ind_ref[ir], 'Comparison star {} ({})'.format(ir+1, ind_ref[ir]))
    f.savefig(o+'/MJD-'+param['field_name'] + '-ref_stars_not_self_corrected.eps', bbox_inches='tight', pad_inches=0.05)
    plt.close(f)


def compute_differential_photometry(cat_ra, cat_dec, cat_mag, cat_mjd, o):
    ind = find_target(cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    ind_ref = find_stars(cat_ra[0][0, :], cat_dec[0][0, :], eval(param['reference_stars']))
    cat_mag_corrected, ind_ref, w = extinction_correction(cat_mag[:], ind, ind_ref, cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    # ref_stars_info(ind_ref, cat_mag_corrected[0][:, :], cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    # plot_self_corrected(cat_mag_corrected, cat_mjd, ind, ind_ref, o+'/MJD-'+param['field_name']+'-ref_stars_self_corrected.eps')
    # plot_not_self_corrected(cat_mag[:], cat_mag_corrected[:], cat_mjd, ind, ind_ref, o, w)
    return cat_mag_corrected, ind, ind_ref


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')
