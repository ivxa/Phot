# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import sys
param = {}
execfile(sys.argv[1])

def find_stars(ra, dec, stars, testing=0):
    ind_list = []
    for star in stars:
        ra0 = star[0]
        dec0 = star[1]
        cat = SkyCoord(ra*u.degree, dec*u.degree)
        # tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
        tar = SkyCoord(ra0*u.hour, dec0*u.degree)#degree)
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

    w = np.array(eval(param['weights']))
    ind_list = np.array(ind_list)
    ind_list = ind_list[w.argsort()]#[::-1]
    assert len(stars) == len(ind_list), 'Could not find reference stars'
    assert len(w) == len(ind_list), 'len(w) != nstars'
    return ind_list, w

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
    print '  - Labeled with (the first 10 of them): {}'.format(ind_ref[0:10])
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


def ref_star_selection(cat_mag, ind, bool_sel, nsel, ra, dec, ra0, dec0,):
    # We want 2 comparison stars
    nsel += 2

    # First correction limiting by magnitude (mmin0 and mmax0)
    mag0 = cat_mag[0][0, :]

    # First magnitude selection
    print '\nITERATION 0:'
    sel0 = np.ones_like(mag0, dtype=bool)
    sel0[mag0 > mag0[ind]+param['mmax']] = False
    sel0[mag0 < mag0[ind]-param['mmin']] = False
    sel0[ind] = False
    ind_ref = np.where(sel0 == True)[0]
    cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref, np.ones_like(ind_ref))
    print '------'

    # Many corrections
    it = 0
    while True:
        it += 1
        print '\nITERATION {}:'.format(it)
        ind_ref_aux = np.copy(ind_ref)

        # Compute the standard deviation of the nightly averages
        avg_m, AVG_m, std_m = compute_std(cat_mag_corrected, it, ind, len(ind_ref), ind_ref)
        std_m_aux = np.copy(std_m)
        sel = np.ones_like(AVG_m, dtype=bool)

        # Remove the target from the comparison star candidates
        std_target = std_m[ind]
        std_m[ind] = 9999999

        # Remove less suitable comparison stars based on distance criteria
        std_m[np.where(bool_sel == False)] = 9999999

        # Select the comparison stars based on parabola fitting
        if param['disable_parab_fit'] == 0:
            sel = parab_select(AVG_m, std_m, sel, it)
            std_m[np.where(sel == False)] = 9999999

        # Magnitude selection
        sel = magnitude_selection(AVG_m, AVG_m[ind], sel, param['mmin'], param['mmax'])
        std_m[np.where(sel == False)] = 9999999

        # Distance selection using dmax_final (dmax_final < dmax; after the parabola has been built)
        sel = distance_selection(ra, dec, ra0, dec0, param['dmax_final'], sel)
        std_m[np.where(sel == False)] = 9999999

        # Apply correction only using the selected stars
        if nsel > len(std_m[std_m < 999999]):
            raise RuntimeError("Not enough reference stars")
        ind_ref = std_m.argsort()[0:nsel]

        print '\nStar label {}, std: {}'.format(ind_ref[0], std_m[ind_ref[0]])
        print 'Star label {}, std: {}'.format(ind_ref[1], std_m[ind_ref[1]])
        print 'Star label {}, std: {}'.format(ind_ref[2], std_m[ind_ref[2]])
        print 'Star label {}, std: {}'.format(ind_ref[3], std_m[ind_ref[3]])

        mag0 = cat_mag[0][0, :]
        cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref, 1./std_m[ind_ref]**2.)

        print '------'
        if np.array_equal(np.sort(ind_ref), np.sort(ind_ref_aux)):
            ind_ref = ind_ref[::-1]
            return ind_ref, 1./std_m[ind_ref]**2.


def ref_stars_info(ii, i, mag, ra, dec, ra0, dec0, w):
    tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
    cat = SkyCoord(ra[i]*u.degree, dec[i]*u.degree)
    d = tar.separation(cat)
    o = param['output_path']+'data/S{}_info'.format(str(ii))
    np.savetxt(o, np.array([d.arcminute, ra[i], dec[i], w]), delimiter=' ')


def comp_stars_info(ii, i, ra, dec, ra0, dec0):
    tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
    cat = SkyCoord(ra[i]*u.degree, dec[i]*u.degree)
    d = tar.separation(cat)
    o = param['output_path']+'data/S{}_comp_info'.format(str(ii))
    np.savetxt(o, np.array([d.arcminute, ra[i], dec[i]]), delimiter=' ')


def pick_comparison_stars(ind_ref, w):
    return ind_ref[2:], w[2:], ind_ref[1], ind_ref[0] # ind_ref is sorted in increasing weight


def compute_differential_photometry(cat_ra, cat_dec, cat_mag, cat_mjd, o):
    ind = find_target(cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])

    if param['auto_sel']:
        bool_sel = np.ones(len(cat_ra[0][0, :]), dtype=bool)
        bool_sel = distance_selection(cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'], param['dmax'], bool_sel)
        ind_ref, w = ref_star_selection(cat_mag[:], ind, bool_sel, param['nsel'], cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    else:
        ind_ref, w = find_stars(cat_ra[0][0, :], cat_dec[0][0, :], eval(param['reference_stars']))

    ind_ref, w, ind_comp1, ind_comp2 = pick_comparison_stars(ind_ref, w)
    comp_stars_info(1, ind_comp1, cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    comp_stars_info(2, ind_comp2, cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'])
    cat_mag_corrected_list = []
    ind_ref_list = []
    ind_comp_list = []
    for i in range(len(ind_ref)):
        ind_excluded = ind_ref[i]
        ind_ref_aux = np.array([v for (k,v) in enumerate(ind_ref) if k != i])
        w_aux = np.array([v for (k,v) in enumerate(w) if k != i])
        mag0 = cat_mag[0][0, :]
        cat_mag_corrected = correct_magnitudes(cat_mag[:], mag0[:], ind_ref_aux, w_aux)
        cat_mag_corrected_list.append(cat_mag_corrected)
        ind_ref_list.append(ind_ref_aux)
        ind_comp_list.append(ind_excluded)
        ref_stars_info(i, ind_excluded, cat_mag_corrected[0][:, :], cat_ra[0][0, :], cat_dec[0][0, :], param['ra'], param['dec'], w[i])
    assert len(ind_ref_list) == len(ind_ref)
    assert len(cat_mag_corrected_list) == len(ind_ref)
    return cat_mag_corrected_list, ind, ind_ref_list, ind_comp_list, ind_comp1, ind_comp2


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')
