# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import sys
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import itertools
param = {}
execfile(sys.argv[1])


def find_target(ra, dec, ra0, dec0, fl, testing=0):
    tar = SkyCoord(ra0, dec0, unit=(u.hour, u.deg))
    cat = SkyCoord(ra*u.degree, dec*u.degree)
    ind, sep2d, dist3d = tar.match_to_catalog_sky(cat)
    if sep2d.arcsec > param['astrometric_tolerance']:
        print("\n  Target not found:\n"
              "  - Target nominal coordinates: {} {}\n"
              "  - Extracted coordinates: {} {}\n"
              "  - Angular separation: {} arcsec\n"
              "  - Frame path:\n  {}".format(ra0, dec0, ra[ind], dec[ind], sep2d.arcsec, fl))
        return False
    return True


def perform_match(cat_ra, cat_dec, cat_mag, cat_mjd, frame_list, testing=0, loop_number=0):
    repeated_stars = 0

    def test(cat_test):
        nstars_max = max([max([len(frame) for frame in night]) for night in cat_test])
        nstars_min = min([min([len(frame) for frame in night]) for night in cat_test])
        print '\n  Loop number: {}'.format(loop_number)
        print '  - Repeated stars identifier: {}'.format(repeated_stars)
        print '  - Number of nights: {}'.format(np.asarray(cat_test).shape)
        print '  - Number of frames in the first night: {}'.format(np.asarray(cat_test[0]).shape)
        print '  - Number of stars in the reference frame: {}'.format(np.asarray(cat_test[0][0]).shape)
        print '  - Maximum number of stars: {}'.format(nstars_max)
        print '  - Minimum number of stars: {}'.format(nstars_min)
        if loop_number == 2 and repeated_stars != 0:
            raise RuntimeError("Duplicated stars after source_match.py")

    # Make some tests
    if testing == 1 and loop_number == 0:
        test(cat_ra[:])

    tol = param['astrometric_tolerance']
    # Convert from arcsec to deg
    tol /= 3600.

    # Creates the catalog of the reference frame (first frame of the first night)
    cat1 = SkyCoord(cat_ra[0][0]*u.degree, cat_dec[0][0]*u.degree)
    nstars_in_cat1 = len(cat_ra[0][0])

    # Array to flag the matched stars
    match_flag = np.ones(nstars_in_cat1, dtype='bool')

    # Iterate over each night
    night_mask = []
    counter = 0
    nnights = len(cat_ra)
    n = 0
    print ''
    for ra, dec, mag, mjd, fl in itertools.izip(cat_ra, cat_dec, cat_mag, cat_mjd, frame_list):
        sys.stdout.write('\r  Computing night {} of {}'.format(n+1, nnights))
        sys.stdout.flush()
        # Iterates over each frame of each night
        img_mask = []
        for i in xrange(len(ra)):
            # Extend the lists if needed
            nstars_in_cat2 = len(ra[i])
            if nstars_in_cat1 > nstars_in_cat2:
                ra[i] = np.append(ra[i], np.array([0]*(nstars_in_cat1-nstars_in_cat2)))
                dec[i] = np.append(dec[i], np.array([0]*(nstars_in_cat1-nstars_in_cat2)))
                mag[i] = np.append(mag[i], np.array([0]*(nstars_in_cat1-nstars_in_cat2)))

            # Creates the catalog of the current frame
            cat2 = SkyCoord(ra[i]*u.degree, dec[i]*u.degree)

            # Match the target nominal position with the current image
            found = False
            if find_target(ra[i], dec[i], param['ra'], param['dec'], fl[i]):
                img_mask.extend([i])
                found = True

            # Match the reference frame with the current catalog
            index, sep2d, dist3d = cat1.match_to_catalog_sky(cat2)
            if len(index) != len(set(index)):
                repeated_stars = 1

            # Sort the stars of the current catalog with the same order of the reference frame
            ra[i] = ra[i][index]
            dec[i] = dec[i][index]
            mag[i] = mag[i][index]

            # Flag the stars with a wrong match unless the frame has been already discarded
            if found:
                wrong_match = [j for (j, d) in enumerate(sep2d.value) if d > tol]
                if len(wrong_match)/len(sep2d) > param['max_nstars_missmatch_tolerance']:
                    # Discard bad frame
                    img_mask.remove(i)
                else:
                    # Discard the star
                    match_flag[wrong_match] = False

        # Remove miss matched frames
        counter += len(ra)-len(img_mask)
        if len(img_mask) < param['min_frames_per_night']:
            night_mask.extend([n])
        else:
            cat_ra[n] = np.asarray(cat_ra[n])[img_mask]
            cat_dec[n] = np.asarray(cat_dec[n])[img_mask]
            cat_mag[n] = np.asarray(cat_mag[n])[img_mask]
            cat_mjd[n] = np.asarray(cat_mjd[n])[img_mask]
            frame_list[n] = np.asarray(frame_list[n])[img_mask]
        n += 1

    # Remove wrong nights
    if len(night_mask) != 0:
        print '\n\n  Discarding the nights: {}'.format(night_mask)
    for (i, bad_night) in enumerate(night_mask):
        del cat_ra[bad_night-i]
        del cat_dec[bad_night-i]
        del cat_mag[bad_night-i]
        del cat_mjd[bad_night-i]
        del frame_list[bad_night-i]
    print '  A total of {} images have been discarded'.format(counter)
    print '  A total of {}/{} nights have been removed'.format(len(night_mask), nnights)

    print ''

    # Creates the output lists
    cat_ra_sorted = []
    cat_dec_sorted = []
    cat_mag_sorted = []

    # Discard the stars with a wrong match
    for ra, dec, mag in itertools.izip(cat_ra, cat_dec, cat_mag):
        ra_aux = []
        dec_aux = []
        mag_aux = []
        for i in xrange(len(ra)):
            ra_aux.append(np.asarray(ra[i][match_flag]))
            dec_aux.append(np.asarray(dec[i][match_flag]))
            mag_aux.append(np.asarray(mag[i][match_flag]))
        cat_ra_sorted.append(np.asarray(ra_aux))
        cat_dec_sorted.append(np.asarray(dec_aux))
        cat_mag_sorted.append(np.asarray(mag_aux))

    # Make some tests
    loop_number += 1
    test(cat_ra_sorted[:])

    if testing == 1 and loop_number == 1:
        cat_ra_sorted, cat_dec_sorted, cat_mag_sorted, cat_mjd, frame_list = perform_match(cat_ra_sorted, cat_dec_sorted, cat_mag_sorted, cat_mjd, frame_list,
                                                                      testing=1, loop_number=loop_number)
    return cat_ra_sorted, cat_dec_sorted, cat_mag_sorted, cat_mjd, frame_list


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')

