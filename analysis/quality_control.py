# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
import sys
import matplotlib.pyplot as plt
param = {}
execfile(sys.argv[1])


def plot_nstars(cat_ra, cat_mjd, suff):
    nstars = [max([len(frame) for frame in ra]) for ra in cat_ra]
    mjd = [np.average(mjd) for mjd in cat_mjd]
    plt.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mjd, nstars, '.')
    ax.set_xlabel('MJD')
    ax.set_ylabel('Maximum number of stars per night')
    fig.savefig(param['output_path']+'maximum_number_of_stars_per_night'+suff, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def remove_individual_frames(cat_ra, cat_dec, cat_mag, cat_mjd, max_stars, frame_list_grouped):
    nnights = len(cat_ra)
    night_mask = []
    counter = 0

    for (i, night) in enumerate(cat_ra):
        img_mask = np.asarray([j for (j, n) in enumerate([len(c) for c in night])
                               if n > max_stars*param['nstars_tolerance']])
        counter += len(night)-len(img_mask)
        if len(img_mask) < param['min_frames_per_night']:
            night_mask.extend([i])
        else:
            cat_ra[i] = np.asarray(cat_ra[i])[img_mask]
            cat_dec[i] = np.asarray(cat_dec[i])[img_mask]
            cat_mag[i] = np.asarray(cat_mag[i])[img_mask]
            cat_mjd[i] = np.asarray(cat_mjd[i])[img_mask]
            frame_list_grouped[i] = np.asarray(frame_list_grouped[i])[img_mask]
    if len(night_mask) != 0:
        print '  Discarding the nights: {}'.format(night_mask)
    for (i, bad_night) in enumerate(night_mask):
        del cat_ra[bad_night-i]
        del cat_dec[bad_night-i]
        del cat_mag[bad_night-i]
        del cat_mjd[bad_night-i]
        del frame_list_grouped[bad_night-i]
    print '  A total of {} images have been discarded'.format(counter)
    print '  A total of {}/{} nights have been removed'.format(len(night_mask), nnights)


def median_number_of_stars_qc(cat_ra, cat_dec, cat_mag, cat_mjd, frame_list_grouped, testing=0):
    plot_nstars(cat_ra, cat_mjd, '.eps')
    if testing == 1:
        print len(cat_ra), len(frame_list_grouped)
        print len(cat_ra[0]), len(frame_list_grouped[0])
        print frame_list_grouped[0][0], frame_list_grouped[0][-1]

    max_stars_per_night = [max([len(c) for c in night]) for night in cat_ra]
    max_stars = max(max_stars_per_night)
    med_stars = np.median(max_stars_per_night)
    print '\n  Maximum number of stars: {}'.format(max_stars)
    print '  Median number of stars: {}'.format(np.int(med_stars))
    print '  Minimum number of stars before frame selection: {}\n'\
        .format(min([min([len(frame) for frame in night]) for night in cat_ra]))

    remove_individual_frames(cat_ra, cat_dec, cat_mag, cat_mjd, max_stars, frame_list_grouped)
    print '  Minimum number of stars after frame selection: {}\n'\
        .format(min([min([len(frame) for frame in night]) for night in cat_ra]))
    plot_nstars(cat_ra, cat_mjd, '-qc.eps')
    return cat_ra, cat_dec, cat_mag, cat_mjd, frame_list_grouped


def photometric_qc(cat_ra, cat_dec, cat_mag, cat_mjd, frame_list):
    # Pick the 30 stars with lowest RMS at a distance < dmax_qc deg
    # Compute their median magnitude and std
    # Discard those frames for which more than qc_tol% of the picked stars
    # differ qc_sigma sigma times their median magnitude
    # Return repeat = True if any frame is discarded
    # Fine tune dmax_qc, qc_tol and qc_sigma parameters
    # While return = True; do matching stars, differential photometry and photometric_qc again
    # Keep track of the loop and add an index to the multi night std plot
    pass


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')