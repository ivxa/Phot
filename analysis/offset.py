# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
import sys
param = {}
execfile(sys.argv[1])


def compute_averaged_mag(cat_mag, ind):
    return np.average([np.average(mag[:, ind]) for mag in cat_mag])


def apply_offset(cat_mag, offset):
    return [mag-offset for mag in cat_mag]


def compute_final_magnitudes(cat_mag, ind):
    nightly_avg_mag = [np.average(mag[:, ind]) for mag in cat_mag]
    nightly_std_mag = [np.std(mag[:, ind]) for mag in cat_mag]

    night_numbering = [[i+1]*len(mag[:, 0]) for (i, mag) in enumerate(cat_mag)]
    night_numbering_list = []
    [night_numbering_list.extend(n) for n in night_numbering]

    mag = [mag[:, ind] for mag in cat_mag]
    mag_list = []
    [mag_list.extend(m) for m in mag]

    frames_per_night = [len(mag[:, ind]) for mag in cat_mag]
    std_list = []
    [std_list.extend([nightly_std_mag[k]]*frames_per_night[k]) for k in xrange(len(frames_per_night))]
    return mag_list, std_list, nightly_avg_mag, nightly_std_mag, night_numbering_list


def compute_final_date(cat_mjd):
    mjd_list = []
    [mjd_list.extend(mjd) for mjd in cat_mjd]
    mjd = np.asarray([np.average(mjd) for mjd in cat_mjd])
    return mjd, mjd_list


def add_offset(cat_mag, cat_mjd, ind, offset, compute_offset):
    if compute_offset:
        target_obs_mag = param['zero_magnitude']
        target_ins_mag = compute_averaged_mag(cat_mag, ind)
        offset = target_ins_mag-target_obs_mag
        cat_mag = apply_offset(cat_mag, offset)
    mag_list, std_list, nightly_avg_mag, nightly_std_mag, night_numbering_list = compute_final_magnitudes(cat_mag, ind)
    mjd, mjd_list = compute_final_date(cat_mjd)
    return cat_mag, mag_list, std_list, nightly_avg_mag, nightly_std_mag, mjd, mjd_list, night_numbering_list, offset


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')
