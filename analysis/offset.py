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


def compute_final_magnitudes(cat_mag, ind, frame_list):
    nightly_avg_mag = [np.average(mag[:, ind]) for mag in cat_mag]
    nightly_std_mag = [np.std(mag[:, ind]) for mag in cat_mag]

    mag = [mag[:, ind] for mag in cat_mag]
    mag_list = []
    [mag_list.extend(m) for m in mag]

    frames_per_night = [len(mag[:, ind]) for mag in cat_mag]
    std_list = []
    [std_list.extend([nightly_std_mag[k]]*frames_per_night[k]) for k in xrange(len(frames_per_night))]
    if 1 == 0:
        print ''
        print len(frame_list)
        for i in range(len(nightly_std_mag)):
            print (i, nightly_std_mag[i], frame_list[i][0])
        print 'Average: {:.2f} +- {:.2f} mag'.format(nightly_avg_mag[25], nightly_std_mag[25])
        print mag_list[25]
    return mag_list, std_list, nightly_avg_mag, nightly_std_mag


def compute_final_date(cat_mjd):
    mjd_list = []
    [mjd_list.extend(mjd) for mjd in cat_mjd]
    mjd = np.asarray([np.average(mjd) for mjd in cat_mjd])
    return mjd, mjd_list


def add_offset(cat_mag, cat_mjd, ind, frame_list):
    target_obs_mag = param['zero_magnitude']
    target_ins_mag = compute_averaged_mag(cat_mag, ind)
    offset = target_ins_mag-target_obs_mag
    cat_mag = apply_offset(cat_mag, offset)
    mag_list, std_list, nightly_avg_mag, nightly_std_mag = compute_final_magnitudes(cat_mag, ind, frame_list)
    mjd, mjd_list = compute_final_date(cat_mjd)
    return cat_mag, mag_list, std_list, nightly_avg_mag, nightly_std_mag, mjd, mjd_list


if __name__ == '__main__':
    # Testing
    print('STOP: Testing should be done from analysis.py')
