# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import numpy as np
import os
import shutil
import source_extraction
import quality_control
import source_match
import multi_night_std_test
import differential_photometry
# import xy_fit
# import detrending
import offset
import sys
param = {}
execfile(sys.argv[1])


def make_output_folder(o):
    def check_and_make(folder_name):
        if os.path.exists(o+folder_name):
            shutil.rmtree(o+folder_name)
        os.makedirs(o+folder_name)

    if not os.path.exists(o):
        os.makedirs(o)
        os.makedirs(o+'nightly_LC')
        os.makedirs(o+'multi_night_LC')
        os.makedirs(o+'std_multi_night_plots')
        os.makedirs(o+'RMSvsMAG')
        os.makedirs(o+'check_images')
    else:
        check_and_make('nightly_LC')
        check_and_make('multi_night_LC')
        check_and_make('std_multi_night_plots')
        check_and_make('RMSvsMAG')
        check_and_make('data')
        check_and_make('check_images')       


def load_frame_list(f):
    fl = sorted(np.genfromtxt(f, dtype='str', comments='#'))
    if len(fl) != len(set(fl)):
        raise RuntimeError("Duplicated frames in {}".format(f))

    frame_list = []
    for f in [d for d in sorted(os.listdir(param['data_path'])) if d[:2] == '20']:
        frame_set = []
        [frame_set.extend([frame]) for frame in fl if f in frame]
        if frame_set != []:
            frame_list.append(frame_set)
    return sorted(frame_list)


def save_data(mjd, mjd_list, mag_list, std_list, nightly_avg_mag, nightly_std_mag, night_numbering_list, o, testing=1):
    if testing == 1:
        print '\n  len(mjd_list): {}'.format(len(mjd_list))
        print '  len(mag_list): {}'.format(len(mag_list))
        print '  len(std_list): {}'.format(len(std_list))
        print '\n  mjd.shape: {}'.format(np.asarray(mjd).shape)
        print '  nightly_avg_mag.shape: {}'.format(np.asarray(nightly_avg_mag).shape)
        print '  nightly_std_mag.shape: {}\n'.format(np.asarray(nightly_std_mag).shape)
    np.savetxt(o+'MJD_MAG_ERR-{}-all_frames.dat'.format(param['field_name']),
               np.transpose((mjd_list, mag_list, std_list, night_numbering_list)), delimiter=' ')
    np.savetxt(o+'MJD_MAG_ERR-{}-nightly_average.dat'.format(param['field_name']),
               np.transpose((mjd, nightly_avg_mag, nightly_std_mag, np.array(range(1,len(mjd)+1)))), delimiter=' ')


def copy_param_files(f, o):
    shutil.copyfile(f, o+'param.py')
    shutil.copyfile('se.sex', o+'se.sex')
    shutil.copyfile('se.param', o+'se.param')


def analyze_data():
    output_path = param['output_path']
    field_name = param['field_name']
    print('Making the output folder...'),
    make_output_folder(output_path)
    print('OK\nLoading the frame list...'),
    frame_list = load_frame_list(param['frame_list'])
    print('OK\nMaking the star catalog from each image of the frame list...\n')
    cat_ra, cat_dec, cat_mag, cat_mjd = source_extraction.perform_extraction(param['data_path'], frame_list)
    print('OK\nPerforming quality control selection (number of detections)...')
    cat_ra, cat_dec, cat_mag, cat_mjd, frame_list = quality_control.median_number_of_stars_qc(cat_ra, cat_dec, cat_mag, cat_mjd, frame_list)
    print('OK\nMatching the stars between catalogs...')
    cat_ra, cat_dec, cat_mag, cat_mjd, frame_list = source_match.perform_match(cat_ra, cat_dec, cat_mag, cat_mjd, frame_list)
    multi_night_std_test.perform_test(cat_mag, output_path+'std_multi_night_plots/std_{}_multi_night_01_qc.eps'.format(field_name))
    print('\nOK\nComputing the differential photometry...\n')
    cat_mag, ind, ind_ref, ind_comp, ind_comp1, ind_comp2 = differential_photometry.compute_differential_photometry(cat_ra, cat_dec, cat_mag, cat_mjd, output_path+'multi_night_LC/')
    multi_night_std_test.perform_test(cat_mag[0], output_path+'std_multi_night_plots/std_{}_multi_night_02_qc-diff.eps'.format(str('0'), field_name))
    for (i, cat_mag_i) in enumerate(cat_mag):
        print('OK\nPerforming a quality control using photometric catalogs...'),
        # cat_ra, cat_dec, cat_mag_i, cat_mjd, frame_list = quality_control.photometric_qc(cat_ra, cat_dec, cat_mag_i, cat_mjd, frame_list)
        # multi_night_std_test.perform_test(cat_mag_i, output_path+'std_multi_night_plots/std_{}_multi_night_02_qc-diff-qc-.eps'.format(field_name))
        print('\nOK\nAplying xy fit correction...'),
        # cat_mag_i = xy_fit.fit_residuals(cat_ra, cat_dec, cat_mag_i)
        # multi_night_std_test.perform_test(cat_mag_i, output_path+'std_multi_night_plots/std_{}_multi_night_03_qc-diff-qc-xy.eps'.format(field_name))
        print('OK\nDetrending the light curves...'),
        # cat_mag_i = detrending.detrend_light_curve(cat_ra, cat_dec, cat_mag_i)
        # multi_night_std_test.perform_test(cat_mag_i, output_path+'std_multi_night_plots/std_{}_multi_night_04_qc-diff-qc-xy-det.eps'.format(field_name))
        for (k, ii) in enumerate([int(ind), int(ind_comp1), int(ind_comp2), int(ind_comp[i])]):
            if k == 0:
                fname = 'target'
                offset_value = 0.
                compute_offset = True
            elif k == 1:
                fname = 'compa1'
                compute_offset = False
            elif k == 2:
                fname = 'compa2'
                compute_offset = False
            else:
                fname = 'refere'
                compute_offset = False
            print('OK\nApplying an artificial offset...'),
            cat_mag_i, mag_list, std_list, nightly_avg_mag, nightly_std_mag, mjd, mjd_list, night_numbering_list, offset_value = offset.add_offset(cat_mag_i, cat_mjd, ii, offset_value, compute_offset)
            print('OK\nSaving the data files of the light curves...')
            save_data(mjd, mjd_list, mag_list, std_list, nightly_avg_mag, nightly_std_mag, night_numbering_list, output_path+'data/S{}_{}_'.format(str(i),fname))
        multi_night_std_test.perform_test(cat_mag_i, output_path+'std_multi_night_plots/S{}_std_{}_multi_night_02_qc-diff.eps'.format(str(i), field_name), ind, ind_comp[i], ind_ref[i], ind_comp1, ind_comp2)
    print('OK\nCopying the setup input file...')
    np.savetxt(output_path+'data/nstars', np.array([len(cat_mag)]), fmt='%i')
    copy_param_files(sys.argv[1], output_path+'data/')
    print 'OK\n'


if __name__ == '__main__':
    # Testing
    analyze_data()
    print('DONE')
