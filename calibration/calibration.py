# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
import shutil
import numpy as np
import crop
import astrometry
import nightly_std_test
import source_check
import linearity_map
import shutter_map
import bias_darks_flats
import sys
param = {}
execfile(sys.argv[1])


def make_folder_list(p):
    t = [p+d+'/' for d in os.listdir(p) if d[:2] == '20']
    t.sort()
    return t


def check_calibration(f):
    t = os.listdir(f)
    if 'cal_completed' not in t:
        return False
    return True


def check_data(f, n):
    t = os.listdir(f)
    if 'not_revised' in t:
        raise RuntimeError("Images not revised in {}".format(f))
    elif 'org_completed' not in t:
        raise RuntimeError("Organization not completed in {}".format(f))
    elif 'supermasterflat_link' not in t and param['disable_standard_cal'] == 0:
        raise RuntimeError("Flat images not found in {}".format(f))
    elif 'bias_no_cal' not in t and param['disable_standard_cal'] == 0:
        raise RuntimeError("Bias images not found in {}".format(f))
    elif 'shutter_map_link' not in t and param['disable_calibration_shutter'] == 0:
        raise RuntimeError("Shutter map not found in {}".format(f))
    # elif 'linearity_map_link' not in t and param['disable_calibration_lin'] == 0:
    #     raise RuntimeError("Linearity map not found in {}".format(f))
    if not os.path.exists(f+n+'/raw_no_cal/darks_no_cal') and param['disable_standard_cal'] == 0:
        raise RuntimeError("Dark images not found in {}".format(f+n+'/raw_no_cal/'))


def make_output_folders(f, n, o):
    """Make the following folder tree
       data_path/YYYYMMDD/phot/
       data_path/YYYYMMDD/phot/field/
       data_path/YYYYMMDD/phot/field/cal/
       data_path/YYYYMMDD/phot/field/tmp/"""

    def make_tmp_folders():
        os.makedirs(f+n+'/tmp')
        os.makedirs(f+n+'/tmp/flats')
        os.makedirs(f+n+'/tmp/bias')
        os.makedirs(f+n+'/tmp/darks')
        os.makedirs(f+n+'/tmp/science')
        os.makedirs(f+n+'/tmp/shutter')
        os.makedirs(f+n+'/tmp/linear')

    if not os.path.exists(f):
        os.makedirs(f)
        os.makedirs(f+n)
        os.makedirs(f+n+'/cal')
        make_tmp_folders()
    elif not os.path.exists(f+n):
        os.makedirs(f+n)
        os.makedirs(f+n+'/cal')
        make_tmp_folders()
    elif os.path.exists(f+n):
        shutil.rmtree(f+n)
        os.makedirs(f+n)
        os.makedirs(f+n+'/cal')
        make_tmp_folders()
    else:
        raise RuntimeError("Check data path")

    if not os.path.exists(o):
        os.makedirs(o)


def check_first_run(f):
    t = os.listdir(f)
    if 'not_first_run' in t:
        return False
    return True


def update_frame_list(frame_list_file, fp):
    t = [fp+d+'\n' for d in os.listdir(fp) if d[-5:] == '.fits']
    t.sort()
    with open(frame_list_file, "a+") as f:
        [f.write(frame) for frame in t]


def check_double_entries(f):
    frame_list = np.genfromtxt(f, dtype='str')
    if len(frame_list) != len(set(frame_list)):
        raise RuntimeError("Duplicated frames in {}".format(f))


def make_completed_file_flag(f):
    open(f, 'a').close()


def move_files(i, o):
    if (os.listdir(i+'bias/') and param['disable_standard_cal'] == 0
        and param['disable_bias'] == 0):
        shutil.copy2(i+'bias/master_bias.fits', o)
    if os.listdir(i+'darks/') and param['disable_standard_cal'] == 0:
        shutil.copy2(i+'darks/master_dark.fits', o)
    if os.listdir(i+'flats/') and (param['disable_normal_flat'] == 0 or param['disable_screen_flat'] == 0):
        shutil.copy2(i+'flats/master_flat.fits', o)
    if os.listdir(i+'science/'):
        [shutil.copy2(i+'science/'+f, o+'/cal/') for f in os.listdir(i+'science') if f[-5:] == '.fits']
        shutil.rmtree(i)


def calibrate_data():
    data_path = param['data_path']
    output_path = param['output_path']
    field_name = param['field_name']
    rerun = param['rerun']
    frame_list_file = param['frame_list']
    frame_list = sorted(np.genfromtxt(frame_list_file, dtype='str', comments='#'))
    fl = make_folder_list(data_path)
    count = 0
    for f in fl:
        if (os.path.exists(f+field_name) and
           (not check_calibration(f+field_name) or rerun == 1)):
            frame_list_cropped = [frame[:len(f)] for frame in frame_list]

            if not check_first_run(f+field_name) and f not in frame_list_cropped:
                count += 1
                print 'Skipped {}: {}'.format(count, f)
                continue
            else:
                print ""

            print('Image set: {}'.format(f[-9:-1]))
            print('Checking the data tree...'),
            check_data(f, field_name)
            print('OK\nMaking the output folder...'),
            make_output_folders(f+'phot/', field_name, output_path)
            print('OK\nCropping the images...'),
            crop.crop_data(f, f+'phot/'+field_name)
            print('OK\nSetting the WCS...'),
            astrometry.set_wcs(f+'phot/'+field_name+'/tmp/science/', f+'phot/'+field_name+'/tmp/science/wcs/')
            nightly_std_test.perform_test(f+'phot/'+field_name+'/tmp/science/',
                                          f+'phot/'+field_name+'/std_{}_{}_01_wcs.eps'.format(field_name, f[-9:-1]))
            print('OK\nSaturation and centering control...'),
            source_check.perform_control(f+'phot/'+field_name+'/tmp/science/')
            if param['disable_calibration_lin'] == 0:
                print('OK\nApplying non-linear correction...'),
                linearity_map.apply_correction(f+'phot/'+field_name+'/tmp/')
                nightly_std_test.perform_test(f+'phot/'+field_name+'/tmp/science/',
                                              f+'phot/'+field_name+'/std_{}_{}_02_wcs-lin.eps'.format(field_name, f[-9:-1]))
            if param['disable_calibration_shutter'] == 0:
                print('OK\nApplying shutter map correction...'),
                shutter_map.apply_correction(f+'phot/'+field_name+'/tmp/')
                nightly_std_test.perform_test(f+'phot/'+field_name+'/tmp/science/',
                                              f+'phot/'+field_name+'/std_{}_{}_03_wcs-lin-shu.eps'.format(field_name, f[-9:-1]))
            if param['disable_standard_cal'] == 0:
                print('OK\nCalibrating the CCD (bias, darks, flats)...'),
                bias_darks_flats.calibrate_data(f+'phot/'+field_name+'/tmp/')
                nightly_std_test.perform_test(f+'phot/'+field_name+'/tmp/science/',
                                              f+'phot/'+field_name+'/std_{}_{}_04_wcs-lin-shu-bdf.eps'.format(field_name, f[-9:-1]))
            print 'OK'
            move_files(f+'phot/'+field_name+'/tmp/', f+'phot/'+field_name)
            if check_first_run(f+field_name):
                update_frame_list(frame_list_file, f+'phot/'+field_name+'/cal/')
                check_double_entries(frame_list_file)
                make_completed_file_flag(f+field_name+'/not_first_run')
            make_completed_file_flag(f+field_name+'/cal_completed')
            print 'Calibration of {} successfully completed.'.format(f[-9:-1])


if __name__ == '__main__':
    # Testing
    calibrate_data()
    print('DONE')
