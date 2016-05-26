# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
from astropy.io import fits
import shutil
import sys
param = {}
execfile(sys.argv[1])

def make_output_folder(o):
    if os.path.exists(o):
        shutil.rmtree(o)
    os.makedirs(o)


def make_file_list(i):
    file_list = [n for n in os.listdir(i) if n[-5:] == '.fits']
    return file_list


def crop(i, o, n, x0, x1, y0, y1):
    data, header = fits.getdata(i+n, header=True)

    (x0, x1, y0, y1) = param['crop_region_old_camera']
    try:
        if int(header['MJD']) > 57425:
            (x0, x1, y0, y1) = param['crop_region']
    except:
        pass

    data = data[y0:y1, x0:x1]
    fits.writeto(o+n, data, header)


def crop_data(fi, fo):
    fn = param['field_name']
    (x0, x1, y0, y1) = param['crop_region']

    def crop_set(i, o):
        make_output_folder(o)
        file_list = make_file_list(i)
        for n in file_list:
            crop(i, o, n, x0, x1, y0, y1)

    if param['disable_standard_cal'] == 0:
        if param['disable_screen_flat'] == 1:
            crop_set(fi+'supermasterflat_link/', fo+'/tmp/flats/')
        else:
            crop_set(param['data_path']+'flat_pantalla/', fo+'/tmp/flats/')
        crop_set(fi+'bias_no_cal/', fo+'/tmp/bias/')
        crop_set(fi+fn+'/raw_no_cal/darks_no_cal/', fo+'/tmp/darks/')
    if param['disable_calibration_lin'] == 0:
        # crop_set(fi+'linearity_map_link/', fo+'/tmp/linear/')
        crop_set('/home/gamma/garrofa/xparedes/data/tfrm_data/linearity_map/',
            fo+'/tmp/linear/')
    if param['disable_calibration_shutter'] == 0:
        crop_set(fi+'shutter_map_link/', fo+'/tmp/shutter/')
    crop_set(fi+fn+'/raw_no_cal/', fo+'/tmp/science/')



if __name__ == '__main__':
    # Testing
    field_name = param['field_name']
    f = param['test_path']
    crop_data(f, f+'/phot/'+field_name)
    print('DONE')
