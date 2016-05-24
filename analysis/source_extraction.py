# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
import shutil
import sys
import numpy as np
import subprocess
from astropy.io import fits
import pyfits
param = {}
execfile(sys.argv[1])


def make_folder_list(i):
    t = [i+d+'/' for d in sorted(os.listdir(i)) if d[:2] == '20' and os.path.exists(i+d+'/'+param['field_name'])]
    return t


def make_cat_folder(o):
    if os.path.exists(o):
        shutil.rmtree(o)
    os.makedirs(o)


def make_image_list(i, fl):
    t = [d for d in sorted(os.listdir(i)) if d[-5:] == '.fits' and i+d in fl]
    return t


def call_sextractor(i, im, sl):
    header = pyfits.getheader(i+'/cal/'+im)
    if int(header['MJD']) < 57416:
        GAIN = 12.5
        PIXEL_SCALE = 3.9
    else:
        GAIN = 0.34
        PIXEL_SCALE = 2.37

    cat_name = i+'/cat/'+im[:-6]+'.cat'
    cmd = 'sex {} -c {} -CATALOG_NAME {} -SATUR_LEVEL {} -GAIN {} -PIXEL_SCALE {}'.format(i+'/cal/'+im, param['sextractor_file'], cat_name, sl, GAIN, PIXEL_SCALE)
    if param['disable_analysis_extraction'] == 0:
        subprocess.call(cmd, shell=True)
    return cat_name


def create_catalog_arrays(i, il, sl):
    cat_list_ra = []
    cat_list_dec = []
    cat_list_mag = []
    for im in il:
        cat_name = call_sextractor(i, im, sl)
        mag, x, y, flag = np.loadtxt(cat_name, usecols=(0, 2, 3, 4), unpack=True)

        # SExtractor is unable to read the tan-sip wcs produced by Astrometry.net
        from astropy import wcs
        w = wcs.WCS(i+'/cal/'+im)
        ra, dec = w.all_pix2world(x, y, 1)

        cat_list_mag.append(mag[flag == 0])
        cat_list_ra.append(ra[flag == 0])
        cat_list_dec.append(dec[flag == 0])
    return cat_list_ra, cat_list_dec, cat_list_mag


def remove_cat_folder(i):
    if os.path.exists(i):
        shutil.rmtree(i)


def create_mjd_catalog(i, il):
    return [fits.getheader(i+fn)['MJD'] for fn in il]


def perform_extraction(i, frame_list, testing=1):
    folder_list_aux = make_folder_list(i)
    field_name = param['field_name']
    sl = param['saturation_level_post_calibration']
    cat_mjd = []
    cat_ra = []
    cat_dec = []
    cat_mag = []
    folder_list = []
    frame_list_flat = [item for sublist in frame_list for item in sublist]
    [folder_list.extend([f]) for f in folder_list_aux if f in [frame[:len(f)] for frame in frame_list_flat]]
    nnights = len(folder_list)
    for (n, f) in enumerate(folder_list):
        sys.stdout.write('\r  Computing night {} of {}'.format(n+1, nnights))
        sys.stdout.flush()
        if param['disable_analysis_extraction'] == 0:
            make_cat_folder(f+'phot/'+field_name+'/cat')
        il = make_image_list(f+'phot/'+field_name+'/cal/', frame_list_flat)
        cat_list_mjd = create_mjd_catalog(f+'phot/'+field_name+'/cal/', il)
        cat_mjd.append(cat_list_mjd)
        cat_list_ra, cat_list_dec, cat_list_mag = create_catalog_arrays(f+'phot/'+field_name, il, sl)
        cat_ra.append(cat_list_ra)
        cat_dec.append(cat_list_dec)
        cat_mag.append(cat_list_mag)
        if testing == 0:
            remove_cat_folder(f+'phot/'+field_name+'/cat')
    print '\n'
    return cat_ra, cat_dec, cat_mag, cat_mjd


if __name__ == '__main__':
    # Testing
    perform_extraction(param['data_path'])
    print 'DONE'
