# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
import shutil
from astropy.io import fits
import numpy as np
import sys
param = {}
execfile(sys.argv[1])


def make_tmp_folder(o):
    if os.path.exists(o):
        shutil.rmtree(o)
    os.makedirs(o)


def make_file_list(i):
    file_list = [n for n in os.listdir(i) if n[-5:] == '.fits']
    return file_list


def shutter_correction(i, o, f, s):
    smap = fits.getdata(s+'shutter_map.fits', header=False)
#    dx = np.int(smap.shape[1]*0.05/2.)
#    dy = np.int(smap.shape[0]*0.05/2.)
#    xcenter = smap.shape[1]/2-1
#    ycenter = smap.shape[0]/2-1
#    norm = np.median(smap[ycenter-dy:ycenter+dy, xcenter-dx:xcenter+dx])
#    smap /= norm
    data, header = fits.getdata(i+f, header=True)
    texp = np.float(header['EXPTIME'])
    data /= 1.0 + smap/texp
    fits.writeto(o+f, data, header)


def move_files(i, o):
    if os.path.exists(o):
        [shutil.copy2(o+f, i) for f in os.listdir(o) if f[-5:] == '.fits']
        shutil.rmtree(o)


def apply_correction(fi):

    def correct_set(i, s):
        make_tmp_folder(i+'shu/')
        file_list = make_file_list(i)
        for f in file_list:
            shutter_correction(i, i+'shu/', f, s)
        move_files(i, i+'shu/')
    if param['disable_screen_flat'] == 1:       
        correct_set(fi+'flats/', fi+'shutter/')
    correct_set(fi+'science/', fi+'shutter/')


if __name__ == '__main__':
    # Testing
    field_name = param['field_name']
    fp = param['test_path']
    apply_correction(fp+'phot/'+field_name+'/tmp/')
    print('DONE')
