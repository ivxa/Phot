# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
import shutil
from astropy.io import fits
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


def nonlinear_correction(i, o, f, l):
    b = fits.getdata(l+'b.fits', header=False)
    c = fits.getdata(l+'c.fits', header=False)
    data, header = fits.getdata(i+f, header=True)
    data = data+b*data**2.+c*data**3.
    fits.writeto(o+f, data, header)


def move_files(i, o):
    if os.path.exists(o):
        [shutil.copy2(o+f, i) for f in os.listdir(o) if f[-5:] == '.fits']
        shutil.rmtree(o)


def apply_correction(fi):

    def correct_set(i, l):
        make_tmp_folder(i+'lin/')
        file_list = make_file_list(i)
        for f in file_list:
            nonlinear_correction(i, i+'lin/', f, l)
        move_files(i, i+'lin/')

    correct_set(fi+'bias/', fi+'linear/')
    correct_set(fi+'darks/', fi+'linear/')
    correct_set(fi+'flats/', fi+'linear/')
    correct_set(fi+'science/', fi+'linear/')


if __name__ == '__main__':
    # Testing
    field_name = param['field_name']
    fp = param['test_path']
    apply_correction(fp+'phot/'+field_name+'/tmp/')
    print('DONE')

