# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
import shutil
import subprocess
import sys
param = {}
execfile(sys.argv[1])


def make_tmp_folder(o):
    if os.path.exists(o):
        shutil.rmtree(o)
    os.makedirs(o)


def call_astrometry_dot_net(i, o, ra, dec, radius, scale_low, scale_high):
    """ -D: Output path
        -p: No plots
        -U -M -R -B (with "none"): Less output files
        -J: If *.solved exist don't fit again
        -m: where to put temp files, default /tmp
        -O: Overwrite the output files
        --new-fits: Output file name"""
    cmd = ('solve-field --ra {} --dec {} --radius {} '
           '--scale-units arcsecperpix --scale-low {} --scale-high {} '
           '-D {} -p -U none -M none -R none -B none -J -m {} -O --new-fits %s.fits {}')\
        .format(ra, dec, radius, scale_low, scale_high, o, o, i+'*.fits')
    with open(os.devnull, "w") as f:
        subprocess.call(cmd, shell=True, stdout=f, stderr=f)


def move_files(i, o):
    if os.path.exists(o):
        [shutil.copy2(o+f, i) for f in os.listdir(o) if f[-5:] == '.fits']
        shutil.rmtree(o)


def set_wcs(i, o):
    ra = param['ra']
    dec = param['dec']
    radius = param['radius']
    scale_low = param['scale_low']
    scale_high = param['scale_high']

    make_tmp_folder(o)
    call_astrometry_dot_net(i, o, ra, dec, radius, scale_low, scale_high)
    move_files(i, o)


if __name__ == '__main__':
    # Testing
    field_name = param['field_name']
    fp = param['test_path']
    set_wcs(fp+'phot/'+field_name+'/tmp/science/', fp+'phot/'+field_name+'/tmp/science/wcs/')
    print('DONE')

