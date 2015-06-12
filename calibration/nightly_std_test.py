# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
import shutil
import subprocess
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
param = {}
execfile(sys.argv[1])


def make_image_list(i):
    t = [d for d in os.listdir(i) if d[-5:] == '.fits']
    return t


def make_cat_folder(i):
    if os.path.exists(i+'cat'):
        shutil.rmtree(i+'cat')
    os.makedirs(i+'cat')


def remove_cat_folder(i):
    if os.path.exists(i+'cat'):
        shutil.rmtree(i+'cat')


def call_sextractor(i, im, sl):
    cat_name = i+'cat/'+im[:-6]+'.cat'
    cmd = 'sex {} -c se.sex -CATALOG_NAME {} -SATUR_LEVEL {}'.format(i+im, cat_name, sl)
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
        w = wcs.WCS(i+'/'+im)
        ra, dec = w.all_pix2world(x, y, 1)

        cat_list_mag.append(mag[flag == 0])
        cat_list_ra.append(ra[flag == 0])
        cat_list_dec.append(dec[flag == 0])
    return cat_list_ra, cat_list_dec, cat_list_mag


def match_stars(cat_list_ra, cat_list_dec, cat_list_mag):
    tol = param['astrometric_tolerance']
    # Convert from arcsec to deg
    tol /= 3600.

    cat1 = SkyCoord(cat_list_ra[0]*u.degree, cat_list_dec[0]*u.degree)
    match_flag = np.ones(len(cat_list_ra[0]), dtype='bool')
    for i in xrange(1, len(cat_list_ra)):
        cat2 = SkyCoord(cat_list_ra[i]*u.degree, cat_list_dec[i]*u.degree)
        index, sep2d, dist3d = cat1.match_to_catalog_sky(cat2)

        cat_list_ra[i] = cat_list_ra[i][index]
        cat_list_dec[i] = cat_list_dec[i][index]
        cat_list_mag[i] = cat_list_mag[i][index]
        wrong_match = [j for (j, d) in enumerate(sep2d.value) if d > tol]
        match_flag[wrong_match] = False

    for i in xrange(len(cat_list_ra)):
        cat_list_ra[i] = cat_list_ra[i][match_flag]
        cat_list_dec[i] = cat_list_dec[i][match_flag]
        cat_list_mag[i] = cat_list_mag[i][match_flag]


def compute_statistics(cat_list_mag):
    avg_mag = [np.average(cat_list_mag[:, k]) for k in xrange(len(cat_list_mag[0, :]))]
    std_mag = [np.std(cat_list_mag[:, k]) for k in xrange(len(cat_list_mag[0, :]))]
    return np.array(avg_mag), np.array(std_mag)


def plot(x, y, o):
    plt.rcdefaults()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '.')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\overline{m}$ (mag)')
    ax.set_ylabel(r'$\sigma_{m}$ (mag)')
    ax.set_xlim((min(x)*(1-0.05), max(x)*(1+0.05)))
    ax.set_ylim((min(y)*(1-0.05), max(y)*(1+0.05)))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.table(cellText=[['N', r'$\overline{{\sigma}}$'],
                        [1, '{:.3f}'.format(y[0])],
                        [5, '{:.3f}'.format(np.average(y[0:5]))],
                        [10, '{:.3f}'.format(np.average(y[0:10]))],
                        [25, '{:.3f}'.format(np.average(y[0:25]))],
                        [50, '{:.3f}'.format(np.average(y[0:50]))],
                        [100, '{:.3f}'.format(np.average(y[0:100]))]],
              colWidths=[0.1, 0.1],
              loc='center left')
    fig.savefig(o, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def perform_test(i, o):
    sl = param['saturation_level']
    il = make_image_list(i)
    make_cat_folder(i)
    cat_list_ra, cat_list_dec, cat_list_mag = create_catalog_arrays(i, il, sl)
    match_stars(cat_list_ra, cat_list_dec, cat_list_mag)
    cat_list_mag = np.array(cat_list_mag)
    avg_mag, std_mag = compute_statistics(cat_list_mag)
    plot(avg_mag[std_mag.argsort()], sorted(std_mag), o)
    remove_cat_folder(i)


if __name__ == '__main__':
    # Testing
    fn = param['field_name']
    f = param['test_path']
    perform_test(f+'phot/'+fn+'/tmp/science/', f+'phot/'+fn+'/std_00_test.eps')
    print 'DONE'
