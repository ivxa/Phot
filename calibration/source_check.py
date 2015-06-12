# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import os
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from astropy.io import fits
import sys
param = {}
execfile(sys.argv[1])


def make_image_list(i):
    t = [i+d for d in os.listdir(i) if d[-5:] == '.fits']
    return t


def check_saturation(img, x, y, sl, tol=10):
    data = fits.getdata(img, header=False)
    pix_max = np.max(data[y-tol/2-1:y+tol/2-1, x-tol/2-1:x+tol/2-1])
    if pix_max > sl:
        raise RuntimeError("Source saturated at {}".format(img))


def check_center(img, x, y, tol=500):
    crop_region = param['crop_region']
    source_xy_shift = param['source_xy_shift']
    x_expected = (crop_region[1]-crop_region[0])/2.+source_xy_shift[0]
    y_expected = (crop_region[3]-crop_region[2])/2.+source_xy_shift[1]
    d = np.sqrt((x_expected-x)**2.+(y_expected-y)**2.)
    if d > tol:
        raise RuntimeError("Image not centered at {}".format(img))


def perform_control(i):
    sl = param['saturation_level']
    c = SkyCoord(param['ra'], param['dec'], unit=(u.hour, u.deg))
    il = make_image_list(i)
    for img in il:
        w = wcs.WCS(img)
        x, y = w.wcs_world2pix(c.ra.deg, c.dec.deg, 1)
        if param['check_centering'] == 1:
            check_center(img, x, y)
        check_saturation(img, np.int(x), np.int(y), sl)


if __name__ == '__main__':
    # Testing
    fn = param['field_name']
    f = param['test_path']
    perform_control(f+'phot/'+fn+'/tmp/science/')
    print 'DONE'
