# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md
# based on http://www.astropy.org/astropy-tutorials/FITS-images.html

import matplotlib
import numpy as np
import sys
import os
from astropy.io import fits
import matplotlib.pyplot as plt


def make_file_list(p):
    return [os.path.join(p, d) for d in os.listdir(p) if d[0] == 'T']


def main():
    data_path = '/home/gamma/garrofa/xparedes/Baixades/20151112/I/'
    image_list = make_file_list(data_path)
    image_concat = [ fits.getdata(image) for image in image_list ]
    final_image = np.sum(image_concat, axis=0)
    from matplotlib.colors import LogNorm
    plt.imshow(final_image, cmap='gray', norm=LogNorm(vmin=0.01, vmax=max(final_image.flat)))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    try:
        main()
        print '\nNormal termination.'
    except Exception, err:
        sys.stderr.write('\nERROR: %s\n' % str(err))
