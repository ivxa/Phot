# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import matplotlib
import numpy as np
import sys
import os
import subprocess
import pyfits


def listdirs(p):
    return [os.path.join(p, d) for d in os.listdir(p) \
            if os.path.isdir(os.path.join(p, d)) and d[0] == '2']


def file_type(d, fl):
    bias = []
    darks = []
    flats = []
    V = []
    B = []
    R = []
    I = []
    for f in fl:
        header = pyfits.getheader(os.path.join(d, f))
        try:
            imtype = header['OBSTYPE']
        except:
            imtype = 'noimtype'
        try:
            filtr = header['FILTER']
        except:
            filtr = 'nofiltr'
        if imtype == 'Science':
            if filtr == 'B':
                B.append(f)
            elif filtr == 'V':
                V.append(f)
            elif filtr == 'R':
                R.append(f)
            elif filtr == 'I':
                I.append(f)
        elif imtype == 'Bias':
            bias.append(f)
        elif imtype == 'Flat':
            flats.append(f)
        elif imtype == 'Dark':
            darks.append(f)
    assert (len(bias)+len(darks)+len(flats)+len(V)+len(B)+len(R)+len(I)) \
            == len(fl), 'The total file number is wrong at {}'.format(f)
    return bias, darks, flats, V, B, R, I


def make_directories(new_dir):
    for n in new_dir:
        if not os.path.exists(n):
            os.makedirs(n)

def move_files(i, o, to_move):
    for f in to_move:
        os.rename(os.path.join(i, f), os.path.join(o, f))


def remove_empty_directories(dir_list):
    for d in dir_list:
        if not os.listdir(d):
            os.removedirs(d)


def main():
    data_path = '/home/gamma/garrofa/xparedes/data/tjo/'
    # data_path = '/home/gamma/garrofa/xparedes/data/tjo_test/'
    dir_list = np.sort(listdirs(data_path))
    for d in dir_list:
        print d

    for d in dir_list:
        file_list = os.listdir(d)
        if 'org_completed' not in file_list:
            file_list = np.sort([f for f in file_list if f[:3] == 'TJO'])

            bias, darks, flats, V, B, R, I = file_type(d, file_list)

            new_dir = [os.path.join(d, i) for i in \
                       ['bias', 'darks', 'flats', 'V', 'B', 'R', 'I']]
            make_directories(new_dir)

            for (i, t) in enumerate([bias, darks, flats, V, B, R, I]):
                move_files(d, new_dir[i], t)

            remove_empty_directories(new_dir)

            subprocess.call(['touch', os.path.join(d, 'org_completed')])
            subprocess.call(['touch', os.path.join(d, 'not_revised')])


if __name__ == '__main__':
    try:
        main()
        print '\nNormal termination.'
    except Exception, err:
        sys.stderr.write('\nERROR: %s\n' % str(err))
