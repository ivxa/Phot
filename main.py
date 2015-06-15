# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


import sys
import calibration
import analysis
import plots
param = {}
execfile(sys.argv[1])


def main():
    if param['disable_calibration'] == 0:
        print('\nCCD CALIBRATION:\n')
        calibration.calibrate_data()
        print('Calibration successfully completed.')

    if param['disable_analysis'] == 0:
        print('\n----------------------------------------')
        print('\nDATA ANALYSIS:\n')
        analysis.analyze_data()
        print('Data analysis successfully completed.')

    if param['disable_plots'] == 0:
        print('\n----------------------------------------')
        print('\nPLOTTING:\n')
        if param['field_name'] == 'lsi61303':
            plots.make_plots_lsi61303()
        else:
            plots.make_plots()
        print('Plotting successfully completed.')


if __name__ == '__main__':
    try:
        main()
        print '\nNormal termination.'
    except Exception, err:
        sys.stderr.write('\nERROR: %s\n' % str(err))
