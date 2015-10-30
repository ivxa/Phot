# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


param = {
    'rerun': 0,                          # 1 to recalibrate the images
    'disable_calibration': 0,            # 1 to disable the calibration
    'disable_standard_cal': 0,           # 1 to disable the calibration
    'disable_screen_flat': 0,            # 1 to disable screen flats
    'disable_bias': 1,                   # 1 to disable bias correction
    'disable_calibration_shutter': 0,    # 1 to disable shutter map correction
    'disable_calibration_lin': 1,        # 1 to disable non-linearity correction
    'disable_analysis': 0,               # 1 to disable the analysis
    'disable_analysis_extraction': 0,    # 1 to disable the source extraction
    'disable_parab_fit': 1,              # 1 to disable parabola fitting for comparison star selection process
    'disable_plots': 0,                  # 1 to disable the plotting
    'disable_plots_cycles': 0,           # 1 to disable the orbital cycle plots (useful only for binary systems)
    'disable_plots_error_bars': 0,       # 1 to hide de error bars
    'check_centering': 0,                # 1 to check the centering of the images
    'tol_center': 100,                   # Maximum distance in pixels of decentering
    'field_name': 'mwc656',              # Field name without empty spaces (it is used to build the directory tree)
    'title_name': 'MWC 656',             # Title name for plotting
    'zero_magnitude': 8.81,              # Artificial offset to the mean magnitude
    'period': 60.37,                     # Period if the light curve presents a periodical signal
    'JD0': 2453243.3,                    # JD_0 of the zero phase corresponding to
    'JD0_cycle': 2453243.3,              # JD_0 of the zero phase for colouring each period
    'colormap_cycles': 'Accent',         # Colour scale for the periodic cycles
    'colormap_cycles_range': (0, 1),     # To cut the colour scale
    'ra': '22:42:57',                    # RA coordinate of the target
    'dec': '+44:43:18',                  # DEC coordinate of the target
    'reference_stars': "[('22:43:07', '+44:42:44'), ('22:42:42', '+44:44:52'), ('22:42:50', '+44:40:20')]", # Preselected reference stars (ONLY FOR THE PRESEL BRANCH)
    'dmax': 1.0,                         # Maximum distance with respect any star in the FOV
    'dmax_final': 0.3,                   # Maximum distance of the comparison stars
    'mmin0': 2,                          # Minimum magnitude difference of the comparison stars at first iteration
    'mmax0': 3,                          # Maximum magnitude difference of the comparison stars at first iteration
    'mmin': 2.0,                         # Minimum magnitude difference of the comparison stars
    'mmax': 2.0,                         # Maximum magnitude diff. of the comparison stars (YOU SHOULD CHOOSE VALUES CLOSE TO YOUR TARGET TO AVOID CORRECTING ONLY THE MOST ABUNDANT STARS)
    'nsel': 5,                           # Number of comparison stars stars
    'nsel_plots': 5,                     # How many comparison stars do you want to use
    'radius': 2,                         # Astrometry.net input parameter for the astrometric solution
    'min_frames_per_night': 4,           # Discard the nights with less than this number of images
    'nstars_tolerance': 0.35,            # Minimum ratio with respect the maximum number of stars
    'astrometric_tolerance': 6,          # Astrometric tolerance when matching stars at different images (in arcsec)
    'max_nstars_missmatch_tolerance': 0.75, # Maximum ratio of bad matching between frames
    'scale_low': 3.87,                   # Minimum pixel scale of the CCD in arcsec/pix
    'scale_high': 3.89,                  # Maximum pixel scale of the CCD in arcsec/pix
    'frame_list': '/home/gamma/garrofa/xparedes/photometry/test/mwc656/frame_list.txt',  # Frame list path
    'data_path': '/home/gamma/garrofa/xparedes/data/tfrm_data/test/',                    # Data path
    'crop_region': (998.0, 3098.0, 998.0, 3098.0), #(1550.0, 2950.0, 1550.0, 2550.0)     # Cropping coordinates of the image (in pixels)
    'source_xy_shift': (0, 0),                                                           # If the source is not centered (needed for centering test)
    'saturation_level': 55000.,                                                          # Stars above this value are discarded (or the image if the star is saturated)
    'saturation_level_post_calibration': 53000.,                                         # After calibration the saturation level changes
    'output_path': '/home/gamma/garrofa/xparedes/Dropbox/photometry/mwc656/'             # The output path
}
