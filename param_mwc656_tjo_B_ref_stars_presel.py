# Author: Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)
# License: MIT, see LICENSE.md


param = {
    'rerun': 0,
    'disable_calibration': 0,
    'disable_standard_cal': 1,
    'disable_flats': 0,
    'flats_dir': '/home/gamma/garrofa/xparedes/data/tjo/flats_B/',
    'disable_bias': 1,
    'disable_darks': 1,
    'disable_calibration_shutter': 1,
    'disable_calibration_lin': 1,
    'disable_analysis': 0,
    'disable_analysis_extraction': 0,
    'disable_parab_fit': 1,
    'disable_plots': 0,
    'disable_plots_cycles': 0,
    'disable_plots_error_bars': 0,
    'disable_plots_nightly': 0,
    'check_centering': 0,
    'tol_center': 100,
    'field_name': 'B',
    'title_name': 'MWC 656',
    'zero_magnitude': 8.81,
    'period': 60.37,
    'JD0': 2453243.3,
    'JD0_cycle': 2453243.3,
    'colormap_cycles': 'Accent',
    'colormap_cycles_range': (0, 1),
    'ra': '22:42:57',
    'dec': '+44:43:18',
    'auto_sel': True,
    'create_ref_star_list': True,
    'ref_star_file': 'ref_and_comp_stars.dat',
    'ref_star_file_out': '/home/gamma/garrofa/xparedes/photometry_tjo/mwc656/',
    'dmax': 1.0,
    'dmax_final': 0.3,
    'mmin0': 2,
    'mmax0': 3,
    'mmin': 2.0,
    'mmax': 3.5,  # YOU SHOULD CHOOSE VALUES CLOSE TO YOUR TARGET TO AVOID CORRECTING ONLY THE MOST ABUNDANT STARS
    'nsel': 4,
    'nsel_plots': 5,
    'radius': 0.15,
    'min_frames_per_night': 5,
    'nstars_tolerance': 0.35,
    'astrometric_tolerance': 12,
    'max_nstars_missmatch_tolerance': 0.75,
    'sextractor_file': 'se_mwc656_tjo.sex',
    'scale_low': 0.34,
    'scale_high': 0.38,
    'frame_list': '/home/gamma/garrofa/xparedes/photometry_tjo/mwc656/B/frame_list.txt',
    'data_path': '/home/gamma/garrofa/xparedes/data/tjo/',
    'crop_region': (998.0, 3098.0, 998.0, 3098.0), #(1550.0, 2950.0, 1550.0, 2550.0)
    'source_xy_shift': (0, 0),
    'saturation_level': 55000.,
    'saturation_level_post_calibration': 53000.,
    'output_path': '/home/gamma/garrofa/xparedes/Dropbox/photometry_tjo/mwc656/B/'
}
