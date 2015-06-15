# Phot

**Phot** is a **pipeline** aimed to reduce and analyze optical images through **differential photometry**. The *input* files are the calibration images and the multi-night science images. The *output* files are the corrected light curves. The code is aimed for multi-night data and it averages the nightly data in one single point. If you only have one single shot per night you should modify the offset.py file (it's straightforward).

**WARNING**: the pipeline creates and deletes temporary folders and this might be dangerous if you make any mistake setting up the file and directory tree (see Requirements). Make sure that you have a backup of your data images before running the pipeline for the first time.

The *main steps* of the pipeline are:

1. Calibration
  1. Cropping of the images (Astropy)
  2. Astrometric reduction (Astrometry.net: tan-sip fit; valid for large field of views)
  3. Saturation and centering control
  4. Non-linearity correction (optional; disable at param.py)
  5. Shutter map correction (optional; disable at param.py)
  6. Standard calibration (bias, darks, flats; PyRAF)
  7. Nightly standard deviation versus magnitude plot
2. Analysis
  1. Source extraction (SExtractor)
  2. Quality control
  3. Matching between catalogs (Astropy)
  4. Mean differential photometry correction
  5. (not implemented yet) Quality control using optical catalogs
  6. (not implemented yet) XY fitting
  7. (not implemented yet) Long term detrending of the light curves
  8. Multi-night standard deviation versus magnitude plot
  9. Artificial offset to the mean magnitude and error estimation
3. Plotting

This software has been developed during my **PhD thesis** (University of Barcelona) to reduce and analyze the optical images of some amazing galactic objects called gamma-ray binaries. I would like to thank and give the appropriate credit to my thesis supervisors Marc Ribó and Valentí Bosch-Ramon, without their guide and discussions none of this would have been possible. I also want to thank to Octavi Fors and Daniel del Ser for the very valuable discussions on the technical implementation, and to Benito Marcote and Javier Moldón for their useful Python tips. The pipeline involves several steps and uses external tools and Python packages such as Astropy, PyIRAF, Astrometry.net and SExtractor, finally I want to give credit to the authors and people who contributed to these tools.

The following **papers** have been published in refereed journals using a **previous** version of this software:

* http://cdsads.u-strasbg.fr/abs/2015A%26A...575L...6P
* http://cdsads.u-strasbg.fr/abs/2014IJMPS..2860197P
* http://cdsads.u-strasbg.fr/abs/2012AIPC.1505..390P

[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

## Author
Xavier Paredes-Fortuny (xparedesfortuny@gmail.com)


## License
See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).


## Usage
1. Install all of the required dependencies
2. Create the directory tree (see below)
3. Edit the `param.py` file
4. Edit the `se.sex` file (see external SExtractor documentation)
5. Run `python main.py param.py`


## Requirements

### Dependencies
* Astrometry.net and the catalog files
* SExtractor
* Some Python packages including PyIRAF (the easiest way to install PyIRAF is installing the Ureka distribution; see the [requirements.txt](requirements.txt) file)
* LaTeX (optional: for nice plots)
* epstool (optional: to perform a tight crop when using LaTeX, it can be disabled at plots.py)


### Input
* {$YYYYMMDD}/ 
  * flats_link (symbolic link or folder with the flat field images)
  * bias_no_cal/ (folder with the bias images)
  * org_completed (needed for linking with an external download script which is not included. You can create a dummy file: "touch org_completed")
  * shutter_map_link (optional: you can disable shutter map correction at param.py)
  * linearity_map_link (optional: you can disable linearity correction at param.py)
  * {$field_name}/ (same field name as in param.py)
    * cal_completed (created at first run after calibration)
    * not_first_run (created at first run)
    * raw_no_cal/
      * 20*.fits
* {$output_path}/ (defined at param.py)


## Output
* {$YYYYMMDD}/phot/
  * {$field_name}/
    * master_flat
    * mater_bias
    * master_dark
    * shutter_map
    * linearity_map_A
    * linearity_map_B
    * std_*.eps
    * cal/
    * tmp/ (tmp)
      * bias
      * darks
      * flats
      * linear
      * science
      * shutter
    * cat/ (tmp)

* output_path/
  * frame_list.txt
  * maximum_number_of_stars_per_night-qc.eps
  * nightly_LC/
    * {$YYYYMMDD}-MJD-{$field_name}-target.eps
    * {$YYYYMMDD}-PHA-{$field_name}-target.eps
    * {$YYYYMMDD}-MJD-{$field_name}-ref_stars_self_corrected.eps
    * {$YYYYMMDD}-MJD-{$field_name}-ref_stars_not_self_corrected.eps
  * std_multi_night_plots/
    * std_{$field_name}_multi_night_*_*.eps
  * multi_night_LC/
    * MJD-{$field_name}-target-all_frames.eps
    * MJD-{$field_name}-target-nightly_average.eps
    * MJD-{$field_name}-target-nightly_average_cycles.eps
    * PHA-{$field_name}-target-all_frames.eps
    * PHA-{$field_name}-target-nightly_average.eps
    * PHA-{$field_name}-target-nightly_average_cycles.eps
    * MJD-{$field_name}-ref_stars_self_corrected.eps
    * MJD-{$field_name}-ref_stars_not_self_corrected.eps
  * data/
    * MJD_MAG_ERR-{$field_name}-all_frames.dat
    * MJD_MAG_ERR-{$field_name}-nightly_average.dat
    * param.py (copy of the input param file)
    * se.param (copy)
    * se.sex (copy)

