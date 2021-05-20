#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:06:23 2021

@author: ghkerr
"""
# Local environment
DIR = '/Users/ghkerr/GW/edf/'
DIR_OUT = '/Users/ghkerr/GW/edf/'

# Pegasus
DIR = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'
DIR_OUT = DIR+'af/'

def calculate_af(no2, rr):
    """Calculate the fraction of asthma burden attributable to NO2 
    concentrations (i.e., the attributable fraction = AF) based on the     
    concentration-response factor (from the meta-analysis of epidemiological 
    studies) and gridded average NO2 concentrations. 
    
    Parameters
    ----------
    no2 : numpy.ndarray
        NO2 concentrations, units of ppb, [lat, lng]
    rr : float
        Relative risk (e.g., 1.26 (1.10-1.37) per 10 ppb)

    Returns
    -------
    af : numpy.ndarray
    New paediatric asthma cases attributable to NO2, units of percent, 
    [lat, lng]
    
    References
    ----------
    P. Achakulwisut, M. Brauer, P. Hystad, S. C. Anenberg, Global, national, 
        and urban burdens of paediatric asthma incidence attributable to 
        ambient NO2 pollution: estimates from global datasets. The Lancet 
        Planetary Health 3, e166–e178 (2019).
    S. C. Anenberg, L. W. Horowitz, D. Q. Tong, J. J. West, An Estimate of the 
        Global Burden of Anthropogenic Ozone and Fine Particulate Matter on 
        Premature Human Mortality Using Atmospheric Modeling. Environmental 
        Health Perspectives 118, 1189–1195 (2010).
    H. Khreis, et al., Exposure to traffic-related air pollution and risk of 
        development of childhood asthma: A systematic review and meta-
        analysis. Environment International 100, 1–31 (2017).        
    """
    import numpy as np
    # This the concentration-response factor of 1.26 (1.10 - 1.37) per 10 
    # ppb is used in Achakulwisut et al. (2019) and taken from Khreis et al. 
    # (2017). Note that this "log-linear"  relationship comes from 
    # epidemiological studies that log-transform concentration before 
    # regressing with incidence of health outcome (where log is the natural 
    # logarithm). Additional details can be found in Anenberg et al. (2010)
    beta = np.log(rr)/10.
    af = (1-np.exp(-beta*no2))
    return af 
    
def write_af(year):
    """Calculate the NO2-attributable asthma fraction for the ~United States
    (lower 48 states, Canada, Northern Mexico, Alaska) for a given year and 
    save the contents as a netCDF file. Native 

    Parameters
    ----------
    year : str
        Year of interest for which NO2-attributable fraction will be 
        calculated.

    Returns
    -------
    None
    
    References
    ----------    
    S. C. Anenberg, A. Mohegh, D. L. Goldberg, G. H. Kerr, M. Brauer, K. 
    Burkart, et al. Long-term trends in urban NO2 concentrations and 
    associated pediatric asthma incidence: estimates from global datasets.
    (in prep.)
    """
    import netCDF4 as nc
    import numpy as np
    from datetime import datetime
    import calendar
    import sys
    sys.path.append(DIR)
    import edf_open
    print('Calculating NO2-attributable asthma incidence for %s...'%year)
    rr = 1.36
    # Open Anenberg, Mohegh, et al. (2021) NO2 for year of interest
    lng_mohegh, lat_mohegh, no2_mohegh = edf_open.open_no2pop_tif(
        '%s_final_1km_usa'%year, -999., 'NO2')
    af_mohegh = calculate_af(no2_mohegh, rr)
    # Save as netCDF
    root_grp = nc.Dataset(DIR_OUT+'af_anenbergmoheghno2_usa_%s.nc'%year, 
        'w', format='NETCDF4')
    root_grp.title = u'NO2-attributable asthma burden for %s '%year + \
        'using RR = %.2f '%rr
    root_grp.history = 'Generated %d %s %d by Gaige Kerr (gaigekerr@gwu.edu)'\
        %(datetime.today().day, calendar.month_abbr[datetime.today().month],
        datetime.today().year)
    # Dimensions
    root_grp.createDimension('longitude', len(lng_mohegh))
    root_grp.createDimension('latitude', len(lat_mohegh))
    # Variables
    # Longitude
    var_lon = root_grp.createVariable('longitude', 'f8', ('longitude',))
    var_lon[:] = lng_mohegh
    var_lon.long_name = 'longitude'
    var_lon.units = 'degrees east'
    # Latitude
    var_lat = root_grp.createVariable('latitude', 'f8', ('latitude',))
    var_lat[:] = lat_mohegh
    var_lat.long_name = 'latitude'
    var_lat.units = 'degrees north'
    # TROPOMI extract
    var_out = root_grp.createVariable('AF', 'f8', 
        ('latitude', 'longitude',), fill_value=nc.default_fillvals['f8'])
    af_mohegh[np.where(np.isnan(af_mohegh)==True)] = nc.default_fillvals['f8']
    var_out[:] = af_mohegh
    var_out.long_name = 'attributable_fraction'
    var_out.units = 'None'
    # Closing
    root_grp.close()
    return

# Calculate attributable fraction 
for year in [1990, 1995, 2000, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 
    2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]:
    write_af(year)