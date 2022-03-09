#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:06:23 2021

@author: ghkerr
"""
# Local environment
DIR = '/Users/ghkerr/GW/edf/'
DIR_OUT = '/Users/ghkerr/GW/edf/'
DIR_GBD = '/Users/ghkerr/GW/data/gbd/'

# # Pegasus
# DIR = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'
# DIR_OUT = DIR+'af/'

def calculate_af(no2, rr=1.26):
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
    rr = 1.26
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

def latlon_to_geoid(lat, lng, year):
    """Run API (through FCC) to return information about census block GEOID
    from latitude/longitude coordinates. Adapted from https://
    gis.stackexchange.com/questions/294641/python-code-for-transforming-lat
    -long-into-fips-codes

    Parameters
    ----------
    lat : float
        Latitude of air quality monitor, units of degrees north
    lng : float
        Longitude of air quality monitor, units of degrees east

    Returns
    -------
    geoid : string
        FIPS data about the county, state, and census tract
    """
    import requests
    import urllib
    # Encode parameters 
    params = urllib.parse.urlencode({'latitude':lat, 
        'longitude':lng, 'censusYear':year, 'format':'json'})
    # Contruct request URL
    url = 'https://geo.fcc.gov/api/census/block/find?'+params
    # Get response from API
    response = requests.get(url)
    # Parse json in response
    data = response.json()
    # FIPS codes represents census block values; strip off the block values 
    # and retain only the tract values (such that returned value is compatible 
    # with GEOID in harmonized tables). In the case that the AQS station 
    # is outside the country (there are some in Mexicali, Tijuana, etc.), 
    # set as None
    try: 
        geoid = data['Block']['FIPS'][:11]
    except TypeError:
        geoid = None
    return geoid

# def calculate_harmburden(harm, cf=2., rr=1.26, refyear=None):
#     """Calculate census tract-level paediatric asthma burden for four age 
#     groups (< 5 years old, 5-9 years old, 10-14 years old, and 15-19 years 
#     old) given census tract population estimates, counterfactual concentration
#     below which there is no risk of paediatric asthma development, and a 
#     concentration-response function. 

#     Parameters
#     ----------
#     harm : pandas.core.frame.DataFrame
#         Harmonized U.S. Census Bureau/NO2 and attributable fraction dataset
#         for particular vintage of geographic boundaries. 
#     cf : float, optional
#         Counterfactual concentartion. The default is 2. 
#     rr : float, optional
#         Concentration-response function, assuming a log-linear relationship 
#         between NO2 concentration and outcome per 10 ppb. The default is 1.26.
        
#     Returns
#     -------
#     harmout : pandas.core.frame.DataFrame
#         Same as above but also containing total paediatric asthma burdens, 
#         asthma burdens per 100,000 children, and burdens for age groups < 5 
#         years old, 5-9 years old, 10-14 years old, and 15-19 years old. 
#     """
#     import numpy as np
#     import pandas as pd 
#     # Calculate Khreis PAF with counterfactual concentration of 2 ppbv
#     beta = np.log(rr)/10.
#     betaupper = np.log(1.37)/10.
#     betalower = np.log(1.1)/10.
#     harm['AFPA'] = harm['AFPA']-(1-np.exp(-beta*cf))
#     harm['AFPALOWER'] = harm['AFPALOWER']-(1-np.exp(-betalower*cf))
#     harm['AFPAUPPER'] = harm['AFPAUPPER']-(1-np.exp(-betaupper*cf))
#     # If attributable fraction < 0, set to 0
#     harm['AFPA'] = harm['AFPA'].mask(harm['AFPA'].lt(0), 0.)
#     harm['AFPALOWER'] = harm['AFPALOWER'].mask(harm['AFPALOWER'].lt(0), 0.)
#     harm['AFPAUPPER'] = harm['AFPAUPPER'].mask(harm['AFPAUPPER'].lt(0), 0.)
    
#     # Calculate GBD PAF with counterfactual concentration based on a TMREL 
#     # defined by a uniform distribution between 4.545 and 6.190 ppb; in our 
#     # case we use the midpoint (5.3675 ppb) as our counterfactual and use the 
#     # lookup tables to calculate the log-transformed RR for this 
#     # counterfactual concentration
#     betagbd = pd.read_csv(DIR_GBD+'no2_rr_draws_summary.csv', sep=',',
#         engine='python')
#     exposureclosest_index = betagbd['exposure'].sub(5.3675).abs(
#         ).idxmin()
#     # Closest exposure to tract-averaged NO2
#     ec = betagbd.iloc[exposureclosest_index]
#     harm['AFPAMEAN_GBD'] = harm['AFPAMEAN_GBD']-(1-np.exp(-ec['mean']))
#     harm['AFPAMEDIAN_GBD'] = harm['AFPAMEDIAN_GBD']-(1-np.exp(-ec['mean']))
#     harm['AFPAUPPER_GBD'] = harm['AFPAUPPER_GBD']-(1-np.exp(-ec['mean']))
#     harm['AFPALOWER_GBD'] = harm['AFPALOWER_GBD']-(1-np.exp(-ec['mean']))
#     harm['AFPAMEAN_GBD'] = harm['AFPAMEAN_GBD'].mask(
#         harm['AFPAMEAN_GBD'].lt(0), 0.)
#     harm['AFPAMEDIAN_GBD'] = harm['AFPAMEDIAN_GBD'].mask(
#         harm['AFPAMEDIAN_GBD'].lt(0), 0.)
#     harm['AFPAUPPER_GBD'] = harm['AFPAUPPER_GBD'].mask(
#         harm['AFPAUPPER_GBD'].lt(0), 0.)
#     harm['AFPALOWER_GBD'] = harm['AFPALOWER_GBD'].mask(
#         harm['AFPALOWER_GBD'].lt(0), 0.)    
#     # Read IHME GBD asthma incidence rates
#     gbd = pd.read_csv('/Users/ghkerr/GW/data/gbd/'+
#         'IHME-GBD_2019_DATA-9abad4e5-1.csv', sep=',', engine='python')
#     # If DataFrame represents multiple years, calculate burdens separately 
#     # for each year and append to output DataFrame
#     harmout = []
#     for year in np.unique(harm['YEAR']):
#         harmyear = harm.loc[harm['YEAR']==year].copy()
#         # Add columns to be filled with NO2-attributable asthma burdens
#         harmyear[['burden_lt5', 'burden_5-9', 'burden_10-14', 'burden_15-19', 
#             'PEDBURDEN', 'PEDBURDEN100K']] = np.nan
#         # Year of NO2/ACS dataset
#         year = int(harmyear['YEAR'].values[0][-4:])
#         # Loop through states
#         for state in np.unique(harmyear['STATE'].astype(str).values):
#             harmstate = harmyear.loc[harmyear['STATE']==state].copy()
#             # GBD naming for DC is inconsistent with ACS
#             if state=='District Of Columbia':
#                 gbdstate = gbd.loc[gbd['location_name']=='District of Columbia']
#             else: 
#                 gbdstate = gbd.loc[gbd['location_name']==state]
#             # Find incidence for each age category; note that there are not 
#             # incidence rates available for every state for every year, find the 
#             # closest year's rates
#             in14 = gbdstate.loc[gbdstate['age_name']=='1 to 4']
#             in59 = gbdstate.loc[gbdstate['age_name']=='5 to 9']
#             in1014 = gbdstate.loc[gbdstate['age_name']=='10 to 14']
#             in1519 = gbdstate.loc[gbdstate['age_name']=='15 to 19']
#             # For rollback scenarios when GBD incidence rates are fixed at 
#             # reference year values
#             if refyear is None:
#                 in14 = in14.iloc[(in14['year']-year).abs().argsort()[:1]]
#                 in59 = in59.iloc[(in59['year']-year).abs().argsort()[:1]]
#                 in1014 = in1014.iloc[(in1014['year']-year).abs().argsort()[:1]]
#                 in1519 = in1519.iloc[(in1519['year']-year).abs().argsort()[:1]]
#             else:                
#                 in14 = in14.iloc[(in14['year']-refyear).abs().argsort()[:1]]
#                 in59 = in59.iloc[(in59['year']-refyear).abs().argsort()[:1]]
#                 in1014 = in1014.iloc[(in1014['year']-refyear).abs().argsort()[:1]]
#                 in1519 = in1519.iloc[(in1519['year']-refyear).abs().argsort()[:1]]
#             in14 = in14['val'].values[0]/100000.
#             in59 = in59['val'].values[0]/100000.
#             in1014 = in1014['val'].values[0]/100000.
#             in1519 = in1519['val'].values[0]/100000.
#             # Calculate counterfactual asthma burdens for each age group as: 
#             # Burden = Inc_{a} * Pop_{a} * (1 - e^{-rr * 2})
#             # where a = age group and rr = concentration-response function
#             harmstate['burden_lt5'] = (in14*((harmstate['pop_m_lt5']+
#                 harmstate['pop_f_lt5'])*harmstate['AFPA']))
#             harmstate['burden_5-9'] = (in59*((harmstate['pop_m_5-9']+
#                 harmstate['pop_f_5-9'])*harmstate['AFPA']))
#             harmstate['burden_10-14'] = (in1014*((harmstate['pop_m_10-14']+
#                 harmstate['pop_f_10-14'])*harmstate['AFPA']))
#             harmstate['burden_15-19'] = (in1519*((harmstate['pop_m_15-17']+
#                 harmstate['pop_f_15-17']+harmstate['pop_m_18-19']+
#                 harmstate['pop_f_18-19'])*harmstate['AFPA']))
#             # In the case that NO2 concentrations in a census tract are less than 
#             # the counterfactual concentration, set burden to 0 (rather than having)
#             # a negative number
#             harmstate.loc[harmstate['burden_lt5']<0.,'burden_lt5']=0
#             harmstate.loc[harmstate['burden_5-9']<0.,'burden_5-9']=0
#             harmstate.loc[harmstate['burden_10-14']<0.,'burden_10-14']=0
#             harmstate.loc[harmstate['burden_15-19']<0.,'burden_15-19']=0
#             # Create total paediatric burden column (summed over four age 
#             # brackets)
#             harmstate['PEDBURDEN'] = (harmstate['burden_lt5']+
#                 harmstate['burden_5-9']+harmstate['burden_10-14']+
#                 harmstate['burden_15-19'])
#             harmstate['PEDBURDEN100K'] = (harmstate['PEDBURDEN']/(
#                 harmstate['pop_m_lt5']+harmstate['pop_f_lt5']+
#                 harmstate['pop_f_5-9']+harmstate['pop_m_5-9']+
#                 harmstate['pop_f_10-14']+harmstate['pop_m_10-14']+
#                 harmstate['pop_m_15-17']+harmstate['pop_f_15-17']+
#                 harmstate['pop_m_18-19']+harmstate['pop_f_18-19']))*100000
#             harmout.append(harmstate)
#             del harmstate
#     harmout = pd.concat(harmout)
#     return harmout  






    
    
def calculate_pm25no2burden(harm, cfpm=4.15, cfno2=5.3675, cfno2_khreis=2.0):
    """Calculate incidence of PM2.5-attributable diseases (IHD, stroke, lower
    respiratory disease, lung cancer, and type 2 diabetes) and NO2-attributable
    diseases (pediatric asthma) for census tracts in the U.S. using annual 
    population estimates from the U.S. Census Bureau/ACS and baseline disease
    rates from IHME/GBD. 

    Parameters
    ----------
    harm : pandas.core.frame.DataFrame
        Harmonized census tract averaged PM2.5 and NO2 concentrations 
        containing PM2.5- and NO2-attributable fractions for various health
        endpoints for given year(s)
    cfpm : float, optional
        Theoretical minimum risk exposure level (TMREL) for PM2.5 from 
        GBD. The default value of 4.15 µg/m^3; this value is the midpoint
        of the uniform distribution from 2.4 to 5.9 µg/m^3 used in the 2019
        GBD. For more information, see Susan's Dropbox folder) 
        DESCRIPTION. The default is 4.15.
    cfno2 : float, optional
        TMREL for NO2 from GBD. This default value of 5.3675 µg/m^3 is the 
        midpoint of the uniform distribution from X to X µg/m^3 used in the 
        2019 GBD. 
    cfno2_khreis : float, optional
        TMREL from Khreis et al. (2017). The default value 2.0 ppbv which 
        corresponds to the 5th percentile of the minimum exposure 
        concentrations reported in the individual studies considered in 
        Khreis et al. 

    Returns
    -------
    harmout : pandas.core.frame.DataFrame
        Same as harm but containing disease incidence (based on population 
        and baseline disease rates from IHME/GBD) for various health endpoints
        
    References
    ----------
    Khreis H, Kelly C, Tate J, Parslow R, Lucas K, Nieuwenhuijsen M.
    Exposure to traffic-related air pollution and risk of development of
    childhood asthma: a systematic review and meta-analysis. Environ Int 2017; 
    100: 1–31.
    """
    import numpy as np
    import pandas as pd

    # # # # Open GBD asthma, IHD, stroke, COPD, diabetes, LRI, and lung cancer 
    # rates
    asthmarate = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-9abad4e5-1.csv', 
        sep=',', engine='python')
    asthmarate.loc[asthmarate['location_name']=='District Of Columbia', 
        'location_name'] = 'District of Columbia'
    ihdrate = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-cvd_ihd.csv', 
        sep=',', engine='python')
    ihdrate.loc[ihdrate['location_name']=='District Of Columbia', 
        'location_name'] = 'District of Columbia'
    strokerate = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-cvd_stroke.csv', 
        sep=',', engine='python')
    strokerate.loc[strokerate['location_name']=='District Of Columbia', 
        'location_name'] = 'District of Columbia' 
    lrirate = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-lri.csv', 
        sep=',', engine='python')
    lrirate.loc[lrirate['location_name']=='District Of Columbia', 
        'location_name'] = 'District of Columbia'
    lcrate = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-neo_lung.csv', 
        sep=',', engine='python')
    lcrate.loc[lcrate['location_name']=='District Of Columbia', 
        'location_name'] = 'District of Columbia' 
    copdrate = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-resp_copd.csv', 
        sep=',', engine='python')
    copdrate.loc[copdrate['location_name']=='District Of Columbia', 
        'location_name'] = 'District of Columbia' 
    t2dmrate = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-t2_dm.csv', 
        sep=',', engine='python')
    t2dmrate.loc[t2dmrate['location_name']=='District Of Columbia', 
        'location_name'] = 'District of Columbia' 
    
    # # # # Open meta-regression-Bayesian, regularized and trimmed RR
    rrno2asthma = pd.read_csv(DIR_GBD+'no2_rr_draws_summary.csv', sep=',',
        engine='python')
    rrpmihd_25 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_25.csv')
    rrpmihd_30 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_30.csv')
    rrpmihd_35 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_35.csv')
    rrpmihd_40 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_40.csv')
    rrpmihd_45 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_45.csv')
    rrpmihd_50 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_50.csv')
    rrpmihd_55 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_55.csv') 
    rrpmihd_60 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_60.csv')
    rrpmihd_65 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_65.csv')
    rrpmihd_70 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_70.csv')
    rrpmihd_75 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_75.csv')
    rrpmihd_80 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_80.csv')
    rrpmihd_85 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_85.csv')
    rrpmihd_90 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_90.csv')
    rrpmihd_95 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_95.csv')
    rrpmst_25 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_25.csv')
    rrpmst_30 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_30.csv')
    rrpmst_35 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_35.csv')
    rrpmst_40 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_40.csv')
    rrpmst_45 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_45.csv')
    rrpmst_50 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_50.csv')
    rrpmst_55 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_55.csv') 
    rrpmst_60 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_60.csv')
    rrpmst_65 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_65.csv')
    rrpmst_70 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_70.csv')
    rrpmst_75 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_75.csv')
    rrpmst_80 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_80.csv')
    rrpmst_85 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_85.csv')
    rrpmst_90 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_90.csv')
    rrpmst_95 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_95.csv')
    rrpmcopd = pd.read_csv(DIR_GBD+'mrbrt_summary/resp_copd.csv')
    rrpmlc = pd.read_csv(DIR_GBD+'mrbrt_summary/neo_lung.csv')
    rrpmdm = pd.read_csv(DIR_GBD+'mrbrt_summary/t2_dm.csv')
    rrpmlri = pd.read_csv(DIR_GBD+'mrbrt_summary/lri.csv')
    
    # # # # Find closest exposure spline to counterfactual PM2.5 and NO2 
    # concentrations
    cipm = rrpmihd_25['exposure_spline'].sub(cfpm).abs().idxmin()
    cino2 = rrno2asthma['exposure'].sub(cfno2).abs().idxmin()
    cirrno2asthma = rrno2asthma.iloc[cino2]['mean']
    cirrpmihd_25 = rrpmihd_25.iloc[cipm]['mean']
    cirrpmihd_30 = rrpmihd_30.iloc[cipm]['mean']
    cirrpmihd_35 = rrpmihd_35.iloc[cipm]['mean']
    cirrpmihd_40 = rrpmihd_40.iloc[cipm]['mean']
    cirrpmihd_45 = rrpmihd_45.iloc[cipm]['mean']
    cirrpmihd_50 = rrpmihd_50.iloc[cipm]['mean']
    cirrpmihd_55 = rrpmihd_55.iloc[cipm]['mean']
    cirrpmihd_60 = rrpmihd_60.iloc[cipm]['mean']
    cirrpmihd_65 = rrpmihd_65.iloc[cipm]['mean']
    cirrpmihd_70 = rrpmihd_70.iloc[cipm]['mean']
    cirrpmihd_75 = rrpmihd_75.iloc[cipm]['mean']
    cirrpmihd_80 = rrpmihd_80.iloc[cipm]['mean']
    cirrpmihd_85 = rrpmihd_85.iloc[cipm]['mean']
    cirrpmihd_90 = rrpmihd_90.iloc[cipm]['mean']
    cirrpmihd_95 = rrpmihd_95.iloc[cipm]['mean']
    cirrpmst_25 = rrpmst_25.iloc[cipm]['mean']
    cirrpmst_30 = rrpmst_30.iloc[cipm]['mean']
    cirrpmst_35 = rrpmst_35.iloc[cipm]['mean']
    cirrpmst_40 = rrpmst_40.iloc[cipm]['mean']
    cirrpmst_45 = rrpmst_45.iloc[cipm]['mean']
    cirrpmst_50 = rrpmst_50.iloc[cipm]['mean']
    cirrpmst_55 = rrpmst_55.iloc[cipm]['mean']
    cirrpmst_60 = rrpmst_60.iloc[cipm]['mean']
    cirrpmst_65 = rrpmst_65.iloc[cipm]['mean']
    cirrpmst_70 = rrpmst_70.iloc[cipm]['mean']
    cirrpmst_75 = rrpmst_75.iloc[cipm]['mean']
    cirrpmst_80 = rrpmst_80.iloc[cipm]['mean']
    cirrpmst_85 = rrpmst_85.iloc[cipm]['mean']
    cirrpmst_90 = rrpmst_90.iloc[cipm]['mean']
    cirrpmst_95 = rrpmst_95.iloc[cipm]['mean']
    cirrpmcopd = rrpmcopd.iloc[cipm]['mean']
    cirrpmlc = rrpmlc.iloc[cipm]['mean']
    cirrpmdm = rrpmdm.iloc[cipm]['mean']
    cirrpmlri = rrpmlri.iloc[cipm]['mean']
    
    # # # # Calculate PAF for counterfactual, either as log-linear or 
    # (RR-1)/RR
    afno2asthma = (1-np.exp(-cirrno2asthma))
    beta = np.log(1.26)/10.
    afno2asthma_khreis = (1-np.exp(-beta*cfno2_khreis))
    afpmihd_25 = (cirrpmihd_25-1.)/cirrpmihd_25
    afpmihd_30 = (cirrpmihd_30-1.)/cirrpmihd_30
    afpmihd_35 = (cirrpmihd_35-1.)/cirrpmihd_35
    afpmihd_40 = (cirrpmihd_40-1.)/cirrpmihd_40
    afpmihd_45 = (cirrpmihd_45-1.)/cirrpmihd_45
    afpmihd_50 = (cirrpmihd_50-1.)/cirrpmihd_50
    afpmihd_55 = (cirrpmihd_55-1.)/cirrpmihd_55
    afpmihd_60 = (cirrpmihd_60-1.)/cirrpmihd_60
    afpmihd_65 = (cirrpmihd_65-1.)/cirrpmihd_65
    afpmihd_70 = (cirrpmihd_70-1.)/cirrpmihd_70
    afpmihd_75 = (cirrpmihd_75-1.)/cirrpmihd_75
    afpmihd_80 = (cirrpmihd_80-1.)/cirrpmihd_80
    afpmihd_85 = (cirrpmihd_85-1.)/cirrpmihd_85
    afpmihd_90 = (cirrpmihd_90-1.)/cirrpmihd_90
    afpmihd_95 = (cirrpmihd_95-1.)/cirrpmihd_95
    afpmst_25 = (cirrpmst_25-1.)/cirrpmst_25
    afpmst_30 = (cirrpmst_30-1.)/cirrpmst_30
    afpmst_35 = (cirrpmst_35-1.)/cirrpmst_35
    afpmst_40 = (cirrpmst_40-1.)/cirrpmst_40
    afpmst_45 = (cirrpmst_45-1.)/cirrpmst_45
    afpmst_50 = (cirrpmst_50-1.)/cirrpmst_50
    afpmst_55 = (cirrpmst_55-1.)/cirrpmst_55
    afpmst_60 = (cirrpmst_60-1.)/cirrpmst_60
    afpmst_65 = (cirrpmst_65-1.)/cirrpmst_65
    afpmst_70 = (cirrpmst_70-1.)/cirrpmst_70
    afpmst_75 = (cirrpmst_75-1.)/cirrpmst_75
    afpmst_80 = (cirrpmst_80-1.)/cirrpmst_80
    afpmst_85 = (cirrpmst_85-1.)/cirrpmst_85
    afpmst_90 = (cirrpmst_90-1.)/cirrpmst_90
    afpmst_95 = (cirrpmst_95-1.)/cirrpmst_95
    afpmcopd = (cirrpmcopd-1.)/cirrpmcopd
    afpmlc = (cirrpmlc-1.)/cirrpmlc
    afpmdm = (cirrpmdm-1.)/cirrpmdm
    afpmlri = (cirrpmlri-1.)/cirrpmlri 
                   
    # Loop through years and states and calculate IHD burdens attributable 
    # to PM25 (since incidence rates change annually and state by state)
    harmout = []
    for year in np.unique(harm['YEAR']):
        harmyear = harm.loc[harm['YEAR']==year].copy()
        # Subtract off the concentration-response function calculated using the
        # counterfactural concentration 
        harmyear['AFPAMEAN_GBD'] = harmyear['AFPAMEAN_GBD']-afno2asthma
        harmyear['AFPA'] = harmyear['AFPA']-afno2asthma_khreis
        harmyear['AFIHD_25'] = harmyear['AFIHD_25']-afpmihd_25
        harmyear['AFIHD_30'] = harmyear['AFIHD_30']-afpmihd_30
        harmyear['AFIHD_35'] = harmyear['AFIHD_35']-afpmihd_35
        harmyear['AFIHD_40'] = harmyear['AFIHD_40']-afpmihd_40
        harmyear['AFIHD_45'] = harmyear['AFIHD_45']-afpmihd_45
        harmyear['AFIHD_50'] = harmyear['AFIHD_50']-afpmihd_50
        harmyear['AFIHD_55'] = harmyear['AFIHD_55']-afpmihd_55
        harmyear['AFIHD_60'] = harmyear['AFIHD_60']-afpmihd_60
        harmyear['AFIHD_65'] = harmyear['AFIHD_65']-afpmihd_65
        harmyear['AFIHD_70'] = harmyear['AFIHD_70']-afpmihd_70
        harmyear['AFIHD_75'] = harmyear['AFIHD_75']-afpmihd_75
        harmyear['AFIHD_80'] = harmyear['AFIHD_80']-afpmihd_80
        harmyear['AFIHD_85'] = harmyear['AFIHD_85']-afpmihd_85
        harmyear['AFIHD_90'] = harmyear['AFIHD_90']-afpmihd_90
        harmyear['AFIHD_95'] = harmyear['AFIHD_95']-afpmihd_95
        harmyear['AFST_25'] = harmyear['AFST_25']-afpmst_25
        harmyear['AFST_30'] = harmyear['AFST_30']-afpmst_30
        harmyear['AFST_35'] = harmyear['AFST_35']-afpmst_35
        harmyear['AFST_40'] = harmyear['AFST_40']-afpmst_40
        harmyear['AFST_45'] = harmyear['AFST_45']-afpmst_45
        harmyear['AFST_50'] = harmyear['AFST_50']-afpmst_50
        harmyear['AFST_55'] = harmyear['AFST_55']-afpmst_55
        harmyear['AFST_60'] = harmyear['AFST_60']-afpmst_60
        harmyear['AFST_65'] = harmyear['AFST_65']-afpmst_65
        harmyear['AFST_70'] = harmyear['AFST_70']-afpmst_70
        harmyear['AFST_75'] = harmyear['AFST_75']-afpmst_75
        harmyear['AFST_80'] = harmyear['AFST_80']-afpmst_80
        harmyear['AFST_85'] = harmyear['AFST_85']-afpmst_85
        harmyear['AFST_90'] = harmyear['AFST_90']-afpmst_90    
        harmyear['AFST_95'] = harmyear['AFST_95']-afpmst_95
        harmyear['AFCOPD'] = harmyear['AFCOPD']-afpmcopd
        harmyear['AFLC'] = harmyear['AFLC']-afpmlc
        harmyear['AFDM'] = harmyear['AFDM']-afpmdm
        harmyear['AFLRI'] = harmyear['AFLRI']-afpmlri
        # Sensitivity simulations
        harmyear['AFPAWHO40'] = harmyear['AFPAWHO40']-afno2asthma_khreis
        harmyear['AFPAWHO30'] = harmyear['AFPAWHO30']-afno2asthma_khreis
        harmyear['AFPAWHO20'] = harmyear['AFPAWHO20']-afno2asthma_khreis
        harmyear['AFPAWHO10'] = harmyear['AFPAWHO10']-afno2asthma_khreis
        harmyear['AFPAMEAN_GBDWHO40'] = harmyear['AFPAMEAN_GBDWHO40']-afno2asthma
        harmyear['AFPAMEAN_GBDWHO30'] = harmyear['AFPAMEAN_GBDWHO30']-afno2asthma
        harmyear['AFPAMEAN_GBDWHO20'] = harmyear['AFPAMEAN_GBDWHO20']-afno2asthma
        harmyear['AFPAMEAN_GBDWHO10'] = harmyear['AFPAMEAN_GBDWHO10']-afno2asthma
        harmyear['AFIHD_25WHO15'] = harmyear['AFIHD_25WHO15']-afpmihd_25
        harmyear['AFIHD_30WHO15'] = harmyear['AFIHD_30WHO15']-afpmihd_30
        harmyear['AFIHD_35WHO15'] = harmyear['AFIHD_35WHO15']-afpmihd_35
        harmyear['AFIHD_40WHO15'] = harmyear['AFIHD_40WHO15']-afpmihd_40
        harmyear['AFIHD_45WHO15'] = harmyear['AFIHD_45WHO15']-afpmihd_45
        harmyear['AFIHD_50WHO15'] = harmyear['AFIHD_50WHO15']-afpmihd_50
        harmyear['AFIHD_55WHO15'] = harmyear['AFIHD_55WHO15']-afpmihd_55
        harmyear['AFIHD_60WHO15'] = harmyear['AFIHD_60WHO15']-afpmihd_60
        harmyear['AFIHD_65WHO15'] = harmyear['AFIHD_65WHO15']-afpmihd_65
        harmyear['AFIHD_70WHO15'] = harmyear['AFIHD_70WHO15']-afpmihd_70
        harmyear['AFIHD_75WHO15'] = harmyear['AFIHD_75WHO15']-afpmihd_75
        harmyear['AFIHD_80WHO15'] = harmyear['AFIHD_80WHO15']-afpmihd_80
        harmyear['AFIHD_85WHO15'] = harmyear['AFIHD_85WHO15']-afpmihd_85
        harmyear['AFIHD_90WHO15'] = harmyear['AFIHD_90WHO15']-afpmihd_90
        harmyear['AFIHD_95WHO15'] = harmyear['AFIHD_95WHO15']-afpmihd_95
        harmyear['AFST_25WHO15'] = harmyear['AFST_25WHO15']-afpmst_25
        harmyear['AFST_30WHO15'] = harmyear['AFST_30WHO15']-afpmst_30
        harmyear['AFST_35WHO15'] = harmyear['AFST_35WHO15']-afpmst_35
        harmyear['AFST_40WHO15'] = harmyear['AFST_40WHO15']-afpmst_40
        harmyear['AFST_45WHO15'] = harmyear['AFST_45WHO15']-afpmst_45
        harmyear['AFST_50WHO15'] = harmyear['AFST_50WHO15']-afpmst_50
        harmyear['AFST_55WHO15'] = harmyear['AFST_55WHO15']-afpmst_55
        harmyear['AFST_60WHO15'] = harmyear['AFST_60WHO15']-afpmst_60
        harmyear['AFST_65WHO15'] = harmyear['AFST_65WHO15']-afpmst_65
        harmyear['AFST_70WHO15'] = harmyear['AFST_70WHO15']-afpmst_70
        harmyear['AFST_75WHO15'] = harmyear['AFST_75WHO15']-afpmst_75
        harmyear['AFST_80WHO15'] = harmyear['AFST_80WHO15']-afpmst_80
        harmyear['AFST_85WHO15'] = harmyear['AFST_85WHO15']-afpmst_85
        harmyear['AFST_90WHO15'] = harmyear['AFST_90WHO15']-afpmst_90
        harmyear['AFST_95WHO15'] = harmyear['AFST_95WHO15']-afpmst_95
        harmyear['AFCOPDWHO15'] = harmyear['AFCOPDWHO15']-afpmcopd
        harmyear['AFLCWHO15'] = harmyear['AFLCWHO15']-afpmlc
        harmyear['AFDMWHO15'] = harmyear['AFDMWHO15']-afpmdm
        harmyear['AFLRIWHO15'] = harmyear['AFLRIWHO15']-afpmlri
        harmyear['AFIHD_25NAAQS12'] = harmyear['AFIHD_25NAAQS12']-afpmihd_25
        harmyear['AFIHD_30NAAQS12'] = harmyear['AFIHD_30NAAQS12']-afpmihd_30
        harmyear['AFIHD_35NAAQS12'] = harmyear['AFIHD_35NAAQS12']-afpmihd_35
        harmyear['AFIHD_40NAAQS12'] = harmyear['AFIHD_40NAAQS12']-afpmihd_40
        harmyear['AFIHD_45NAAQS12'] = harmyear['AFIHD_45NAAQS12']-afpmihd_45
        harmyear['AFIHD_50NAAQS12'] = harmyear['AFIHD_50NAAQS12']-afpmihd_50
        harmyear['AFIHD_55NAAQS12'] = harmyear['AFIHD_55NAAQS12']-afpmihd_55
        harmyear['AFIHD_60NAAQS12'] = harmyear['AFIHD_60NAAQS12']-afpmihd_60
        harmyear['AFIHD_65NAAQS12'] = harmyear['AFIHD_65NAAQS12']-afpmihd_65
        harmyear['AFIHD_70NAAQS12'] = harmyear['AFIHD_70NAAQS12']-afpmihd_70
        harmyear['AFIHD_75NAAQS12'] = harmyear['AFIHD_75NAAQS12']-afpmihd_75
        harmyear['AFIHD_80NAAQS12'] = harmyear['AFIHD_80NAAQS12']-afpmihd_80
        harmyear['AFIHD_85NAAQS12'] = harmyear['AFIHD_85NAAQS12']-afpmihd_85
        harmyear['AFIHD_90NAAQS12'] = harmyear['AFIHD_90NAAQS12']-afpmihd_90
        harmyear['AFIHD_95NAAQS12'] = harmyear['AFIHD_95NAAQS12']-afpmihd_95
        harmyear['AFST_25NAAQS12'] = harmyear['AFST_25NAAQS12']-afpmst_25
        harmyear['AFST_30NAAQS12'] = harmyear['AFST_30NAAQS12']-afpmst_30
        harmyear['AFST_35NAAQS12'] = harmyear['AFST_35NAAQS12']-afpmst_35
        harmyear['AFST_40NAAQS12'] = harmyear['AFST_40NAAQS12']-afpmst_40
        harmyear['AFST_45NAAQS12'] = harmyear['AFST_45NAAQS12']-afpmst_45
        harmyear['AFST_50NAAQS12'] = harmyear['AFST_50NAAQS12']-afpmst_50
        harmyear['AFST_55NAAQS12'] = harmyear['AFST_55NAAQS12']-afpmst_55
        harmyear['AFST_60NAAQS12'] = harmyear['AFST_60NAAQS12']-afpmst_60
        harmyear['AFST_65NAAQS12'] = harmyear['AFST_65NAAQS12']-afpmst_65
        harmyear['AFST_70NAAQS12'] = harmyear['AFST_70NAAQS12']-afpmst_70
        harmyear['AFST_75NAAQS12'] = harmyear['AFST_75NAAQS12']-afpmst_75
        harmyear['AFST_80NAAQS12'] = harmyear['AFST_80NAAQS12']-afpmst_80
        harmyear['AFST_85NAAQS12'] = harmyear['AFST_85NAAQS12']-afpmst_85
        harmyear['AFST_90NAAQS12'] = harmyear['AFST_90NAAQS12']-afpmst_90
        harmyear['AFST_95NAAQS12'] = harmyear['AFST_95NAAQS12']-afpmst_95
        harmyear['AFCOPDNAAQS12'] = harmyear['AFCOPDNAAQS12']-afpmcopd
        harmyear['AFLCNAAQS12'] = harmyear['AFLCNAAQS12']-afpmlc
        harmyear['AFDMNAAQS12'] = harmyear['AFDMNAAQS12']-afpmdm
        harmyear['AFLRINAAQS12'] =  harmyear['AFLRINAAQS12']-afpmlri
        harmyear['AFIHD_25WHO10'] = harmyear['AFIHD_25WHO10']-afpmihd_25
        harmyear['AFIHD_30WHO10'] = harmyear['AFIHD_30WHO10']-afpmihd_30
        harmyear['AFIHD_35WHO10'] = harmyear['AFIHD_35WHO10']-afpmihd_35
        harmyear['AFIHD_40WHO10'] = harmyear['AFIHD_40WHO10']-afpmihd_40
        harmyear['AFIHD_45WHO10'] = harmyear['AFIHD_45WHO10']-afpmihd_45
        harmyear['AFIHD_50WHO10'] = harmyear['AFIHD_50WHO10']-afpmihd_50
        harmyear['AFIHD_55WHO10'] = harmyear['AFIHD_55WHO10']-afpmihd_55
        harmyear['AFIHD_60WHO10'] = harmyear['AFIHD_60WHO10']-afpmihd_60
        harmyear['AFIHD_65WHO10'] = harmyear['AFIHD_65WHO10']-afpmihd_65
        harmyear['AFIHD_70WHO10'] = harmyear['AFIHD_70WHO10']-afpmihd_70
        harmyear['AFIHD_75WHO10'] = harmyear['AFIHD_75WHO10']-afpmihd_75
        harmyear['AFIHD_80WHO10'] = harmyear['AFIHD_80WHO10']-afpmihd_80
        harmyear['AFIHD_85WHO10'] = harmyear['AFIHD_85WHO10']-afpmihd_85
        harmyear['AFIHD_90WHO10'] = harmyear['AFIHD_90WHO10']-afpmihd_90
        harmyear['AFIHD_95WHO10'] = harmyear['AFIHD_95WHO10']-afpmihd_95
        harmyear['AFST_25WHO10'] = harmyear['AFST_25WHO10']-afpmst_25
        harmyear['AFST_30WHO10'] = harmyear['AFST_30WHO10']-afpmst_30
        harmyear['AFST_35WHO10'] = harmyear['AFST_35WHO10']-afpmst_35
        harmyear['AFST_40WHO10'] = harmyear['AFST_40WHO10']-afpmst_40
        harmyear['AFST_45WHO10'] = harmyear['AFST_45WHO10']-afpmst_45
        harmyear['AFST_50WHO10'] = harmyear['AFST_50WHO10']-afpmst_50
        harmyear['AFST_55WHO10'] = harmyear['AFST_55WHO10']-afpmst_55
        harmyear['AFST_60WHO10'] = harmyear['AFST_60WHO10']-afpmst_60
        harmyear['AFST_65WHO10'] = harmyear['AFST_65WHO10']-afpmst_65
        harmyear['AFST_70WHO10'] = harmyear['AFST_70WHO10']-afpmst_70
        harmyear['AFST_75WHO10'] = harmyear['AFST_75WHO10']-afpmst_75
        harmyear['AFST_80WHO10'] = harmyear['AFST_80WHO10']-afpmst_80
        harmyear['AFST_85WHO10'] = harmyear['AFST_85WHO10']-afpmst_85
        harmyear['AFST_90WHO10'] = harmyear['AFST_90WHO10']-afpmst_90
        harmyear['AFST_95WHO10'] = harmyear['AFST_95WHO10']-afpmst_95
        harmyear['AFCOPDWHO10'] = harmyear['AFCOPDWHO10']-afpmcopd
        harmyear['AFLCWHO10'] = harmyear['AFLCWHO10']-afpmlc
        harmyear['AFDMWHO10'] = harmyear['AFDMWHO10']-afpmdm
        harmyear['AFLRIWHO10'] = harmyear['AFLRIWHO10']-afpmlri
        harmyear['AFIHD_25WHO5'] = harmyear['AFIHD_25WHO5']-afpmihd_25
        harmyear['AFIHD_30WHO5'] = harmyear['AFIHD_30WHO5']-afpmihd_30
        harmyear['AFIHD_35WHO5'] = harmyear['AFIHD_35WHO5']-afpmihd_35
        harmyear['AFIHD_40WHO5'] = harmyear['AFIHD_40WHO5']-afpmihd_40
        harmyear['AFIHD_45WHO5'] = harmyear['AFIHD_45WHO5']-afpmihd_45
        harmyear['AFIHD_50WHO5'] = harmyear['AFIHD_50WHO5']-afpmihd_50
        harmyear['AFIHD_55WHO5'] = harmyear['AFIHD_55WHO5']-afpmihd_55
        harmyear['AFIHD_60WHO5'] = harmyear['AFIHD_60WHO5']-afpmihd_60
        harmyear['AFIHD_65WHO5'] = harmyear['AFIHD_65WHO5']-afpmihd_65
        harmyear['AFIHD_70WHO5'] = harmyear['AFIHD_70WHO5']-afpmihd_70
        harmyear['AFIHD_75WHO5'] = harmyear['AFIHD_75WHO5']-afpmihd_75
        harmyear['AFIHD_80WHO5'] = harmyear['AFIHD_80WHO5']-afpmihd_80
        harmyear['AFIHD_85WHO5'] = harmyear['AFIHD_85WHO5']-afpmihd_85
        harmyear['AFIHD_90WHO5'] = harmyear['AFIHD_90WHO5']-afpmihd_90
        harmyear['AFIHD_95WHO5'] = harmyear['AFIHD_95WHO5']-afpmihd_95
        harmyear['AFST_25WHO5'] = harmyear['AFST_25WHO5']-afpmst_25
        harmyear['AFST_30WHO5'] = harmyear['AFST_30WHO5']-afpmst_30
        harmyear['AFST_35WHO5'] = harmyear['AFST_35WHO5']-afpmst_35
        harmyear['AFST_40WHO5'] = harmyear['AFST_40WHO5']-afpmst_40
        harmyear['AFST_45WHO5'] = harmyear['AFST_45WHO5']-afpmst_45
        harmyear['AFST_50WHO5'] = harmyear['AFST_50WHO5']-afpmst_50
        harmyear['AFST_55WHO5'] = harmyear['AFST_55WHO5']-afpmst_55
        harmyear['AFST_60WHO5'] = harmyear['AFST_60WHO5']-afpmst_60
        harmyear['AFST_65WHO5'] = harmyear['AFST_65WHO5']-afpmst_65
        harmyear['AFST_70WHO5'] = harmyear['AFST_70WHO5']-afpmst_70
        harmyear['AFST_75WHO5'] = harmyear['AFST_75WHO5']-afpmst_75
        harmyear['AFST_80WHO5'] = harmyear['AFST_80WHO5']-afpmst_80
        harmyear['AFST_85WHO5'] = harmyear['AFST_85WHO5']-afpmst_85
        harmyear['AFST_90WHO5'] = harmyear['AFST_90WHO5']-afpmst_90
        harmyear['AFST_95WHO5'] = harmyear['AFST_95WHO5']-afpmst_95
        harmyear['AFCOPDWHO5'] = harmyear['AFCOPDWHO5']-afpmcopd
        harmyear['AFLCWHO5'] = harmyear['AFLCWHO5']-afpmlc
        harmyear['AFDMWHO5'] = harmyear['AFDMWHO5']-afpmdm
        harmyear['AFLRIWHO5'] = harmyear['AFLRIWHO5']-afpmlri
        
        # If subtracting off the counterfactual makes a census tract have a 
        # PAF < 0, set that PAF to 0 
        cols = ['AFPA', 'AFPAWHO40', 'AFPAWHO30', 'AFPAWHO20', 'AFPAWHO10',
            'AFPAMEAN_GBD', 'AFPAMEAN_GBDWHO40', 'AFPAMEAN_GBDWHO30', 
            'AFPAMEAN_GBDWHO20', 'AFPAMEAN_GBDWHO10', 'AFIHD_25', 
            'AFIHD_30', 'AFIHD_35', 'AFIHD_40', 'AFIHD_45', 'AFIHD_50', 
            'AFIHD_55', 'AFIHD_60', 'AFIHD_65', 'AFIHD_70', 'AFIHD_75',
            'AFIHD_80', 'AFIHD_85', 'AFIHD_90', 'AFIHD_95', 'AFST_25', 
            'AFST_30', 'AFST_35', 'AFST_40', 'AFST_45', 'AFST_50', 
            'AFST_55', 'AFST_60', 'AFST_65', 'AFST_70', 'AFST_75', 
            'AFST_80', 'AFST_85', 'AFST_90', 'AFST_95', 'AFCOPD', 'AFLC', 
            'AFDM', 'AFLRI', 'AFIHD_25WHO15', 'AFIHD_30WHO15', 'AFIHD_35WHO15', 
            'AFIHD_40WHO15', 'AFIHD_45WHO15', 'AFIHD_50WHO15', 'AFIHD_55WHO15', 
            'AFIHD_60WHO15', 'AFIHD_65WHO15', 'AFIHD_70WHO15', 'AFIHD_75WHO15', 
            'AFIHD_80WHO15', 'AFIHD_85WHO15', 'AFIHD_90WHO15', 'AFIHD_95WHO15', 
            'AFST_25WHO15', 'AFST_30WHO15', 'AFST_35WHO15', 'AFST_40WHO15', 
            'AFST_45WHO15', 'AFST_50WHO15', 'AFST_55WHO15', 'AFST_60WHO15', 
            'AFST_65WHO15', 'AFST_70WHO15', 'AFST_75WHO15', 'AFST_80WHO15', 
            'AFST_85WHO15', 'AFST_90WHO15', 'AFST_95WHO15', 'AFCOPDWHO15', 
            'AFLCWHO15', 'AFDMWHO15', 'AFLRIWHO15','AFIHD_25NAAQS12', 
            'AFIHD_30NAAQS12', 
            'AFIHD_35NAAQS12', 'AFIHD_40NAAQS12', 'AFIHD_45NAAQS12', 
            'AFIHD_50NAAQS12', 'AFIHD_55NAAQS12', 'AFIHD_60NAAQS12', 
            'AFIHD_65NAAQS12', 'AFIHD_70NAAQS12', 'AFIHD_75NAAQS12', 
            'AFIHD_80NAAQS12', 'AFIHD_85NAAQS12', 'AFIHD_90NAAQS12', 
            'AFIHD_95NAAQS12', 'AFST_25NAAQS12', 'AFST_30NAAQS12', 
            'AFST_35NAAQS12', 'AFST_40NAAQS12', 'AFST_45NAAQS12', 
            'AFST_50NAAQS12', 'AFST_55NAAQS12', 'AFST_60NAAQS12',
            'AFST_65NAAQS12', 'AFST_70NAAQS12', 'AFST_75NAAQS12', 
            'AFST_80NAAQS12', 'AFST_85NAAQS12', 'AFST_90NAAQS12', 
            'AFST_95NAAQS12', 'AFCOPDNAAQS12', 'AFLCNAAQS12', 
            'AFDMNAAQS12', 'AFLRINAAQS12', 'AFIHD_25WHO10', 
            'AFIHD_30WHO10', 'AFIHD_35WHO10', 'AFIHD_40WHO10',
            'AFIHD_45WHO10', 'AFIHD_50WHO10', 'AFIHD_55WHO10', 
            'AFIHD_60WHO10', 'AFIHD_65WHO10', 'AFIHD_70WHO10', 
            'AFIHD_75WHO10', 'AFIHD_80WHO10', 'AFIHD_85WHO10', 
            'AFIHD_90WHO10', 'AFIHD_95WHO10', 'AFST_25WHO10',
            'AFST_30WHO10', 'AFST_35WHO10', 'AFST_40WHO10', 'AFST_45WHO10',
            'AFST_50WHO10', 'AFST_55WHO10', 'AFST_60WHO10', 'AFST_65WHO10',
            'AFST_70WHO10', 'AFST_75WHO10', 'AFST_80WHO10', 'AFST_85WHO10',
            'AFST_90WHO10', 'AFST_95WHO10', 'AFCOPDWHO10', 'AFLCWHO10', 
            'AFDMWHO10', 'AFLRIWHO10', 'AFIHD_25WHO5', 'AFIHD_30WHO5', 
            'AFIHD_35WHO5','AFIHD_40WHO5', 'AFIHD_45WHO5', 'AFIHD_50WHO5', 
            'AFIHD_55WHO5', 'AFIHD_60WHO5', 'AFIHD_65WHO5', 'AFIHD_70WHO5', 
            'AFIHD_75WHO5', 'AFIHD_80WHO5', 'AFIHD_85WHO5', 'AFIHD_90WHO5', 
            'AFIHD_95WHO5', 'AFST_25WHO5', 'AFST_30WHO5', 'AFST_35WHO5', 
            'AFST_40WHO5', 'AFST_45WHO5', 'AFST_50WHO5', 'AFST_55WHO5', 
            'AFST_60WHO5', 'AFST_65WHO5', 'AFST_70WHO5', 'AFST_75WHO5', 
            'AFST_80WHO5', 'AFST_85WHO5', 'AFST_90WHO5', 'AFST_95WHO5', 
            'AFCOPDWHO5', 'AFLCWHO5', 'AFDMWHO5', 'AFLRIWHO5']
        for x in cols: 
            harmyear[x] = harmyear[x].mask(harmyear[x].lt(0), 0.)
        # Year of NO2/ACS dataset
        year = int(harmyear['YEAR'].values[0][-4:])
        # Loop through states, 
        for state in np.unique(harmyear['STATE'].astype(str).values):
            harmstate = harmyear.loc[harmyear['STATE']==state].copy()
            # Pediatric asthma incidence (from GBD and Khreis et al., 2017)
            asthmastate = asthmarate.loc[(asthmarate['location_name']==state) &
                (asthmarate['year']==year)]
            harmstate['BURDENASTHMA_LT5'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBD'])
            harmstate['BURDENASTHMA_5'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBD'])        
            harmstate['BURDENASTHMA_10'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBD'])        
            harmstate['BURDENASTHMA_15'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAMEAN_GBD'])        
            harmstate['BURDENASTHMA'] = harmstate[['BURDENASTHMA_LT5',
                'BURDENASTHMA_5', 'BURDENASTHMA_10', 'BURDENASTHMA_15']
                ].sum(axis=1)
            harmstate['BURDENASTHMARATE'] = (harmstate[
                'BURDENASTHMA']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.                        
            harmstate['BURDENASTHMA_LT5WHO40'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO40'])
            harmstate['BURDENASTHMA_5WHO40'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO40'])        
            harmstate['BURDENASTHMA_10WHO40'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO40'])        
            harmstate['BURDENASTHMA_15WHO40'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAMEAN_GBDWHO40'])        
            harmstate['BURDENASTHMAWHO40'] = harmstate[[
                'BURDENASTHMA_LT5WHO40', 'BURDENASTHMA_5WHO40', 
                'BURDENASTHMA_10WHO40', 'BURDENASTHMA_15WHO40']].sum(axis=1)
            harmstate['BURDENASTHMARATEWHO40'] = (harmstate[
                'BURDENASTHMAWHO40']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.                        
            harmstate['BURDENASTHMA_LT5WHO30'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO30'])
            harmstate['BURDENASTHMA_5WHO30'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO30'])        
            harmstate['BURDENASTHMA_10WHO30'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO30'])   
            harmstate['BURDENASTHMA_15WHO30'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAMEAN_GBDWHO30'])        
            harmstate['BURDENASTHMAWHO30'] = harmstate[[
                'BURDENASTHMA_LT5WHO30', 'BURDENASTHMA_5WHO30', 
                'BURDENASTHMA_10WHO30', 'BURDENASTHMA_15WHO30']].sum(axis=1)
            harmstate['BURDENASTHMARATEWHO30'] = (harmstate[
                'BURDENASTHMAWHO30']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.              
            harmstate['BURDENASTHMA_LT5WHO20'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO20'])
            harmstate['BURDENASTHMA_5WHO20'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO20'])        
            harmstate['BURDENASTHMA_10WHO20'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO20'])        
            harmstate['BURDENASTHMA_15WHO20'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAMEAN_GBDWHO20'])        
            harmstate['BURDENASTHMAWHO20'] = harmstate[[
                'BURDENASTHMA_LT5WHO20', 'BURDENASTHMA_5WHO20', 
                'BURDENASTHMA_10WHO20', 'BURDENASTHMA_15WHO20']].sum(axis=1)
            harmstate['BURDENASTHMARATEWHO20'] = (harmstate[
                'BURDENASTHMAWHO20']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.                                                   
            harmstate['BURDENASTHMA_LT5WHO10'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO10'])
            harmstate['BURDENASTHMA_5WHO10'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO10'])
            harmstate['BURDENASTHMA_10WHO10'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAMEAN_GBDWHO10'])        
            harmstate['BURDENASTHMA_15WHO10'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAMEAN_GBDWHO10'])        
            harmstate['BURDENASTHMAWHO10'] = harmstate[[
                'BURDENASTHMA_LT5WHO10', 'BURDENASTHMA_5WHO10', 
                'BURDENASTHMA_10WHO10', 'BURDENASTHMA_15WHO10']].sum(axis=1)
            harmstate['BURDENASTHMARATEWHO10'] = (harmstate[
                'BURDENASTHMAWHO10']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.   
            harmstate['BURDENASTHMA_LT5_KHREIS'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPA'])
            harmstate['BURDENASTHMA_5_KHREIS'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPA'])        
            harmstate['BURDENASTHMA_10_KHREIS'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPA'])        
            harmstate['BURDENASTHMA_15_KHREIS'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPA'])  
            harmstate['BURDENASTHMA_KHREIS'] = harmstate[[
                'BURDENASTHMA_LT5_KHREIS', 'BURDENASTHMA_5_KHREIS', 
                'BURDENASTHMA_10_KHREIS', 'BURDENASTHMA_15_KHREIS']
                ].sum(axis=1)
            harmstate['BURDENASTHMA_KHREISRATE'] = (harmstate[
                'BURDENASTHMA_KHREIS']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.  
            harmstate['BURDENASTHMA_LT5_KHREISWHO40'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAWHO40'])
            harmstate['BURDENASTHMA_5_KHREISWHO40'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAWHO40'])        
            harmstate['BURDENASTHMA_10_KHREISWHO40'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAWHO40'])        
            harmstate['BURDENASTHMA_15_KHREISWHO40'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAWHO40'])  
            harmstate['BURDENASTHMA_KHREISWHO40'] = harmstate[[
                'BURDENASTHMA_LT5_KHREISWHO40', 'BURDENASTHMA_5_KHREISWHO40', 
                'BURDENASTHMA_10_KHREISWHO40', 'BURDENASTHMA_15_KHREISWHO40']
                ].sum(axis=1)
            harmstate['BURDENASTHMA_KHREISRATEWHO40'] = (harmstate[
                'BURDENASTHMA_KHREISWHO40']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000. 
            harmstate['BURDENASTHMA_LT5_KHREISWHO30'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAWHO30'])
            harmstate['BURDENASTHMA_5_KHREISWHO30'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAWHO30'])        
            harmstate['BURDENASTHMA_10_KHREISWHO30'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAWHO30'])        
            harmstate['BURDENASTHMA_15_KHREISWHO30'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAWHO30'])  
            harmstate['BURDENASTHMA_KHREISWHO30'] = harmstate[[
                'BURDENASTHMA_LT5_KHREISWHO30', 'BURDENASTHMA_5_KHREISWHO30', 
                'BURDENASTHMA_10_KHREISWHO30', 'BURDENASTHMA_15_KHREISWHO30']
                ].sum(axis=1)
            harmstate['BURDENASTHMA_KHREISRATEWHO30'] = (harmstate[
                'BURDENASTHMA_KHREISWHO30']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.
            harmstate['BURDENASTHMA_LT5_KHREISWHO20'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAWHO20'])
            harmstate['BURDENASTHMA_5_KHREISWHO20'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAWHO20'])        
            harmstate['BURDENASTHMA_10_KHREISWHO20'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAWHO20'])        
            harmstate['BURDENASTHMA_15_KHREISWHO20'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAWHO20'])  
            harmstate['BURDENASTHMA_KHREISWHO20'] = harmstate[[
                'BURDENASTHMA_LT5_KHREISWHO20', 'BURDENASTHMA_5_KHREISWHO20', 
                'BURDENASTHMA_10_KHREISWHO20', 'BURDENASTHMA_15_KHREISWHO20']
                ].sum(axis=1)
            harmstate['BURDENASTHMA_KHREISRATEWHO20'] = (harmstate[
                'BURDENASTHMA_KHREISWHO20']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.                                                           
            harmstate['BURDENASTHMA_LT5_KHREISWHO10'] = (
                (harmstate[['pop_m_lt5','pop_f_lt5']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='1 to 4'
                ].val.values[0]/100000*harmstate['AFPAWHO10'])
            harmstate['BURDENASTHMA_5_KHREISWHO10'] = (
                (harmstate[['pop_m_5-9','pop_f_5-9']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='5 to 9'
                ].val.values[0]/100000*harmstate['AFPAWHO10'])        
            harmstate['BURDENASTHMA_10_KHREISWHO10'] = (
                (harmstate[['pop_m_10-14','pop_f_10-14']].sum(axis=1))*
                asthmastate.loc[asthmastate['age_name']=='10 to 14'
                ].val.values[0]/100000*harmstate['AFPAWHO10'])        
            harmstate['BURDENASTHMA_15_KHREISWHO10'] = (
                (harmstate[['pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 
                'pop_f_18-19']].sum(axis=1))*asthmastate.loc[
                asthmastate['age_name']=='15 to 19'].val.values[0]/100000*
                harmstate['AFPAWHO10'])  
            harmstate['BURDENASTHMA_KHREISWHO10'] = harmstate[[
                'BURDENASTHMA_LT5_KHREISWHO10', 'BURDENASTHMA_5_KHREISWHO10', 
                'BURDENASTHMA_10_KHREISWHO10', 'BURDENASTHMA_15_KHREISWHO10']
                ].sum(axis=1)
            harmstate['BURDENASTHMA_KHREISRATEWHO10'] = (harmstate[
                'BURDENASTHMA_KHREISWHO10']/harmstate[['pop_m_lt5','pop_f_lt5',
                'pop_m_5-9','pop_f_5-9','pop_m_10-14','pop_f_10-14',
                'pop_m_15-17','pop_f_15-17', 'pop_m_18-19', 'pop_f_18-19'
                ]].sum(axis=1))*100000.                                                           

            # IHD incidence for different age groups
            ihdstate = ihdrate.loc[(ihdrate['location_name']==state) &
                (ihdrate['year']==year)]
            harmstate['BURDENIHD_25'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFIHD_25'])
            harmstate['BURDENIHD_30'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFIHD_30'])
            harmstate['BURDENIHD_35'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFIHD_35'])    
            harmstate['BURDENIHD_40'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFIHD_40'])
            harmstate['BURDENIHD_45'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFIHD_45'])
            harmstate['BURDENIHD_50'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFIHD_50'])
            harmstate['BURDENIHD_55'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFIHD_55'])
            harmstate['BURDENIHD_60'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFIHD_60'])
            harmstate['BURDENIHD_65'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFIHD_65'])        
            harmstate['BURDENIHD_70'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFIHD_70'])
            harmstate['BURDENIHD_75'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFIHD_75']) 
            harmstate['BURDENIHD_80'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFIHD_80'])
            # Since the U.S. Census Bureau population estimates are not 
            # stratified by 5-year intervals for ages > 85 years old, 
            # we handle all the population > 85 years old together using 
            # commensurate IHD rates from GBD and the *average* RR for 
            # the 85-85, 90-94, and >95 year old age groups
            harmstate['BURDENIHD_85'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFIHD_85','AFIHD_90','AFIHD_95']].mean(axis=1))
            # Generate a column representing the total IHD burden and burden
            # per 100K population 
            harmstate['BURDENIHD'] = harmstate[['BURDENIHD_25',
                'BURDENIHD_30', 'BURDENIHD_35', 'BURDENIHD_40', 'BURDENIHD_45',
                'BURDENIHD_50', 'BURDENIHD_55', 'BURDENIHD_60', 'BURDENIHD_65',
                'BURDENIHD_70', 'BURDENIHD_75', 'BURDENIHD_80', 'BURDENIHD_85']
                ].sum(axis=1)
            harmstate['BURDENIHDRATE'] = (harmstate['BURDENIHD']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            harmstate['BURDENIHD_25WHO15'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFIHD_25WHO15'])
            harmstate['BURDENIHD_30WHO15'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFIHD_30WHO15'])
            harmstate['BURDENIHD_35WHO15'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFIHD_35WHO15'])    
            harmstate['BURDENIHD_40WHO15'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFIHD_40WHO15'])
            harmstate['BURDENIHD_45WHO15'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFIHD_45WHO15'])
            harmstate['BURDENIHD_50WHO15'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFIHD_50WHO15'])
            harmstate['BURDENIHD_55WHO15'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFIHD_55WHO15'])
            harmstate['BURDENIHD_60WHO15'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFIHD_60WHO15'])
            harmstate['BURDENIHD_65WHO15'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFIHD_65WHO15'])        
            harmstate['BURDENIHD_70WHO15'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFIHD_70WHO15'])
            harmstate['BURDENIHD_75WHO15'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFIHD_75WHO15']) 
            harmstate['BURDENIHD_80WHO15'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFIHD_80WHO15'])
            harmstate['BURDENIHD_85WHO15'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFIHD_85WHO15','AFIHD_90WHO15','AFIHD_95WHO15']].mean(axis=1))
            harmstate['BURDENIHDWHO15'] = harmstate[['BURDENIHD_25WHO15',
                'BURDENIHD_30WHO15', 'BURDENIHD_35WHO15', 'BURDENIHD_40WHO15', 
                'BURDENIHD_45WHO15', 'BURDENIHD_50WHO15', 'BURDENIHD_55WHO15', 
                'BURDENIHD_60WHO15', 'BURDENIHD_65WHO15', 'BURDENIHD_70WHO15', 
                'BURDENIHD_75WHO15', 'BURDENIHD_80WHO15', 'BURDENIHD_85WHO15']
                ].sum(axis=1)
            harmstate['BURDENIHDRATEWHO15'] = (harmstate['BURDENIHDWHO15']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            harmstate['BURDENIHD_25NAAQS12'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFIHD_25NAAQS12'])
            harmstate['BURDENIHD_30NAAQS12'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFIHD_30NAAQS12'])
            harmstate['BURDENIHD_35NAAQS12'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFIHD_35NAAQS12'])    
            harmstate['BURDENIHD_40NAAQS12'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFIHD_40NAAQS12'])
            harmstate['BURDENIHD_45NAAQS12'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFIHD_45NAAQS12'])
            harmstate['BURDENIHD_50NAAQS12'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFIHD_50NAAQS12'])
            harmstate['BURDENIHD_55NAAQS12'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFIHD_55NAAQS12'])
            harmstate['BURDENIHD_60NAAQS12'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFIHD_60NAAQS12'])
            harmstate['BURDENIHD_65NAAQS12'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFIHD_65NAAQS12'])        
            harmstate['BURDENIHD_70NAAQS12'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFIHD_70NAAQS12'])
            harmstate['BURDENIHD_75NAAQS12'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFIHD_75NAAQS12']) 
            harmstate['BURDENIHD_80NAAQS12'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFIHD_80NAAQS12'])
            harmstate['BURDENIHD_85NAAQS12'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFIHD_85NAAQS12','AFIHD_90NAAQS12',
                'AFIHD_95NAAQS12']].mean(axis=1))
            harmstate['BURDENIHDNAAQS12'] = harmstate[['BURDENIHD_25NAAQS12',
                'BURDENIHD_30NAAQS12', 'BURDENIHD_35NAAQS12', 
                'BURDENIHD_40NAAQS12', 'BURDENIHD_45NAAQS12', 
                'BURDENIHD_50NAAQS12', 'BURDENIHD_55NAAQS12', 
                'BURDENIHD_60NAAQS12', 'BURDENIHD_65NAAQS12', 
                'BURDENIHD_70NAAQS12', 'BURDENIHD_75NAAQS12', 
                'BURDENIHD_80NAAQS12', 'BURDENIHD_85NAAQS12']].sum(axis=1)
            harmstate['BURDENIHDRATENAAQS12'] = (harmstate['BURDENIHDNAAQS12']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            harmstate['BURDENIHD_25WHO10'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFIHD_25WHO10'])
            harmstate['BURDENIHD_30WHO10'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFIHD_30WHO10'])
            harmstate['BURDENIHD_35WHO10'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFIHD_35WHO10'])    
            harmstate['BURDENIHD_40WHO10'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFIHD_40WHO10'])
            harmstate['BURDENIHD_45WHO10'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFIHD_45WHO10'])
            harmstate['BURDENIHD_50WHO10'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFIHD_50WHO10'])
            harmstate['BURDENIHD_55WHO10'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFIHD_55WHO10'])
            harmstate['BURDENIHD_60WHO10'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFIHD_60WHO10'])
            harmstate['BURDENIHD_65WHO10'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFIHD_65WHO10'])
            harmstate['BURDENIHD_70WHO10'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFIHD_70WHO10'])
            harmstate['BURDENIHD_75WHO10'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFIHD_75WHO10']) 
            harmstate['BURDENIHD_80WHO10'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFIHD_80WHO10'])
            harmstate['BURDENIHD_85WHO10'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFIHD_85WHO10','AFIHD_90WHO10','AFIHD_95WHO10']].mean(axis=1))
            harmstate['BURDENIHDWHO10'] = harmstate[['BURDENIHD_25WHO10',
                'BURDENIHD_30WHO10', 'BURDENIHD_35WHO10', 'BURDENIHD_40WHO10', 
                'BURDENIHD_45WHO10', 'BURDENIHD_50WHO10', 'BURDENIHD_55WHO10', 
                'BURDENIHD_60WHO10', 'BURDENIHD_65WHO10', 'BURDENIHD_70WHO10', 
                'BURDENIHD_75WHO10', 'BURDENIHD_80WHO10', 'BURDENIHD_85WHO10']
                ].sum(axis=1)
            harmstate['BURDENIHDRATEWHO10'] = (harmstate['BURDENIHDWHO10']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            harmstate['BURDENIHD_25WHO5'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFIHD_25WHO5'])
            harmstate['BURDENIHD_30WHO5'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFIHD_30WHO5'])
            harmstate['BURDENIHD_35WHO5'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFIHD_35WHO5'])    
            harmstate['BURDENIHD_40WHO5'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFIHD_40WHO5'])
            harmstate['BURDENIHD_45WHO5'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFIHD_45WHO5'])
            harmstate['BURDENIHD_50WHO5'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFIHD_50WHO5'])
            harmstate['BURDENIHD_55WHO5'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFIHD_55WHO5'])
            harmstate['BURDENIHD_60WHO5'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFIHD_60WHO5'])
            harmstate['BURDENIHD_65WHO5'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFIHD_65WHO5'])
            harmstate['BURDENIHD_70WHO5'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFIHD_70WHO5'])
            harmstate['BURDENIHD_75WHO5'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFIHD_75WHO5']) 
            harmstate['BURDENIHD_80WHO5'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFIHD_80WHO5'])
            harmstate['BURDENIHD_85WHO5'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                ihdstate.loc[ihdstate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFIHD_85WHO5','AFIHD_90WHO5','AFIHD_95WHO5']].mean(axis=1))
            harmstate['BURDENIHDWHO5'] = harmstate[['BURDENIHD_25WHO5',
                'BURDENIHD_30WHO5', 'BURDENIHD_35WHO5', 'BURDENIHD_40WHO5', 
                'BURDENIHD_45WHO5', 'BURDENIHD_50WHO5', 'BURDENIHD_55WHO5', 
                'BURDENIHD_60WHO5', 'BURDENIHD_65WHO5', 'BURDENIHD_70WHO5', 
                'BURDENIHD_75WHO5', 'BURDENIHD_80WHO5', 'BURDENIHD_85WHO5']
                ].sum(axis=1)
            harmstate['BURDENIHDRATEWHO5'] = (harmstate['BURDENIHDWHO5']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            
            # Stroke incidence for different age groups
            strokestate = strokerate.loc[
                (strokerate['location_name']==state) & 
                (strokerate['year']==year)]
            harmstate['BURDENST_25'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFST_25'])
            harmstate['BURDENST_30'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFST_30'])
            harmstate['BURDENST_35'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFST_35'])
            harmstate['BURDENST_40'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFST_40'])
            harmstate['BURDENST_45'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFST_45'])
            harmstate['BURDENST_50'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFST_50'])
            harmstate['BURDENST_55'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFST_55'])
            harmstate['BURDENST_60'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFST_60'])
            harmstate['BURDENST_65'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFST_65'])     
            harmstate['BURDENST_70'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFST_70'])
            harmstate['BURDENST_75'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFST_75']) 
            harmstate['BURDENST_80'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFST_80'])
            harmstate['BURDENST_85'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFST_85','AFST_90','AFST_95']].mean(axis=1))
            # Generate a column representing the total IHD burden 
            harmstate['BURDENST'] = harmstate[['BURDENST_25',
                'BURDENST_30', 'BURDENST_35', 'BURDENST_40', 'BURDENST_45',
                'BURDENST_50', 'BURDENST_55', 'BURDENST_60', 'BURDENST_65',
                'BURDENST_70', 'BURDENST_75', 'BURDENST_80', 'BURDENST_85']
                ].sum(axis=1)
            harmstate['BURDENSTRATE'] = (harmstate['BURDENST']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.   
            harmstate['BURDENST_25WHO15'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFST_25WHO15'])
            harmstate['BURDENST_30WHO15'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFST_30WHO15'])
            harmstate['BURDENST_35WHO15'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFST_35WHO15'])
            harmstate['BURDENST_40WHO15'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFST_40WHO15'])
            harmstate['BURDENST_45WHO15'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFST_45WHO15'])
            harmstate['BURDENST_50WHO15'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFST_50WHO15'])
            harmstate['BURDENST_55WHO15'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFST_55WHO15'])
            harmstate['BURDENST_60WHO15'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFST_60WHO15'])
            harmstate['BURDENST_65WHO15'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFST_65WHO15'])     
            harmstate['BURDENST_70WHO15'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFST_70WHO15'])
            harmstate['BURDENST_75WHO15'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFST_75WHO15']) 
            harmstate['BURDENST_80WHO15'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFST_80WHO15'])
            harmstate['BURDENST_85WHO15'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFST_85WHO15','AFST_90WHO15','AFST_95WHO15']].mean(axis=1))
            harmstate['BURDENSTWHO15'] = harmstate[['BURDENST_25WHO15',
                'BURDENST_30WHO15', 'BURDENST_35WHO15', 'BURDENST_40WHO15', 
                'BURDENST_45WHO15', 'BURDENST_50WHO15', 'BURDENST_55WHO15', 
                'BURDENST_60WHO15', 'BURDENST_65WHO15', 'BURDENST_70WHO15', 
                'BURDENST_75WHO15', 'BURDENST_80WHO15', 'BURDENST_85WHO15']
                ].sum(axis=1)
            harmstate['BURDENSTRATEWHO15'] = (harmstate['BURDENSTWHO15']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            harmstate['BURDENST_25NAAQS12'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFST_25NAAQS12'])
            harmstate['BURDENST_30NAAQS12'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFST_30NAAQS12'])
            harmstate['BURDENST_35NAAQS12'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFST_35NAAQS12'])
            harmstate['BURDENST_40NAAQS12'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFST_40NAAQS12'])
            harmstate['BURDENST_45NAAQS12'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFST_45NAAQS12'])
            harmstate['BURDENST_50NAAQS12'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFST_50NAAQS12'])
            harmstate['BURDENST_55NAAQS12'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFST_55NAAQS12'])
            harmstate['BURDENST_60NAAQS12'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFST_60NAAQS12'])
            harmstate['BURDENST_65NAAQS12'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFST_65NAAQS12'])     
            harmstate['BURDENST_70NAAQS12'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFST_70NAAQS12'])
            harmstate['BURDENST_75NAAQS12'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFST_75NAAQS12'])
            harmstate['BURDENST_80NAAQS12'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFST_80NAAQS12'])
            harmstate['BURDENST_85NAAQS12'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFST_85NAAQS12','AFST_90NAAQS12','AFST_95NAAQS12']].mean(axis=1))
            harmstate['BURDENSTNAAQS12'] = harmstate[['BURDENST_25NAAQS12',
                'BURDENST_30NAAQS12', 'BURDENST_35NAAQS12', 'BURDENST_40NAAQS12', 
                'BURDENST_45NAAQS12', 'BURDENST_50NAAQS12', 'BURDENST_55NAAQS12', 
                'BURDENST_60NAAQS12', 'BURDENST_65NAAQS12', 'BURDENST_70NAAQS12', 
                'BURDENST_75NAAQS12', 'BURDENST_80NAAQS12', 'BURDENST_85NAAQS12']
                ].sum(axis=1)
            harmstate['BURDENSTRATENAAQS12'] = (harmstate['BURDENSTNAAQS12']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            harmstate['BURDENST_25WHO10'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFST_25WHO10'])
            harmstate['BURDENST_30WHO10'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFST_30WHO10'])
            harmstate['BURDENST_35WHO10'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFST_35WHO10'])
            harmstate['BURDENST_40WHO10'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFST_40WHO10'])
            harmstate['BURDENST_45WHO10'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFST_45WHO10'])
            harmstate['BURDENST_50WHO10'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFST_50WHO10'])
            harmstate['BURDENST_55WHO10'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFST_55WHO10'])
            harmstate['BURDENST_60WHO10'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFST_60WHO10'])
            harmstate['BURDENST_65WHO10'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFST_65WHO10'])
            harmstate['BURDENST_70WHO10'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFST_70WHO10'])
            harmstate['BURDENST_75WHO10'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFST_75WHO10']) 
            harmstate['BURDENST_80WHO10'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFST_80WHO10'])
            harmstate['BURDENST_85WHO10'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFST_85WHO10','AFST_90WHO10','AFST_95WHO10']].mean(axis=1))
            harmstate['BURDENSTWHO10'] = harmstate[['BURDENST_25WHO10',
                'BURDENST_30WHO10', 'BURDENST_35WHO10', 'BURDENST_40WHO10', 
                'BURDENST_45WHO10', 'BURDENST_50WHO10', 'BURDENST_55WHO10', 
                'BURDENST_60WHO10', 'BURDENST_65WHO10', 'BURDENST_70WHO10', 
                'BURDENST_75WHO10', 'BURDENST_80WHO10', 'BURDENST_85WHO10']
                ].sum(axis=1)
            harmstate['BURDENSTRATEWHO10'] = (harmstate['BURDENSTWHO10']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.
            harmstate['BURDENST_25WHO5'] = (
                (harmstate[['pop_f_25-29','pop_m_25-29']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='25 to 29'
                ].val.values[0]/100000*harmstate['AFST_25WHO5'])
            harmstate['BURDENST_30WHO5'] = (
                (harmstate[['pop_f_30-34','pop_m_30-34']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='30 to 34'
                ].val.values[0]/100000*harmstate['AFST_30WHO5'])
            harmstate['BURDENST_35WHO5'] = (
                (harmstate[['pop_f_35-39','pop_m_35-39']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='35 to 39'
                ].val.values[0]/100000*harmstate['AFST_35WHO5'])
            harmstate['BURDENST_40WHO5'] = (
                (harmstate[['pop_f_40-44','pop_m_40-44']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='40 to 44'
                ].val.values[0]/100000*harmstate['AFST_40WHO5'])
            harmstate['BURDENST_45WHO5'] = (
                (harmstate[['pop_f_45-49','pop_m_45-49']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='45 to 49'
                ].val.values[0]/100000*harmstate['AFST_45WHO5'])
            harmstate['BURDENST_50WHO5'] = (
                (harmstate[['pop_f_50-54','pop_m_50-54']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='50 to 54'
                ].val.values[0]/100000*harmstate['AFST_50WHO5'])
            harmstate['BURDENST_55WHO5'] = (
                (harmstate[['pop_m_55-59','pop_f_55-59']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='55 to 59'
                ].val.values[0]/100000*harmstate['AFST_55WHO5'])
            harmstate['BURDENST_60WHO5'] = (
                (harmstate[['pop_f_60-61','pop_m_60-61','pop_f_62-64',
                'pop_m_62-64']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='60 to 64'
                ].val.values[0]/100000*harmstate['AFST_60WHO5'])
            harmstate['BURDENST_65WHO5'] = (
                (harmstate[['pop_f_65-66','pop_f_67-69','pop_m_65-66',
                'pop_m_67-69']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='65 to 69'
                ].val.values[0]/100000*harmstate['AFST_65WHO5'])
            harmstate['BURDENST_70WHO5'] = (
                (harmstate[['pop_m_70-74','pop_f_70-74']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='70 to 74'
                ].val.values[0]/100000*harmstate['AFST_70WHO5'])
            harmstate['BURDENST_75WHO5'] = (
                (harmstate[['pop_m_75-79','pop_f_75-79']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='75 to 79'
                ].val.values[0]/100000*harmstate['AFST_75WHO5']) 
            harmstate['BURDENST_80WHO5'] = (
                (harmstate[['pop_m_80-84','pop_f_80-84']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='80 to 84'
                ].val.values[0]/100000*harmstate['AFST_80WHO5'])
            harmstate['BURDENST_85WHO5'] = (
                (harmstate[['pop_m_gt85','pop_f_gt85']].sum(axis=1))*
                strokestate.loc[strokestate['age_name']=='85 plus'].val.values[0]/100000*
                harmstate[['AFST_85WHO5','AFST_90WHO5','AFST_95WHO5']].mean(axis=1))
            harmstate['BURDENSTWHO5'] = harmstate[['BURDENST_25WHO5',
                'BURDENST_30WHO5', 'BURDENST_35WHO5', 'BURDENST_40WHO5', 
                'BURDENST_45WHO5', 'BURDENST_50WHO5', 'BURDENST_55WHO5', 
                'BURDENST_60WHO5', 'BURDENST_65WHO5', 'BURDENST_70WHO5', 
                'BURDENST_75WHO5', 'BURDENST_80WHO5', 'BURDENST_85WHO5']
                ].sum(axis=1)
            harmstate['BURDENSTRATEWHO5'] = (harmstate['BURDENSTWHO5']/harmstate[[
                'pop_f_25-29','pop_m_25-29','pop_f_30-34','pop_m_30-34', 
                'pop_f_35-39','pop_m_35-39','pop_f_40-44','pop_m_40-44',
                'pop_f_45-49','pop_m_45-49','pop_f_50-54','pop_m_50-54',
                'pop_f_55-59','pop_m_55-59','pop_f_60-61','pop_m_60-61',
                'pop_f_62-64','pop_m_62-64','pop_f_65-66','pop_m_65-66',
                'pop_f_67-69','pop_m_67-69','pop_f_70-74','pop_m_70-74',
                'pop_f_75-79','pop_m_75-79','pop_f_80-84','pop_m_80-84',
                'pop_f_gt85','pop_m_gt85']].sum(axis=1))*100000.            

            # COPD incidence
            copdstate = copdrate.loc[(copdrate['location_name']==state) &
                (copdrate['year']==year)]
            harmstate['BURDENCOPD'] = (harmstate['pop_tot']*
               copdstate.val.values[0]/100000*harmstate['AFCOPD'])
            harmstate['BURDENCOPDRATE'] = (harmstate['BURDENCOPD']/harmstate[
                'pop_tot'])*100000. 
            harmstate['BURDENCOPDWHO15'] = (harmstate['pop_tot']*
               copdstate.val.values[0]/100000*harmstate['AFCOPDWHO15'])
            harmstate['BURDENCOPDRATEWHO15'] = (harmstate['BURDENCOPDWHO15'
                ]/harmstate['pop_tot'])*100000.
            harmstate['BURDENCOPDNAAQS12'] = (harmstate['pop_tot']*
               copdstate.val.values[0]/100000*harmstate['AFCOPDNAAQS12'])
            harmstate['BURDENCOPDRATENAAQS12'] = (harmstate['BURDENCOPDNAAQS12'
                ]/harmstate['pop_tot'])*100000.    
            harmstate['BURDENCOPDWHO10'] = (harmstate['pop_tot']*
               copdstate.val.values[0]/100000*harmstate['AFCOPDWHO10'])
            harmstate['BURDENCOPDRATEWHO10'] = (harmstate['BURDENCOPDWHO10'
                ]/harmstate['pop_tot'])*100000.            
            harmstate['BURDENCOPDWHO5'] = (harmstate['pop_tot']*
               copdstate.val.values[0]/100000*harmstate['AFCOPDWHO5'])
            harmstate['BURDENCOPDRATEWHO5'] = (harmstate['BURDENCOPDWHO5'
                ]/harmstate['pop_tot'])*100000.            
            
            # Lung cancer incidence
            lcstate = lcrate.loc[(lcrate['location_name']==state) &
                (lcrate['year']==year)]
            harmstate['BURDENLC'] = (harmstate['pop_tot']*
               lcstate.val.values[0]/100000*harmstate['AFLC'])
            harmstate['BURDENLCRATE'] = (harmstate['BURDENLC']/harmstate[
                'pop_tot'])*100000.              
            harmstate['BURDENLCWHO15'] = (harmstate['pop_tot']*
               lcstate.val.values[0]/100000*harmstate['AFLCWHO15'])
            harmstate['BURDENLCRATEWHO15'] = (harmstate['BURDENLCWHO15'
                ]/harmstate['pop_tot'])*100000.              
            harmstate['BURDENLCNAAQS12'] = (harmstate['pop_tot']*
               lcstate.val.values[0]/100000*harmstate['AFLCNAAQS12'])
            harmstate['BURDENLCRATENAAQS12'] = (harmstate['BURDENLCNAAQS12'
                ]/harmstate['pop_tot'])*100000.              
            harmstate['BURDENLCWHO10'] = (harmstate['pop_tot']*
               lcstate.val.values[0]/100000*harmstate['AFLCWHO10'])
            harmstate['BURDENLCRATEWHO10'] = (harmstate['BURDENLCWHO10'
                ]/harmstate['pop_tot'])*100000.              
            harmstate['BURDENLCWHO5'] = (harmstate['pop_tot']*
               lcstate.val.values[0]/100000*harmstate['AFLCWHO5'])
            harmstate['BURDENLCRATEWHO5'] = (harmstate['BURDENLCWHO5'
                ]/harmstate['pop_tot'])*100000.              

            # Type 2 diabetes mellitus
            t2dmstate = t2dmrate.loc[(t2dmrate['location_name']==state) &
                (t2dmrate['year']==year)]
            harmstate['BURDENDM'] = (harmstate['pop_tot']*
               t2dmstate.val.values[0]/100000*harmstate['AFDM'])
            harmstate['BURDENDMRATE'] = (harmstate['BURDENDM']/harmstate[
                'pop_tot'])*100000.                          
            harmstate['BURDENDMWHO15'] = (harmstate['pop_tot']*
               t2dmstate.val.values[0]/100000*harmstate['AFDMWHO15'])
            harmstate['BURDENDMRATEWHO15'] = (harmstate['BURDENDMWHO15'
                ]/harmstate['pop_tot'])*100000.
            harmstate['BURDENDMNAAQS12'] = (harmstate['pop_tot']*
               t2dmstate.val.values[0]/100000*harmstate['AFDMNAAQS12'])
            harmstate['BURDENDMRATENAAQS12'] = (harmstate['BURDENDMNAAQS12'
                ]/harmstate['pop_tot'])*100000.
            harmstate['BURDENDMWHO10'] = (harmstate['pop_tot']*
               t2dmstate.val.values[0]/100000*harmstate['AFDMWHO10'])
            harmstate['BURDENDMRATEWHO10'] = (harmstate['BURDENDMWHO10'
                ]/harmstate['pop_tot'])*100000.
            harmstate['BURDENDMWHO5'] = (harmstate['pop_tot']*
               t2dmstate.val.values[0]/100000*harmstate['AFDMWHO5'])
            harmstate['BURDENDMRATEWHO5'] = (harmstate['BURDENDMWHO5'
                ]/harmstate['pop_tot'])*100000.            

            # Lower respiratory infection
            lristate = lrirate.loc[(lrirate['location_name']==state) &
                (lrirate['year']==year)]
            harmstate['BURDENLRI'] = (harmstate['pop_tot']*
               lristate.val.values[0]/100000*harmstate['AFLRI'])
            harmstate['BURDENLRIRATE'] = (harmstate['BURDENLRI']/harmstate[
                'pop_tot'])*100000.    
            harmstate['BURDENLRIWHO15'] = (harmstate['pop_tot']*
               lristate.val.values[0]/100000*harmstate['AFLRIWHO15'])
            harmstate['BURDENLRIRATEWHO15'] = (harmstate['BURDENLRIWHO15'
                ]/harmstate['pop_tot'])*100000.
            harmstate['BURDENLRINAAQS12'] = (harmstate['pop_tot']*
               lristate.val.values[0]/100000*harmstate['AFLRINAAQS12'])
            harmstate['BURDENLRIRATENAAQS12'] = (harmstate['BURDENLRINAAQS12'
                ]/harmstate['pop_tot'])*100000.
            harmstate['BURDENLRIWHO10'] = (harmstate['pop_tot']*
               lristate.val.values[0]/100000*harmstate['AFLRIWHO10'])
            harmstate['BURDENLRIRATEWHO10'] = (harmstate['BURDENLRIWHO10'
                ]/harmstate['pop_tot'])*100000.
            harmstate['BURDENLRIWHO5'] = (harmstate['pop_tot']*
               lristate.val.values[0]/100000*harmstate['AFLRIWHO5'])
            harmstate['BURDENLRIRATEWHO5'] = (harmstate['BURDENLRIWHO5'
                ]/harmstate['pop_tot'])*100000.            
                          
            # Calculate total PM2.5-attributable mortality 
            harmstate['BURDENPMALL'] = harmstate[['BURDENLRI', 'BURDENDM',
                'BURDENLC', 'BURDENCOPD', 'BURDENIHD', 'BURDENST']].sum(axis=1)
            harmstate['BURDENPMALLRATE'] = (harmstate['BURDENPMALL']/
                harmstate['pop_tot'])*100000.
            harmstate['BURDENPMALLWHO15'] = harmstate[['BURDENLRIWHO15', 
                'BURDENDMWHO15', 'BURDENLCWHO15', 'BURDENCOPDWHO15', 
                'BURDENIHDWHO15', 'BURDENSTWHO15']].sum(axis=1)
            harmstate['BURDENPMALLRATEWHO15'] = (harmstate['BURDENPMALLWHO15'
                ]/harmstate['pop_tot'])*100000.    
            harmstate['BURDENPMALLNAAQS12'] = harmstate[['BURDENLRINAAQS12', 
                'BURDENDMNAAQS12', 'BURDENLCNAAQS12', 'BURDENCOPDNAAQS12', 
                'BURDENIHDNAAQS12', 'BURDENSTNAAQS12']].sum(axis=1)
            harmstate['BURDENPMALLRATENAAQS12'] = (harmstate[
                'BURDENPMALLNAAQS12']/harmstate['pop_tot'])*100000.    
            harmstate['BURDENPMALLWHO10'] = harmstate[['BURDENLRIWHO10', 
                'BURDENDMWHO10', 'BURDENLCWHO10', 'BURDENCOPDWHO10', 
                'BURDENIHDWHO10', 'BURDENSTWHO10']].sum(axis=1)
            harmstate['BURDENPMALLRATEWHO10'] = (harmstate['BURDENPMALLWHO10'
                ]/harmstate['pop_tot'])*100000.    
            harmstate['BURDENPMALLWHO5'] = harmstate[['BURDENLRIWHO5', 
                'BURDENDMWHO5', 'BURDENLCWHO5', 'BURDENCOPDWHO5', 
                'BURDENIHDWHO5', 'BURDENSTWHO5']].sum(axis=1)
            harmstate['BURDENPMALLRATEWHO5'] = (harmstate['BURDENPMALLWHO5'
                ]/harmstate['pop_tot'])*100000.                

            # Append temporary DataFrame to list        
            harmout.append(harmstate)
            del harmstate
    harmout = pd.concat(harmout)
    return harmout