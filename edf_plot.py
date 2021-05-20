#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:02:42 2021

@author: ghkerr
"""
# Local environment
DIR = '/Users/ghkerr/GW/edf/'
DIR_NO2 = DIR+'data/no2/'
DIR_TROPOMI = '/Users/ghkerr/GW/data/'
DIR_HARM = '/Users/ghkerr/GW/data/census_no2_harmonzied/'
DIR_FIG = DIR+'figs/'
DIR_OUT = DIR
DIR_GEO = '/Users/ghkerr/GW/data/geography/'
DIR_TYPEFACE = '/Users/ghkerr/Library/Fonts/'
# Load custom font
import sys
if 'mpl' not in sys.modules:
    import matplotlib.font_manager
    prop = matplotlib.font_manager.FontProperties(
            fname=DIR_TYPEFACE+'cmunbmr.ttf')
    matplotlib.rcParams['font.family'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname=DIR_TYPEFACE+'cmunbbx.ttf')
    matplotlib.rcParams['mathtext.bf'] = prop.get_name()
    prop = matplotlib.font_manager.FontProperties(
        fname=DIR_TYPEFACE+'cmunbmr.ttf')
    matplotlib.rcParams['mathtext.it'] = prop.get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False

# # Pegasus
# DIR = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'
# DIR_OUT = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'

def gbd_incidence(year=2019): 
    """Fetch age-group sepcific asthma incidence rates from the Institute 
    for Health Metrics and Evaluation for the USA. Results are taken from 
    various GBD reports (year argument) for the year 2015; note that rates
    for a given year can change with updates to the GBD. 
    
    Rates for the 2019 GBD can be found at http://ghdx.healthdata.org/gbd-results-tool
    Rates for the 2017 GBD can be found at https://gbd2017.healthdata.org/gbd-search/
    Base -> Single
    Location -> United States of America
    Year -> 2015
    Context -> 
    Age -> 1 to 4, 5 to 9, 10 to 14, 15 to 19
    Metric -> Rate
    Measure -> Incidence
    Sex -> Both
    Cause -> B.3.3 Asthma
    Note that these values should mirror values from choropleth maps at 
    https://vizhub.healthdata.org/gbd-compare/
    
    Parameters
    ----------
    year : int
        Year of GBD report

    Returns
    -------
    gbd : pandas.core.frame.DataFrame
        2015 age-group-specific asthma incidence rates per 100,000. 
    """
    import pandas as pd
    if year==2019:
        gbd = pd.DataFrame([
            ['1-4', '2015', 6534.79, 8931.04, 4723.77],
            ['5-9', '2015', 3051.12, 4582.46, 1696.35],
            ['10-14', '2015', 1780.07, 2636.55, 854.20],
            ['15-19', '2015', 1184.64, 1677.33, 752.49]], 
            columns=['Age', 'Year', 'Value', 'Upper', 'Lower'])
        gbd.set_index('Age', inplace=True)
    
    # Older GBD 
    if year==2017:
        gbd = pd.DataFrame([
                ['1-4', '2015', 3657.58, 4613.36, 2768.08],
                ['5-9', '2015', 944.50, 1451.03, 599.97],
                ['10-14', '2015', 684.12, 966.37, 452.09],
                ['15-19', '2015', 620.67, 793.04, 461.91]], 
                columns=['Age', 'Year', 'Value', 'Upper', 'Lower'])
        gbd.set_index('Age', inplace=True)
    return gbd

def geo_idx(dd, dd_array):
    """Function searches for nearest decimal degree in an array of decimal 
    degrees and returns the index. np.argmin returns the indices of minimum 
    value along an axis. So subtract dd from all values in dd_array, take 
    absolute value and find index of minimum. 
    
    Parameters
    ----------
    dd : int
        Latitude or longitude whose index in dd_array is being sought
    dd_array : numpy.ndarray 
        1D array of latitude or longitude 
    
    Returns
    -------
    geo_idx : int
        Index of latitude or longitude in dd_array that is closest in value to 
        dd
    """
    import numpy as np   
    from scipy import stats
    geo_idx = (np.abs(dd_array - dd)).argmin()
    # if distance from closest cell to intended value is 2x the value of the
    # spatial resolution, raise error 
    res = np.abs(stats.mode(np.diff(dd_array))[0][0])
    if np.abs(dd_array[geo_idx] - dd) > (2*res):
        print('Closet index far from intended value!')
        return 
    return geo_idx

def scale_larkin(lng_larkin, lat_larkin, no2_larkin, lng_mohegh, lat_mohegh, 
    no2_mohegh_2010_2012, no2_mohegh_2019):
    """Generate 100 m NO2 dataset for 2019 (or any recent year, depending on 
    which year's NO2 dataset is passed in as "no2_mohegh_2019"). This dataset
    is calculated by multiplying a given Larkin et al. (2017) grid cell NO2 
    concentration by the ratio of 2019:(2010-2012) NO2 at the nearest gridcell 
    from the Mohegh et al. (2021) dataset. 

    Parameters
    ----------
    lng_larkin : numpy.ndarray
        Longitude for Larkin NO2 dataset, units of degrees east, [lng_lark,]
    lat_larkin : numpy.ndarray
        Latitude for Larkin NO2 dataset, units of degrees north, [lat_lark,]
    no2_larkin : numpy.ndarray
        Larkin NO2 dataset, representing mean 2010-2012 concentrations, 
        units of ppbv, [lat_lark, lng_lark]
    lng_mohegh : numpy.ndarray
        Latitude for Mohegh NO2 dataset, units of degrees north, [lng_moh,]
    lat_mohegh : numpy.ndarray
        Latitude for Mohegh NO2 dataset, units of degrees north, [lat_moh,]
    mohegh_closest_2010_2012 : numpy.ndarray
        Mohegh NO2 dataset averaged over 2010-2012, units of ppbv, [lat_moh, 
        lng_moh]
    mohegh_closest_2019 : numpy.ndarray
        Mohegh NO2 dataset for 2019, units of ppbv, [lat_moh, lng_moh]
        
    Returns
    -------
    no2_larkin_scaled : numpy.ndarray
        Larkin NO2 dataset scaled to 2019 concentrations using ratio of 
        2010-2012:2019 Mohegh NO2, units of ppbv, [lat_lark, lng_lark]
        
    References
    ----------
    A. Larkin, J. A. Geddes, R. V. Martin, Q. Xiao, Y. Liu, J. D. Marshall,
        M. Brauer, P. Hystad, Global Land Use Regression Model for Nitrogen 
        Dioxide Air Pollution. Environ. Sci. Technol. 51, 6957–6964 (2017).
    A. Mohegh, D. L. Goldberg, S. C. Anenberg. Methods for estimating annual
        average NO2 concentrations globally from 1990-2019 from the Global 
        Burden of Disease Study. in prep. 
    """
    import time
    import numpy as np
    no2_larkin_scaled = np.empty(shape=no2_larkin.shape)
    no2_larkin_scaled[:] = np.nan
    start = time.time()
    for i, lat in enumerate(lat_larkin):
        print('For %d in %d...'%(i, len(lat_larkin)))
        for j, lng in enumerate(lng_larkin): 
            no2_larkin_ij = no2_larkin[i,j]
            lat_larkin_ij = lat_larkin[i]
            lng_larkin_ij = lng_larkin[j]
            # Find grid cells in Mohegh et al. (2021) dataset containing 
            # Larkin grid cell
            lat_closest = geo_idx(lat_larkin_ij, lat_mohegh)
            lng_closest = geo_idx(lng_larkin_ij, lng_mohegh)
            # Select Mohegh et al. (2021) NO2 concentrations in 2010-2012 
            # and 2019 and form ration 
            mohegh_closest_2010_2012 = no2_mohegh_2010_2012[lat_closest, 
                lng_closest]
            mohegh_closest_2019 = no2_mohegh_2019[lat_closest, lng_closest]
            ratio_ij = mohegh_closest_2019/mohegh_closest_2010_2012
            # Scale 2010-2012 Larkin NO2 by 2019:2010-2012 ratio from 
            # Mohegh et al. (2021)
            no2_larkin_scaled[i,j] = no2_larkin_ij*ratio_ij
        end = time.time()
        print(end - start)            
    return no2_larkin_scaled

def calculate_incidence(no2, rr, gbd, cfconc, pop1_4, pop5_9, pop10_14, 
    pop15_18, verbose=False):
    """This function is the main workhorse of this script and calculates 
   NO2-attributable paediatric asthma incidence per 100k children and the 
   percentage of total incidence attributable to NO2 over a given area 
   assuming a particular counterfactual concentration and relative risk (RR). 

    Parameters
    ----------
    no2 : numpy.ndarray
        Gridded NO2 concentrations, units of ppbv, [lat, lng]
    rr : float
        Relative risk; note that values should come from Khreis et al. (2017)
    gbd : pandas.core.frame.DataFrame
        Age-group-specific asthma incidence rates per 100000
    cfconc : float
        DESCRIPTION.
    pop1_4 : numpy.ndarray
        Gridded population for 1-4 year old age group. Total population is 
        derived from GPWv4, and the paediatric population in the 1-4 year old 
        age group is estimated using Basic Demographic Characteristics (v4.10)
        for 2010, [lat, lng]
    pop5_9 : numpy.ndarray
        Gridded population for 5-9 year old age group, [lat, lng]
    pop10_14 : numpy.ndarray
        Gridded population for 10-14 year old age group, [lat, lng]
    pop15_18 : numpy.ndarray
        Gridded population for 15-18 year old age group, [lat, lng]
        
    Returns
    -------
    burden : float
        NO2-attributable asthma burden 
    cp100 : float
         NO2-attributable paediatric asthma incidence per 100k children
    percent1_4 : float
        Percentage of total incidence attributable to NO2 for 1-4 year old age
        group
    percent5_9 : float
        Percentage of total incidence attributable to NO2 for 5-9 year old age
        group    
    percent10_14:: float
        Percentage of total incidence attributable to NO2 for 10-14 year old 
        age group    
    percent15_18 : float
        Percentage of total incidence attributable to NO2 for 15-18 year old 
        age group    
    percent : float
        Percentage of total incidence attributable to NO2
    
    References
    ----------
    H. Khreis, et al., Exposure to traffic-related air pollution and risk of 
        development of childhood asthma: A systematic review and meta-
        analysis. Environment International 100, 1–31 (2017).         
    """
    import numpy as np
    # Calculate attributable fraction 
    af = calculate_af(no2, rr)
    # Counterfactual; a counterfactual of 2ppb is reasonable given Ploy and 
    # Arash's work. To implement the counterfactual, we do two difference
    # asthma burden calculations. 
    # 1) Using the estimated concentrations (i.e., from Larkin or Mohegh)
    # 2) Using an NO2 array where every grid cell is equal to 2ppb. 
    # Then, by subtracting the two, we are subtracting out the risk 
    # below the counterfactual.
    cf = np.empty(shape=af.shape)
    cf[:] = cfconc 
    cf = calculate_af(cf, rr)
    # Asthma burden for estimated NO2 concentrations; note that the health
    # impact function is given in Achakulwisut et al. (2019). 
    burden1_4 = (gbd.loc['1-4']['Value']/100000.)*np.nansum(pop1_4*af)
    burden5_9 = (gbd.loc['5-9']['Value']/100000.)*np.nansum(pop5_9*af)
    burden10_14 = (gbd.loc['10-14']['Value']/100000.)*np.nansum(pop10_14*af)
    burden15_18 = (gbd.loc['15-19']['Value']/100000.)*np.nansum(pop15_18*af)
    # For the counterfactual NO2 concentrations
    cfburden1_4 = (gbd.loc['1-4']['Value']/100000.)*np.nansum(pop1_4*cf)
    cfburden5_9 = (gbd.loc['5-9']['Value']/100000.)*np.nansum(pop5_9*cf)
    cfburden10_14 = (gbd.loc['10-14']['Value']/100000.)*np.nansum(pop10_14*cf)
    cfburden15_18 = (gbd.loc['15-19']['Value']/100000.)*np.nansum(pop15_18*cf)
    # Subtract out counterfactual 
    burden1_4 = burden1_4-cfburden1_4
    burden5_9 = burden5_9-cfburden5_9
    burden10_14 = burden10_14-cfburden10_14
    burden15_18 = burden15_18-cfburden15_18
    # Total burden and paediatric population 
    burden = burden1_4+burden5_9+burden10_14+burden15_18
    paediatricpop = (np.nansum(pop1_4)+np.nansum(pop5_9)+np.nansum(pop10_14)+
        np.nansum(pop15_18))
    # Calculate cases per year (or per time period, if using an average of 
    # several years) per 100k
    cp100 = (burden/paediatricpop)*100000.
    # Calculate NO2-attributable percentage
    percent1_4 = (burden1_4/((gbd.loc['1-4']['Value']/100000.)*
        np.nansum(pop1_4)))*100
    percent5_9 = (burden5_9/((gbd.loc['5-9']['Value']/100000.)*
        np.nansum(pop5_9)))*100
    percent10_14 = (burden10_14/((gbd.loc['10-14']['Value']/100000.)*
        np.nansum(pop10_14)))*100
    percent15_18 = (burden15_18/((gbd.loc['15-19']['Value']/100000.)*
        np.nansum(pop15_18)))*100
    # Calculate population weighted percent (5 April 2020: note that this 
    # method is preferred to a simple average. Normally, we calculate NO2-
    # attributable asthma cases (then can sum the attributable cases across
    # different age groups and divide by total asthma cases in those age 
    # groups). Since we can't do that for attributable percentages, we are
    # consistent and calculate population weighted percentages)
    paediatricpop = (np.nansum(pop1_4)+np.nansum(pop5_9)+np.nansum(pop10_14)+
        np.nansum(pop15_18))
    percent = (((np.nansum(pop1_4)/paediatricpop)*percent1_4)+
        ((np.nansum(pop5_9)/paediatricpop)*percent5_9)+
        ((np.nansum(pop10_14)/paediatricpop)*percent10_14)+
        ((np.nansum(pop15_18)/paediatricpop)*percent10_14))        
    percent_old = np.nanmean([percent1_4, percent5_9, percent10_14, 
        percent10_14])
    print(percent_old)
    if verbose is not False: 
        print('Cases per year per 100k...','%.3f'%cp100)
        print('Simple average NO2-attributable percentage....','%.1f%%'%(
            percent))
        print('NO2-attributable percentage for 1-4 yr olds...','%.1f%%'%(
            percent1_4))
        print('NO2-attributable percentage for 5-9 yr olds...','%.1f%%'%(
            percent5_9))
        print('NO2-attributable percentage for 10-14 yr olds...','%.1f%%'%(
            percent10_14))
        print('NO2-attributable percentage for 15-18 yr olds...','%.1f%%'%(
            percent15_18))
    return (burden, cp100, percent1_4, percent5_9, percent10_14, percent15_18,
        percent)

def calculate_incidence_UI(no2, pop1_4, pop5_9, pop10_14, pop15_18, fstr):
    """The meta-analysis of Khreis et al. (2017) reported a central mean 
    relative risk (RR) of 1.26 per 10 ppb NO2 95% uncertainty interval (UI)
    1.10–1.37). This function uses this mean RR and UI to calculate the 
    total number of cases, total paediatric asthma incidence per 100k cases, 
    and NO2-attributable percentage reflecting uncertainty in the RR. 

    Parameters
    ----------
    no2 : numpy.ndarray
        Gridded NO2 concentrations, units of ppbv, [lat, lng]
    pop1_4 : numpy.ndarray
        Gridded population for 1-4 year old age group. Total population is 
        derived from GPWv4, and the paediatric population in the 1-4 year old 
        age group is estimated using Basic Demographic Characteristics (v4.10)
        for 2010, [lat, lng]
    pop5_9 : numpy.ndarray
        Gridded population for 5-9 year old age group, [lat, lng]
    pop10_14 : numpy.ndarray
        Gridded population for 10-14 year old age group, [lat, lng]
    pop15_18 : numpy.ndarray
        Gridded population for 15-18 year old age group, [lat, lng]
    fstr : str
        String for output file (indicating region, resolution, GBD version)

    Returns
    -------
    None

    References
    ----------
    H. Khreis, et al., Exposure to traffic-related air pollution and risk of 
        development of childhood asthma: A systematic review and meta-
        analysis. Environment International 100, 1–31 (2017).       
    """
    import pandas as pd
    # GBD 
    gbd = gbd_incidence()    
    # Counterfactual concentration
    cfconc = 2.
    # Relative risk and UI from Khreis et al. (2017)
    rrlow = 1.10
    rr = 1.26
    rrhigh = 1.37
    # Lower cutoff of UI for RR
    (burdenlow, cp100low, percent1_4low, percent5_9low, percent10_14low,
        percent15_18low, percentlow) = calculate_incidence(no2, rrlow, gbd, 
        cfconc, pop1_4, pop5_9, pop10_14, pop15_18)
    # Mean RR
    (burden, cp100, percent1_4, percent5_9, percent10_14, percent15_18,
        percent) = calculate_incidence(no2, rr, gbd, cfconc, pop1_4, pop5_9, 
        pop10_14, pop15_18)
    # Upper cutoff of UI for RR
    (burdenhigh, cp100high, percent1_4high, percent5_9high, percent10_14high,
        percent15_18high, percenthigh) = calculate_incidence(no2, rrhigh, gbd, 
        cfconc, pop1_4, pop5_9, pop10_14, pop15_18)
    print('NO2-attributable incidence assuming counterfactual '+\
        'concentration of %d ppb: '%(cfconc)+\
        '%d (%d - %d)'%(burden, burdenlow, burdenhigh))
    print('Percentage of NO2-attributable incidence: %.2f (%.2f - %.2f)'%(
        percent, percentlow, percenthigh))
    print('NO2-attributable paediatric asthma incidence per 100k children: '+\
        '%.2f (%.2f - %.2f)'%(cp100, cp100low, cp100high))
    # Save output to CSV file  
    output = pd.DataFrame([
        ['Incidence assuming counterfactual of %d ppb'%cfconc, 
         burden, burdenlow, burdenhigh],
        ['Percentage of NO2-attributable incidence', 
         percent, percentlow, percenthigh],
        ['NO2-attributable paediatric asthma incidence per 100k children', 
         cp100, cp100low, cp100high]],
        columns=['Metric', 'RR=%.2f'%rr, 'RR=%.2f'%rrlow, 'RR=%.2f'%rrhigh])
    output.set_index('Metric', inplace=True)    
    output.to_csv(DIR_OUT+'%s.csv'%fstr)    
    return

import numpy as np
import pandas as pd
import sys
sys.path.append(DIR)
import edf_open




# # # # # Load data
# Bay Area
bayarea = [-122.643473,-122.034418,37.669612,37.933289]
# Oakland Area 
oakland = [-122.362192,-122.232588,37.788004,37.848965]
# Contiguous US
conus = [-124.7844079, -66.9513812, 24.7433195, 49.3457868]
# # Mohegh 1 km NO2 for CONUS, 2010-2012
# lng_mohegh, lat_mohegh, no2_mohegh_2010_2012 = edf_open.open_no2pop_tif(
#     ['2010_final_1km_usa', '2011_final_1km_usa', '2012_final_1km_usa'], 
#     -999., 'NO2', conus)
# #  Larkin 100m NO2 (2010-2012)
# lng_larkin, lat_larkin, no2_larkin = edf_open.open_no2pop_tif('lur_average2', 
#     128., 'NO2', clip)
# # Mohegh 1 km NO2 for CONUS, 2019
# lng_mohegh, lat_mohegh, no2_mohegh_2019 = edf_open.open_no2pop_tif(
#     '2019_final_1km_usa', -999., 'NO2', conus)
# # Cooper ~2.8km NO2 for CONUS, 2019
# lng_cooper, lat_cooper, no2_cooper_2019 = edf_open.open_cooperno2(conus)
# # AQS NO2 observations for 2015 and 2019
# aqs_amean_2019 = edf_open.read_aqs_amean(2019)
# aqs_amean_2015 = edf_open.read_aqs_amean(2015)
# aqs_hourly_2019 = edf_open.read_aqs_hourly(2019)
# # Find urban/rural state and county FIPS codes
# rural_lookup = pd.read_csv(DIR_HARM+'County_Rural_Lookup.csv', 
#     delimiter=',', header=0, engine='python')
# # Designate counties that have a rural population < 5% as urbans
# urbancounties = rural_lookup[rural_lookup
#     ['2010 Census \nPercent Rural'] < 5.]
# # Transform 2015 GEOID column so the length matches the GEOIDs in 
# # harmonized data
# urbancounties = [(str(x).zfill(5)) for x in urbancounties['2015 GEOID']]
# # Add columns to AQS NO2 observations corresponding to co-located grid 
# # cells from Mohegh and Cooper datasets
# aqs_amean_2019['Mohegh NO2'] = np.nan
# aqs_amean_2019['Rural-Urban'] = np.nan
# for index, row in aqs_amean_2019.iterrows():
#     lat_aqs = row['Latitude']
#     lng_aqs = row['Longitude']
#     # Colocated grid cell
#     lat_idx = geo_idx(lat_aqs, lat_mohegh)
#     lng_idx = geo_idx(lng_aqs, lng_mohegh)
#     no2_ataqs = no2_mohegh_2019[lat_idx, lng_idx]
#     # Add Mohegh NO2 value to DataFrame
#     aqs_amean_2019.loc[index, 'Mohegh NO2'] = no2_ataqs
#     # Look up rural population level of corresponding county
#     clu = str(row['State Code']).zfill(2)+str(row['County Code']).zfill(3)
#     if clu in urbancounties:
#         aqs_amean_2019.loc[index, 'Rural-Urban'] = 'Urban'
#     else:
#         aqs_amean_2019.loc[index, 'Rural-Urban'] = 'Rural'
# aqs_hourly_2019['Cooper NO2'] = np.nan
# aqs_hourly_2019['Rural-Urban'] = np.nan
# for index, row in aqs_hourly_2019.iterrows():
#     lat_aqs = row['Latitude']
#     lng_aqs = row['Longitude']
#     lat_idx = geo_idx(lat_aqs, lat_cooper)
#     lng_idx = geo_idx(lng_aqs, lng_cooper)
#     no2_ataqs = no2_cooper_2019[lat_idx, lng_idx]
#     aqs_hourly_2019.loc[index, 'Cooper NO2'] = no2_ataqs    
#     clu = (str(int(row['State Code'])).zfill(2)+
#         str(int(row['County Code'])).zfill(3))
#     if clu in urbancounties:
#         aqs_hourly_2019.loc[index, 'Rural-Urban'] = 'Urban'
#     else:
#         aqs_hourly_2019.loc[index, 'Rural-Urban'] = 'Rural'    
# from scipy.spatial.distance import cdist
# def closest_point(point, points):
#     """Find closest point from a list of points. Adapted from https://
#     stackoverflow.com/questions/38965720/find-closest-point-in-pandas-dataframes
#     """
#     return points[cdist([point], points).argmin()]
# no2_cases = pd.read_csv('/Users/ghkerr/Downloads/cases_no2_2015_tract.csv',
#     sep=',', engine='python')
# # Find closest lat/lon coordinates of CACES NO2 at census tracts to
# # each AQS monitor
# no2_cases['point'] = [(x, y) for x,y in zip(no2_cases['lat'], 
#     no2_cases['lon'])]
# aqs_amean_2015['point'] = [(x, y) for x,y in zip(aqs_amean_2015['Latitude'], 
#     aqs_amean_2015['Longitude'])]
# aqs_amean_2015['closest'] = [closest_point(x, list(no2_cases['point'])
#     ) for x in aqs_amean_2015['point']]
# aqs_amean_2015['Cases NO2'] = np.nan
# aqs_amean_2015['Rural-Urban'] = np.nan
# for index, row in aqs_amean_2015.iterrows():
#     cases_coord = row['closest']
#     no2_cases_point = no2_cases.loc[no2_cases['point']==cases_coord]
#     aqs_amean_2015.loc[index, 'Cases NO2'] = no2_cases_point['pred_wght'].values
#     clu = (str(int(row['State Code'])).zfill(2)+
#         str(int(row['County Code'])).zfill(3))
#     if clu in urbancounties:
#         aqs_amean_2015.loc[index, 'Rural-Urban'] = 'Urban'
#     else:
#         aqs_amean_2015.loc[index, 'Rural-Urban'] = 'Rural'    

"""MAPS OF ANENBERG, COOPER NO2 AND BIAS AGAINST AQS"""
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# # Colormaps and levels
# cmap = plt.cm.get_cmap('magma_r')
# levels = np.arange(0,20+2,2)
# cmapb = plt.cm.get_cmap('bwr')
# levelsb = np.arange(-10,10+2,2)
# proj = ccrs.PlateCarree(central_longitude=0.0)
# # Plotting
# fig = plt.figure(figsize=(10,4))
# ax1 = plt.subplot2grid((2,3),(0,0), projection=proj)
# ax2 = plt.subplot2grid((2,3),(0,1), projection=proj)
# ax3 = plt.subplot2grid((2,3),(0,2), projection=proj)
# ax4 = plt.subplot2grid((2,3),(1,0), projection=proj)
# ax5 = plt.subplot2grid((2,3),(1,1), projection=proj)
# ax6 = plt.subplot2grid((2,3),(1,2), projection=proj)
# skip = 10
# # Anenberg & Mohegh et al. (2021) dataset
# mb = ax1.scatter(aqs_amean['Longitude'], aqs_amean['Latitude'], 
#     c=aqs_amean['Arithmetic Mean'],s=2, cmap=cmap, 
#     transform=ccrs.PlateCarree(), norm=mpl.colors.BoundaryNorm(levels,
#     ncolors=cmap.N, clip=False), zorder=12)
# ax2.pcolormesh(lng_mohegh[::skip], lat_mohegh[::skip], 
#     no2_mohegh_2019[::skip,::skip], cmap=cmap, transform=ccrs.PlateCarree(), 
#     norm=mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# # Bias
# mbb = ax3.scatter(aqs_amean['Longitude'], aqs_amean['Latitude'], 
#     c=(aqs_amean['Mohegh NO2']-aqs_amean['Arithmetic Mean']), s=2, 
#     cmap=cmapb, transform=ccrs.PlateCarree(), 
#     norm=mpl.colors.BoundaryNorm(levelsb, ncolors=cmapb.N, clip=False), 
#     zorder=12)
# # Cooper et al. (2020) dataset
# ax4.scatter(aqs_hourly['Longitude'], aqs_hourly['Latitude'], 
#     c=aqs_hourly['Sample Measurement'],s=2, cmap=cmap, 
#     transform=ccrs.PlateCarree(), norm=mpl.colors.BoundaryNorm(levels, 
#     ncolors=cmap.N, clip=False), zorder=12)
# ax5.pcolormesh(lng_cooper[::skip], lat_cooper[::skip], 
#     no2_cooper_2019[::skip,::skip],cmap=cmap, transform=ccrs.PlateCarree(), 
#     norm=mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# ax6.scatter(aqs_hourly['Longitude'], aqs_hourly['Latitude'], 
#     c=(aqs_hourly['Cooper NO2']-aqs_hourly['Sample Measurement']), s=2, 
#     cmap=cmapb, transform=ccrs.PlateCarree(), 
#     norm=mpl.colors.BoundaryNorm(levelsb, ncolors=cmapb.N, clip=False), 
#     zorder=12)
# # Axis titles
# ax1.set_title('(a) AQS 24-hour average', loc='left')
# ax2.set_title('(b) Anenberg, Mohegh, et al. (2021)', loc='left')
# ax3.set_title('(c) Bias (b-a)', loc='left')
# ax4.set_title('(d) AQS overpass average', loc='left')
# ax5.set_title('(e) Cooper et al. (2020)', loc='left')
# ax6.set_title('(f) Bias (e-d)', loc='left')
# #'; ${\mathregular{(Anenberg\:&\:Mohegh} - \mathregular{AQS)}}$'
# #; $\frac{\mathregular{(Anenberg\:&\:Mohegh}-\mathregular{AQS)}}
# # {\mathregular{AQS}}$$\mathregular{\times}$100%'
# for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
#     ax.set_extent([-126,-66.6,24.5,46])
#     ax.add_feature(cfeature.OCEAN, facecolor='lightgrey', zorder=15)
#     # ax.add_feature(cfeature.LAKES, facecolor='lightgrey')
#     state_borders = cfeature.NaturalEarthFeature(category='cultural', 
#         name='admin_1_states_provinces_lakes', scale='10m', 
#         facecolor='None', zorder=16)
#     ax.add_feature(state_borders, edgecolor='black', lw=0.1)    
# plt.subplots_adjust(left=0.02, right=0.95, hspace=0.05, wspace=0.1, 
#     top=0.92, bottom=0.14)
# # Args are [left, bottom, width, height]
# posn1 = ax4.get_position()
# posn2 = ax5.get_position()
# p10 = posn1.x0
# p11 = posn1.x1
# p20 = posn2.x0
# p21 = posn2.x1
# cax = fig.add_axes([(p10+p11)/2, posn1.y0-0.05, ((p20+p21)/2)-(p10+p11)/2, 
#     0.02])
# fig.colorbar(mb, orientation='horizontal', extend='max', cax=cax, 
#     label='[ppb]')
# posn3 = ax6.get_position()
# cax = fig.add_axes([posn3.x0, posn3.y0-0.05, posn3.x1-posn3.x0, 0.02])
# fig.colorbar(mbb, orientation='horizontal', extend='both', cax=cax,
#     label='[ppb]')
# plt.savefig(DIR_FIG+'maps_no2_anenberg_cooper.png', dpi=500)

"""SCATTERPLOTS OF ANENBERG, COOPER, AND KIM NO2 AND BIAS AGAINST AQS"""
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# fig = plt.figure(figsize=(6,6))
# ax1 = plt.subplot2grid((2,2),(0,0))
# ax2 = plt.subplot2grid((2,2),(0,1))
# ax3 = plt.subplot2grid((2,2),(1,0))
# ax4 = plt.subplot2grid((2,2),(1,1))
# col_rural = '#66c2a5'
# col_urban = '#fc8d62'
# # Anenberg, Mohegh et al. (2021) dataset
# ax1.plot(aqs_amean_2019['Arithmetic Mean'], aqs_amean_2019['Mohegh NO2'], 
#     'ko', markersize=3, alpha=0.6)
# # Add line of best fit
# m, b = np.polyfit(aqs_amean_2019['Arithmetic Mean'], 
#     aqs_amean_2019['Mohegh NO2'], 1)
# xs = np.linspace(0,35,500)
# ax1.plot(xs, m*xs+b, ls='-', color='k')
# # Add text
# ax1.text(0.03, 0.93, 'MB = %.2f ppb'%np.mean(aqs_amean_2019['Mohegh NO2']-
#     aqs_amean_2019['Arithmetic Mean']), ha='left', va='center', 
#     transform=ax1.transAxes, fontsize=8)
# ax1.text(0.03, 0.86, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%m, ha='left',
#     va='center', transform=ax1.transAxes, fontsize=8)
# ax1.text(0.03, 0.79, 'b = %.2f ppb'%b, ha='left', va='center', 
#     transform=ax1.transAxes, fontsize=8)
# # Urban vs. rural 
# amean_rural = aqs_amean_2019.loc[aqs_amean_2019['Rural-Urban']=='Rural']
# amean_urban = aqs_amean_2019.loc[aqs_amean_2019['Rural-Urban']=='Urban']
# ax2.plot(amean_urban['Arithmetic Mean'], amean_urban['Mohegh NO2'], 'o',
#     color=col_urban, markersize=3, alpha=0.6)
# ax2.plot(amean_rural['Arithmetic Mean'], amean_rural['Mohegh NO2'], 'o',
#     color=col_rural, markersize=3, alpha=0.6)
# mu, bu = np.polyfit(amean_urban['Arithmetic Mean'], 
#     amean_urban['Mohegh NO2'], 1)
# mr, br = np.polyfit(amean_rural['Arithmetic Mean'], 
#     amean_rural['Mohegh NO2'], 1)
# ax2.plot(xs, mu*xs+bu, color=col_urban)
# ax2.plot(xs, mr*xs+br, color=col_rural)
# ax2.text(0.03, 0.93, 'MB = %.2f ppb'%np.mean(amean_urban['Mohegh NO2']-
#     amean_urban['Arithmetic Mean']), ha='left', va='center', 
#     transform=ax2.transAxes, color=col_urban, fontsize=8)
# ax2.text(0.03, 0.86, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%mu, ha='left',
#     va='center', transform=ax2.transAxes, color=col_urban, fontsize=8)
# ax2.text(0.03, 0.79, 'b = %.2f ppb'%bu, ha='left', va='center', 
#     transform=ax2.transAxes, color=col_urban, fontsize=8)
# ax2.text(0.52, 0.21, 'MB = %.2f ppb'%np.mean(amean_rural['Mohegh NO2']-
#     amean_rural['Arithmetic Mean']), ha='left', va='center', 
#     transform=ax2.transAxes, color=col_rural, fontsize=8)
# ax2.text(0.52, 0.14, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%mr, ha='left',
#     va='center', transform=ax2.transAxes, color=col_rural, fontsize=8)
# ax2.text(0.52, 0.07, 'b = %.2f ppb'%br, ha='left', va='center', 
#     transform=ax2.transAxes, color=col_rural, fontsize=8)
# # Cooper et al. (2020) dataset
# ax3.plot(aqs_hourly_2019['Sample Measurement'], aqs_hourly_2019['Cooper NO2'], 
#     'ko', markersize=3, alpha=0.6)
# m, b = np.polyfit(aqs_hourly_2019['Sample Measurement'], 
#     aqs_hourly_2019['Cooper NO2'], 1)
# ax3.plot(xs, m*xs+b, ls='-', color='k')
# ax3.text(0.03, 0.93, 'MB = %.2f ppb'%np.mean(aqs_hourly_2019['Cooper NO2']-
#     aqs_hourly_2019['Sample Measurement']), ha='left', va='center', 
#     transform=ax3.transAxes, fontsize=8)
# ax3.text(0.03, 0.86, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%m, ha='left',
#     va='center', transform=ax3.transAxes, fontsize=8)
# ax3.text(0.03, 0.79, 'b = %.2f ppb'%b, ha='left', va='center', 
#     transform=ax3.transAxes, fontsize=8)
# hourly_rural = aqs_hourly_2019.loc[aqs_hourly_2019['Rural-Urban']=='Rural']
# hourly_urban = aqs_hourly_2019.loc[aqs_hourly_2019['Rural-Urban']=='Urban']
# ax4.plot(hourly_urban['Sample Measurement'], hourly_urban['Cooper NO2'], 'o',
#     color=col_urban, markersize=3, alpha=0.6)
# ax4.plot(hourly_rural['Sample Measurement'], hourly_rural['Cooper NO2'], 'o',
#     color=col_rural, markersize=3, alpha=0.6)
# mu, bu = np.polyfit(hourly_urban['Sample Measurement'], 
#     hourly_urban['Cooper NO2'], 1)
# mr, br = np.polyfit(hourly_rural['Sample Measurement'], 
#     hourly_rural['Cooper NO2'], 1)
# ax4.plot(xs, mu*xs+bu, color=col_urban)
# ax4.plot(xs, mr*xs+br, color=col_rural)
# ax4.text(0.03, 0.93, 'MB = %.2f ppb'%np.mean(hourly_urban['Cooper NO2']-
#     hourly_urban['Sample Measurement']), ha='left', va='center', 
#     transform=ax4.transAxes, color=col_urban, fontsize=8)
# ax4.text(0.03, 0.86, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%mu, ha='left',
#     va='center', transform=ax4.transAxes, color=col_urban, fontsize=8)
# ax4.text(0.03, 0.79, 'b = %.2f ppb'%bu, ha='left', va='center', 
#     transform=ax4.transAxes, color=col_urban, fontsize=8)
# ax4.text(0.52, 0.21, 'MB = %.2f ppb'%np.mean(hourly_rural['Cooper NO2']-
#     hourly_rural['Sample Measurement']), ha='left', va='center', 
#     transform=ax4.transAxes, color=col_rural, fontsize=8)
# ax4.text(0.52, 0.14, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%mr, ha='left',
#     va='center', transform=ax4.transAxes, color=col_rural, fontsize=8)
# ax4.text(0.52, 0.07, 'b = %.2f ppb'%br, ha='left', va='center', 
#     transform=ax4.transAxes, color=col_rural, fontsize=8)
# for ax in [ax1, ax2, ax3, ax4]:
#     lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#         np.max([ax.get_xlim(), ax.get_ylim()]),]  # max of both axes
#     # Now plot both limits against eachother
#     ax.plot(lims, lims, '--', color='darkgrey', alpha=0.75, zorder=0)
#     ax.set_aspect('equal')
#     ax.set_xlim([0,35])
#     ax.set_ylim([0,35])
# plt.subplots_adjust(top=0.95, hspace=0.3)
# ax1.set_title('(a)', loc='left')
# ax1.set_xlabel('AQS 24-hour average')
# ax1.set_ylabel('Anenberg, Mohegh, et al. (2021)')
# ax2.set_title('(b)', loc='left')
# ax2.set_xlabel('AQS 24-hour average')
# ax3.set_title('(c)', loc='left')
# ax3.set_xlabel('AQS overpass average')
# ax3.set_ylabel('Cooper et al. (2020)')
# ax4.set_title('(d)', loc='left')
# ax4.set_xlabel('AQS overpass average')
# # Custom legend
# custom_lines = [Line2D([0],[0], marker='o', color=col_urban, lw=1),
#     Line2D([0], [0], marker='o', color=col_rural, lw=1)]
# ax3.legend(custom_lines, ['Urban', 'Rural'], 
#     bbox_to_anchor=(1.06, -0.38), loc=8, ncol=2, fontsize=10, 
#     frameon=False)
# plt.savefig(DIR_FIG+'scatter_no2_anenberg_cooper.png', dpi=500)
# # Same as above but for CACES NO2
# fig = plt.figure(figsize=(6,3))
# ax1 = plt.subplot2grid((1,2),(0,0))
# ax2 = plt.subplot2grid((1,2),(0,1))
# col_rural = '#66c2a5'
# col_urban = '#fc8d62'
# ax1.plot(aqs_amean_2015['Arithmetic Mean'], aqs_amean_2015['Cases NO2'], 
#     'ko', markersize=3, alpha=0.6)
# # Add line of best fit
# m, b = np.polyfit(aqs_amean_2015['Arithmetic Mean'], 
#     aqs_amean_2015['Cases NO2'], 1)
# xs = np.linspace(0,35,500)
# ax1.plot(xs, m*xs+b, ls='-', color='k')
# ax1.text(0.03, 0.93, 'MB = %.2f ppb'%np.mean(aqs_amean_2015['Cases NO2']-
#     aqs_amean_2015['Arithmetic Mean']), ha='left', va='center', 
#     transform=ax1.transAxes, fontsize=8)
# ax1.text(0.03, 0.86, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%m, ha='left',
#     va='center', transform=ax1.transAxes, fontsize=8)
# ax1.text(0.03, 0.79, 'b = %.2f ppb'%b, ha='left', va='center', 
#     transform=ax1.transAxes, fontsize=8)
# amean_rural = aqs_amean_2015.loc[aqs_amean_2015['Rural-Urban']=='Rural']
# amean_urban = aqs_amean_2015.loc[aqs_amean_2015['Rural-Urban']=='Urban']
# ax2.plot(amean_urban['Arithmetic Mean'], amean_urban['Cases NO2'], 'o',
#     color=col_urban, markersize=3, alpha=0.6)
# ax2.plot(amean_rural['Arithmetic Mean'], amean_rural['Cases NO2'], 'o',
#     color=col_rural, markersize=3, alpha=0.6)
# mu, bu = np.polyfit(amean_urban['Arithmetic Mean'], 
#     amean_urban['Cases NO2'], 1)
# mr, br = np.polyfit(amean_rural['Arithmetic Mean'], 
#     amean_rural['Cases NO2'], 1)
# ax2.plot(xs, mu*xs+bu, color=col_urban)
# ax2.plot(xs, mr*xs+br, color=col_rural)
# ax2.text(0.03, 0.93, 'MB = %.2f ppb'%np.mean(amean_urban['Cases NO2']-
#     amean_urban['Arithmetic Mean']), ha='left', va='center', 
#     transform=ax2.transAxes, color=col_urban, fontsize=8)
# ax2.text(0.03, 0.86, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%mu, ha='left',
#     va='center', transform=ax2.transAxes, color=col_urban, fontsize=8)
# ax2.text(0.03, 0.79, 'b = %.2f ppb'%bu, ha='left', va='center', 
#     transform=ax2.transAxes, color=col_urban, fontsize=8)
# ax2.text(0.52, 0.21, 'MB = %.2f ppb'%np.mean(amean_rural['Cases NO2']-
#     amean_rural['Arithmetic Mean']), ha='left', va='center', 
#     transform=ax2.transAxes, color=col_rural, fontsize=8)
# ax2.text(0.52, 0.14, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%mr, ha='left',
#     va='center', transform=ax2.transAxes, color=col_rural, fontsize=8)
# ax2.text(0.52, 0.07, 'b = %.2f ppb'%br, ha='left', va='center', 
#     transform=ax2.transAxes, color=col_rural, fontsize=8)
# for ax in [ax1, ax2]:
#     lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
#         np.max([ax.get_xlim(), ax.get_ylim()]),] 
#     ax.plot(lims, lims, '--', color='darkgrey', alpha=0.75, zorder=0)
#     ax.set_aspect('equal')
#     ax.set_xlim([0,35])
#     ax.set_ylim([0,35])
# plt.subplots_adjust(top=0.95, hspace=0.3)
# ax1.set_title('(a)', loc='left')
# ax1.set_xlabel('AQS 24-hour average')
# ax1.set_ylabel('Kim et al. (2020)')
# ax2.set_title('(b)', loc='left')
# ax2.set_xlabel('AQS 24-hour average')
# plt.subplots_adjust(bottom=0.23, top=0.88)
# custom_lines = [Line2D([0],[0], marker='o', color=col_urban, lw=1),
#     Line2D([0], [0], marker='o', color=col_rural, lw=1)]
# ax1.legend(custom_lines, ['Urban', 'Rural'], 
#     bbox_to_anchor=(1.06, -0.38), loc=8, ncol=2, fontsize=10, 
#     frameon=False)
# plt.savefig(DIR_FIG+'scatter_no2_kim.png', dpi=500)

"""ATTRIBUTABLE FRACTION FOR BAY AREA FROM ANENBERG, COOPER DATASETS"""
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# # Define mapping variables
# cmap = plt.cm.get_cmap('OrRd')
# levels = np.arange(0,65,5)
# proj = ccrs.PlateCarree(central_longitude=0.0)
# # Open primary roads
# sfi = '06'
# shp = shapereader.Reader(DIR_GEO+
#     'tigerline/roads/tl_2019_%s_prisecroads/'%sfi+
#     'tl_2019_%s_prisecroads.shp'%sfi)   
# roads = list(shp.geometries())
# roads = cfeature.ShapelyFeature(roads, proj)
# # Mohegh 1 km NO2 for CONUS, 2019
# lng_mohegh, lat_mohegh, no2_mohegh_2019 = edf_open.open_no2pop_tif(
#     '2019_final_1km_usa', -999., 'NO2', bayarea)
# # Cooper ~2.8km NO2 for CONUS, 2019
# lng_cooper, lat_cooper, no2_cooper_2019 = edf_open.open_cooperno2(bayarea)
# af_mohegh_2019 = calculate_af(no2_mohegh_2019, 1.36)
# af_cooper_2019 = calculate_af(no2_cooper_2019, 1.36)

""" BAY AREA MEAN NO2 AND ATTRIBUTABLE FRACTION FOR ANENBERG ET AL """
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['hatch.linewidth'] = 0.3     
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# proj = ccrs.PlateCarree(central_longitude=0.0)
# fig = plt.figure(figsize=(6,6))
# proj = ccrs.PlateCarree(central_longitude=0.0)
# ax1 = plt.subplot2grid((2,1),(0,0), projection=proj)
# ax1.set_title('(a) Anenberg, Mohegh, et al. (2021)', loc='left')
# ax2 = plt.subplot2grid((2,1),(1,0), projection=proj)
# ax2.set_title('(b) Attributable fraction', loc='left')
# cmap1 = plt.cm.get_cmap('magma_r',12)
# norm1 = matplotlib.colors.Normalize(vmin=0, vmax=20)
# cmap2 = plt.cm.get_cmap('OrRd',12)
# norm2 = matplotlib.colors.Normalize(vmin=0, vmax=65)
# ax1.pcolormesh(lng_mohegh, lat_mohegh, no2_mohegh_2019,
#     cmap=cmap1, vmin=levels1[0], vmax=levels1[-1], transform=ccrs.PlateCarree(), 
#     norm=mpl.colors.BoundaryNorm(levels1, ncolors=cmap1.N, clip=False))
# ax2.pcolormesh(lng_mohegh, lat_mohegh, af_mohegh_2019*100, cmap=cmap2, 
#     vmin=levels2[0], vmax=levels2[-1], transform=ccrs.PlateCarree(), 
#     norm=mpl.colors.BoundaryNorm(levels2, ncolors=cmap2.N, clip=False))
# for ax in [ax1, ax2]:
#     ax.set_extent(bayarea)
#     ax.add_feature(roads, facecolor='None', edgecolor='k', 
#         zorder=16, lw=0.5)    
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#         facecolor='silver', name='ocean', lw=0.5, edgecolor='None'), zorder=10)
# # Add colorbars
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('bottom', size='5%', pad=0.1, axes_class=plt.Axes)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap1, norm=norm1, 
#     ticks=np.arange(0,25,5), spacing='proportional', orientation='horizontal', 
#     extend='max', label='[ppb]')  
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes('bottom', size='5%', pad=0.1, axes_class=plt.Axes)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap2, norm=norm2, 
#     spacing='proportional', orientation='horizontal', extend='max', 
#     ticks=np.arange(0,70,10), label='[%]')  
# plt.subplots_adjust(hspace=0.4)
# plt.savefig(DIR_FIG+'maps_no2_af_anenberg.png', dpi=500)

""" BAY AREA MEAN NO2 AND ATTRIBUTABLE FRACTION FOR COOPER ET AL """
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['hatch.linewidth'] = 0.3     
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# proj = ccrs.PlateCarree(central_longitude=0.0)
# fig = plt.figure(figsize=(6,6))
# proj = ccrs.PlateCarree(central_longitude=0.0)
# ax1 = plt.subplot2grid((2,1),(0,0), projection=proj)
# ax1.set_title('(a) Cooper et al. (2020)', loc='left')
# ax2 = plt.subplot2grid((2,1),(1,0), projection=proj)
# ax2.set_title('(b) Attributable fraction', loc='left')
# cmap1 = plt.cm.get_cmap('magma_r',12)
# norm1 = matplotlib.colors.Normalize(vmin=0, vmax=20)
# cmap2 = plt.cm.get_cmap('OrRd',12)
# norm2 = matplotlib.colors.Normalize(vmin=0, vmax=65)
# ax1.pcolormesh(lng_cooper, lat_cooper, no2_cooper_2019, cmap=cmap1, 
#     vmin=levels1[0], vmax=levels1[-1], transform=ccrs.PlateCarree(), 
#     norm=norm1)
# ax2.pcolormesh(lng_cooper, lat_cooper, af_cooper_2019*100, cmap=cmap2, 
#     vmin=levels2[0], vmax=levels2[-1], transform=ccrs.PlateCarree(), 
#     norm=norm2)
# for ax in [ax1, ax2]:
#     ax.set_extent(bayarea)
#     ax.add_feature(roads, facecolor='None', edgecolor='k', 
#         zorder=16, lw=0.5)    
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#         facecolor='silver', name='ocean', lw=0.5, edgecolor='None'), zorder=10)
# # Add colorbars
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('bottom', size='5%', pad=0.1, axes_class=plt.Axes)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap1, norm=norm1, 
#     spacing='proportional', orientation='horizontal', extend='max', 
#     ticks=np.arange(0,25,5), label='[ppb]')
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes('bottom', size='5%', pad=0.1, axes_class=plt.Axes)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap2, norm=norm2, 
#     spacing='proportional', orientation='horizontal', extend='max', 
#     ticks=np.arange(0,70,10), label='[%]')  
# plt.subplots_adjust(hspace=0.4)
# plt.savefig(DIR_FIG+'maps_no2_af_cooper.png', dpi=500)

""" BAY AREA MEAN NO2 AND ATTRIBUTABLE FRACTION FOR KIM ET AL """
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['hatch.linewidth'] = 0.3     
# import matplotlib as mpl
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# proj = ccrs.PlateCarree(central_longitude=0.0)
# no2_caces_2015 = pd.read_csv('/Users/ghkerr/Downloads/cases_no2_2015_tract.csv',
#     sep=',', engine='python')
# no2_caces_2015['fips'] = no2_caces_2015['fips'].astype(str).apply(lambda x: x.zfill(11))
# no2_caces_2015['fips_state'] = no2_caces_2015['fips'].apply(lambda x: x[:2])
# # Calculate attributable fration 
# af_caces_2015 = calculate_af(np.array(no2_caces_2015['pred_wght']), 1.36)
# no2_caces_2015['AF'] = af_caces_2015
# # Load shapefile for CA
# shp = shapereader.Reader('/Users/ghkerr/Downloads/tl_2010_06_tract10/'+
#     'tl_2010_06_tract10.shp')
# recordsi = shp.records()
# tractsi = shp.geometries()
# records = list(recordsi)
# tracts = list(tractsi)
# fig = plt.figure(figsize=(6,6))
# proj = ccrs.PlateCarree(central_longitude=0.0)
# ax1 = plt.subplot2grid((2,1),(0,0), projection=proj)
# ax1.set_title('(a) Kim et al. (2020)', loc='left')
# ax2 = plt.subplot2grid((2,1),(1,0), projection=proj)
# ax2.set_title('(b) Attributable fraction', loc='left')
# cmap1 = plt.cm.get_cmap('magma_r',12)
# norm1 = matplotlib.colors.Normalize(vmin=0, vmax=20)
# cmap2 = plt.cm.get_cmap('OrRd',13)
# norm2 = matplotlib.colors.Normalize(vmin=0, vmax=65)
# # Loop through census tracts in state
# for i,x in enumerate(tracts):
#     tract = tracts[i]
#     recordid = records[i].attributes['GEOID10']
#     # 
#     caces_recordid = no2_caces_2015.loc[no2_caces_2015['fips']==recordid]
#     if caces_recordid.empty==False:
#         ax1.add_geometries([tract], proj, facecolor=cmap1(norm1(
#             caces_recordid['pred_wght'].values[0])), edgecolor='None',
#             zorder=10)     
#         ax2.add_geometries([tract], proj, facecolor=cmap2(norm2(
#             caces_recordid['AF'].values[0]*100)), edgecolor='None',
#             zorder=10)     
# # Aesthetics 
# for ax in [ax1, ax2]:
#     sfi = '06'
#     shproads = shapereader.Reader(DIR_GEO+
#         'tigerline/roads/tl_2019_%s_prisecroads/'%sfi+
#         'tl_2019_%s_prisecroads.shp'%sfi)   
#     roads = list(shproads.geometries())
#     roads = cfeature.ShapelyFeature(roads, proj)
#     ax.set_extent(bayarea)
#     ax.add_feature(roads, facecolor='None', edgecolor='k', 
#         zorder=16, lw=0.5)    
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#         facecolor='silver', name='ocean', lw=0.5, edgecolor='None'), 
#         zorder=12)
# # Add colorbars
# divider = make_axes_locatable(ax1)
# cax = divider.append_axes('bottom', size='5%', pad=0.1, axes_class=plt.Axes)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap1, norm=norm1, 
#     spacing='proportional', orientation='horizontal', extend='max', 
#     label='[ppb]')  
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes('bottom', size='5%', pad=0.1, axes_class=plt.Axes)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap2, norm=norm2, 
#     spacing='proportional', orientation='horizontal', extend='max', 
#     label='[%]')  
# plt.subplots_adjust(hspace=0.4)
# plt.savefig(DIR_FIG+'maps_no2_af_kim.png', dpi=500)

# # # # #  Scale Larkin dataset to 2019 values
# no2_larkin_scaled = scale_larkin(lng_larkin, lat_larkin, no2_larkin, 
#     lng_mohegh, lat_mohegh, no2_mohegh_2010_2012, no2_mohegh_2019)

# # # # # Calculate attributable fraction
# af_mohegh_2019 = calculate_af(no2_mohegh_2019, 1.36)
# af_cooper_2019 = calculate_af(no2_cooper_2019, 1.36)
# af_larkin = calculate_af(no2_larkin, 1.36)
# af_larkin_scaled = calculate_af(no2_larkin_scaled, 1.36)
# af_mohegh_2010_2012 = calculate_af(no2_mohegh_2010_2012, 1.36)


# # For 1 km (Mohegh) 
# lng_pop1_4_1km, lat_pop1_4_1km, pop1_4_1km = open_no2pop_tif(
#     '1-4_final_1km', -9999., 'pop', conus)
# lng_pop5_9_1km, lat_pop5_9_1km, pop5_9_1km = open_no2pop_tif(
#     '5-9_final_1km', -9999., 'pop', conus)
# lng_pop10_14_1km, lat_pop10_14_1km, pop10_14_1km = open_no2pop_tif(
#     '10-14_final_1km', -9999., 'pop', conus)
# lng_pop15_18_1km, lat_pop15_18_1km, pop15_18_1km = open_no2pop_tif(
#     '15-18_final_1km', -9999., 'pop', conus)
# # For Mohegh 1km dataset
# calculate_incidence_UI(no2_mohegh_2010_2012, pop1_4_1km, pop5_9_1km, 
#     pop10_14_1km, pop15_18_1km, 'conus_mohegh2010-2012_1km_GBD2019')
# calculate_incidence_UI(no2_mohegh_2019, pop1_4_1km, pop5_9_1km, 
#     pop10_14_1km, pop15_18_1km, 'conus_mohegh2019_1km_GBD2019')

# # Load Larkin 100m NO2 (2010-2012)
# lng_larkin, lat_larkin, no2_larkin = open_no2pop_tif('lur_average2', 
#     128., 'NO2', conus)
# print('Larkin NO2 loaded!')

# # # # # #  Scale Larkin dataset to 2019 values
# no2_larkin_scaled = scale_larkin(lng_larkin, lat_larkin, no2_larkin, 
    # lng_mohegh, lat_mohegh, no2_mohegh_2010_2012, no2_mohegh_2019)
# # Load 100m population
# lng_pop1_4_100m, lat_pop1_4_100m, pop1_4_100m = open_no2pop_tif(
#     '1-4_final', -9999., 'pop', conus)
# print('100m 1-4 population loaded!')
# lng_pop5_9_100m, lat_pop5_9_100m, pop5_9_100m = open_no2pop_tif(
#     '5-9_final', -9999., 'pop', conus)
# print('100m 5-9 population loaded!')
# lng_pop10_14_100m, lat_pop10_14_100m, pop10_14_100m = open_no2pop_tif(
#     '10-14_final', -9999., 'pop', conus)
# print('100m 10-14 population loaded!')
# lng_pop15_18_100m, lat_pop15_18_100m, pop15_18_100m = open_no2pop_tif(
#     '15-18_final', -9999., 'pop', conus)
# print('100m 15-18 population loaded!')
# calculate_incidence_UI(no2_larkin, pop1_4_100m, pop5_9_100m, 
#     pop10_14_100m, pop15_18_100m, 'conus_larkin2010-2012_1km_GBD2019')

"""DOWNSCALING EXAMPLE"""
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import matplotlib as mpl
# # Select arbitrary part of Mohegh NO2 grid
# no2_coarse_old = no2_mohegh_2010_2012[16:21,41:48]
# no2_coarse_new = no2_mohegh_2019[16:21,41:48]
# lat_coarse = lat_mohegh[16:21]
# lng_coarse = lng_mohegh[41:48]
# # Select colocated grid cells from Mohegh dataset
# lat1 = geo_idx(lat_coarse[0], lat_larkin)
# lat2 = geo_idx(lat_coarse[-1], lat_larkin)
# lng1 = geo_idx(lng_coarse[0], lng_larkin)
# lng2 = geo_idx(lng_coarse[-1], lng_larkin)
# # Select smaller fine resolution grid representative of Larkin dataset
# lat_fine = lat_larkin[lat1:lat2+1]
# lng_fine = lng_larkin[lng1:lng2+1]
# no2_fine = no2_larkin[lat1:lat2+1,lng1:lng2+1]
# # Scale fake data
# no2_fake_scaled = scale_larkin(lng_fine, lat_fine, no2_fine, 
#     lng_coarse, lat_coarse, no2_coarse_old, no2_coarse_new)
# # Define mapping variables
# cmap = plt.cm.get_cmap('YlGnBu')
# levels = np.arange(5,25,2)
# norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False)
# # Plotting 
# fig = plt.figure(figsize=(8,8))
# ax1 = plt.subplot2grid((2,2),(0,0))
# ax2 = plt.subplot2grid((2,2),(0,1))
# ax3 = plt.subplot2grid((2,2),(1,0))
# ax4 = plt.subplot2grid((2,2),(1,1))
# mb = ax1.pcolormesh(lng_coarse, lat_coarse, no2_coarse_old, shading='auto',
#     cmap=cmap, norm=norm, edgecolors='k')
# ax2.pcolormesh(lng_coarse, lat_coarse, no2_coarse_new, shading='auto',
#     cmap=cmap, norm=norm, edgecolors='k')              
# ax3.pcolormesh(lng_fine, lat_fine, no2_fine, shading='auto',
#     cmap=cmap, edgecolors='k', norm=norm)
# ax4.pcolormesh(lng_fine, lat_fine, no2_fake_scaled, shading='auto',
#     cmap=cmap, edgecolors='k', norm=norm)
# # Add axis labels, titles to plots
# ax3.set_xlabel('Longitude [$^{\circ}$E]')
# ax4.set_xlabel('Longitude [$^{\circ}$E]')
# ax1.set_ylabel('Latitude [$^{\circ}$N]')
# ax3.set_ylabel('Latitude [$^{\circ}$N]')
# ax1.set_title('(a) 1 km (2010-2012)', loc='left')
# ax2.set_title('(b) 1 km (2019)', loc='left')
# ax3.set_title('(c) 100 m (2010-2012)', loc='left')
# ax4.set_title('(d) 100 m (2010-2012) downscaled to 2019', loc='left')
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_xlim([-122.27155, -122.2547])
#     ax.set_ylim([37.7726168+0.0045, 37.7726168+0.021])    
# # Remove scientific notation for longitudes
# ax3.ticklabel_format(useOffset=False, style='plain')
# ax4.ticklabel_format(useOffset=False, style='plain')
# # Stop x-tick label crowding
# label_format = '{:,.3f}'
# locs = [-122.270,-122.2675,-122.265,-122.2625,-122.26,-122.2575, -122.255]
# locslabels = ['-122.270','','-122.265','','-122.600','', '-122.255']
# for ax in [ax3, ax4]:
#     ax.xaxis.set_major_locator(mticker.FixedLocator(locs))
#     ax.set_xticklabels(locslabels)
# ax1.set_xticklabels([])
# ax2.set_xticklabels([])
# ax2.set_yticklabels([])
# ax4.set_yticklabels([])
# # Add values to heatmap
# for y in range(lat_coarse.shape[0]):
#     for x in range(lng_coarse.shape[0]):
#         ax1.text(lng_coarse[x], lat_coarse[y], '%.1f'%no2_coarse_old[y, x],
#             ha='center', va='center', color='w', clip_on=True)
#         ax2.text(lng_coarse[x], lat_coarse[y], '%.1f'%no2_coarse_new[y, x],
#             ha='center', va='center', color='k', clip_on=True)
# for y in range(lat_fine.shape[0]):
#     for x in range(lng_fine.shape[0]):
#         ax3.text(lng_fine[x], lat_fine[y], '%.1f'%no2_fine[y, x],
#             ha='center', va='center', color='w', fontsize=4, clip_on=True)
#         ax4.text(lng_fine[x], lat_fine[y], '%.1f'%no2_fake_scaled[y, x],
#             ha='center', va='center', color='k', fontsize=4, clip_on=True)  
# # Add colorbar
# fig.subplots_adjust(bottom=0.18, top=0.95, wspace=0.1)
# cbar_ax = fig.add_axes([0.30, 0.07, 0.4, 0.03])
# fig.colorbar(mb, extend='both', cax=cbar_ax, orientation='horizontal', 
#     label='NO$_{2}$ [ppbv]')
# plt.savefig(DIR_FIG+'downscalingexample.png', dpi=500)

"""MEAN NO2 CONCENTRATIONS"""
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# # Define mapping variables
# cmap = plt.cm.get_cmap('YlGnBu')
# levels = np.arange(0,30+3,3)
# proj = ccrs.PlateCarree(central_longitude=0.0)
# # Open primary roads
# sfi = '06'
# shp = shapereader.Reader(DIR_GEO+
#     'tigerline/roads/tl_2019_%s_prisecroads/'%sfi+
#     'tl_2019_%s_prisecroads.shp'%sfi)   
# roads = list(shp.geometries())
# roads = cfeature.ShapelyFeature(roads, proj)
# # Plotting
# fig = plt.figure(figsize=(8,4))
# ax1 = plt.subplot2grid((2,2),(0,0), projection=proj)
# ax2 = plt.subplot2grid((2,2),(0,1), projection=proj)
# ax3 = plt.subplot2grid((2,2),(1,0), projection=proj)
# ax4 = plt.subplot2grid((2,2),(1,1), projection=proj)
# mb = ax1.pcolormesh(lng_mohegh, lat_mohegh, no2_mohegh_2010_2012,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# ax2.pcolormesh(lng_mohegh, lat_mohegh, no2_mohegh_2019,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# ax3.pcolormesh(lng_larkin, lat_larkin, no2_larkin,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# ax4.pcolormesh(lng_larkin, lat_larkin, no2_larkin_scaled,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# # Axis titles
# ax1.set_title('(a) 1 km (2010-2012)', loc='left')
# ax2.set_title('(b) 1 km (2019)', loc='left')
# ax3.set_title('(c) 100 m (2010-2012)', loc='left')
# ax4.set_title('(d) 100 m (2010-2012) downscaled to 2019', loc='left')
# for ax in [ax1, ax2, ax3, ax4]:
#     ax.set_extent(clip)
#     ax.add_feature(roads, facecolor='None', edgecolor='k', 
#         zorder=16, lw=0.5)    
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#         facecolor='silver', name='ocean', lw=0.5, edgecolor='None'), zorder=0)
# # Add text corresponding to mean NO2 concentrations and standard deviation
# mu1, sigma1 = np.nanmean(no2_mohegh_2010_2012), np.nanstd(no2_mohegh_2010_2012)
# ax1.text(0.02, 0.12,'$\mathregular{\mu}$ = %.2f\n'%(mu1)+
#     '$\mathregular{\sigma}$ = %.2f'%(sigma1), ha='left', va='center', 
#     transform=ax1.transAxes)
# mu2, sigma2 = np.nanmean(no2_mohegh_2019), np.nanstd(no2_mohegh_2019)
# ax2.text(0.02, 0.12,'$\mathregular{\mu}$ = %.2f\n'%(mu2)+
#     '$\mathregular{\sigma}$ = %.2f'%(sigma2), ha='left', va='center', 
#     transform=ax2.transAxes)
# mu3, sigma3 = np.nanmean(no2_larkin), np.nanstd(no2_larkin)
# ax3.text(0.02, 0.12,'$\mathregular{\mu}$ = %.2f\n'%(mu3)+
#     '$\mathregular{\sigma}$ = %.2f'%(sigma3), ha='left', va='center', 
#     transform=ax3.transAxes)
# mu4, sigma4 = np.nanmean(no2_larkin_scaled), np.nanstd(no2_larkin_scaled)
# ax4.text(0.02, 0.12,'$\mathregular{\mu}$ = %.2f\n'%(mu4)+
#     '$\mathregular{\sigma}$ = %.2f'%(sigma4), ha='left', va='center', 
#     transform=ax4.transAxes)
# # Add colorbar
# fig.subplots_adjust(bottom=0.15, top=0.95, wspace=0.1)
# cbar_ax = fig.add_axes([0.30, 0.1, 0.4, 0.03])
# fig.colorbar(mb, extend='max', cax=cbar_ax, orientation='horizontal', 
#     label='NO$_{2}$ [ppbv]')
# plt.savefig(DIR_FIG+'no2_oakland.png', dpi=500)

"""ATTRIBUTABLE FRACTION"""
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io import shapereader
# # Define mapping variables
# cmap = plt.cm.get_cmap('OrRd')
# cmap2 = plt.cm.get_cmap('Blues_r')
# levels = np.arange(0,70,10)
# levels2 = np.arange(-18,0+2,2)
# proj = ccrs.PlateCarree(central_longitude=0.0)
# # Open primary roads
# sfi = '06'
# shp = shapereader.Reader(DIR_GEO+
#     'tigerline/roads/tl_2019_%s_prisecroads/'%sfi+
#     'tl_2019_%s_prisecroads.shp'%sfi)   
# roads = list(shp.geometries())
# roads = cfeature.ShapelyFeature(roads, proj)
# Plotting
# fig = plt.figure(figsize=(12,4))
# ax1 = plt.subplot2grid((2,3),(0,0), projection=proj)
# ax2 = plt.subplot2grid((2,3),(0,1), projection=proj)
# ax3 = plt.subplot2grid((2,3),(0,2), projection=proj)
# ax4 = plt.subplot2grid((2,3),(1,0), projection=proj)
# ax5 = plt.subplot2grid((2,3),(1,1), projection=proj)
# ax6 = plt.subplot2grid((2,3),(1,2), projection=proj)
# mb = ax1.pcolormesh(lng_mohegh, lat_mohegh, af_mohegh_2010_2012*100,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# ax2.pcolormesh(lng_mohegh, lat_mohegh, af_mohegh_2019*100,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# mb2 = ax3.pcolormesh(lng_mohegh, lat_mohegh, (af_mohegh_2019-
#     af_mohegh_2010_2012)*100, cmap=cmap2, vmin=levels2[0], 
#     vmax=levels2[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels2, ncolors=cmap.N, clip=False))
# ax4.pcolormesh(lng_larkin, lat_larkin, af_larkin*100,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# ax5.pcolormesh(lng_larkin, lat_larkin, af_larkin_scaled*100,
#     cmap=cmap, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False))
# ax6.pcolormesh(lng_larkin, lat_larkin, (af_larkin_scaled-
#     af_larkin)*100, cmap=cmap2, vmin=levels2[0], 
#     vmax=levels2[-1], transform=ccrs.PlateCarree(), 
#     norm = mpl.colors.BoundaryNorm(levels2, ncolors=cmap.N, clip=False))
# # Axis titles
# ax1.set_title('(a) 1 km (2010-2012)', loc='left')
# ax2.set_title('(b) 1 km (2019)', loc='left')
# ax3.set_title('(c) 1 km (2019$-$(2010-2012))', loc='left')
# ax4.set_title('(d) 100 m (2010-2012)', loc='left')
# ax5.set_title('(e) 100 m (2010-2012) downscaled to 2019', loc='left')
# ax6.set_title('(f) 100 m (2019$-$(2010-2012))', loc='left')
# for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
#     ax.set_extent(clip)
#     ax.add_feature(roads, facecolor='None', edgecolor='k', 
#         zorder=16, lw=0.5)    
#     ax.add_feature(cfeature.NaturalEarthFeature('physical', scale='10m',
#         facecolor='silver', name='ocean', lw=0.5, edgecolor='None'), zorder=0)
# # Add text corresponding to mean NO2 concentrations and standard deviation
# mu1, sigma1 = np.nanmean(af_mohegh_2010_2012*100), np.nanstd(af_mohegh_2010_2012*100)
# ax1.text(0.02, 0.12,'$\mathregular{\mu}$ = %d%%\n'%(mu1)+
#     '$\mathregular{\sigma}$ = %d%%'%(sigma1), ha='left', va='center', 
#     transform=ax1.transAxes)
# mu2, sigma2 = np.nanmean(af_mohegh_2019*100), np.nanstd(af_mohegh_2019*100)
# ax2.text(0.02, 0.12,'$\mathregular{\mu}$ = %d%%\n'%(mu2)+
#     '$\mathregular{\sigma}$ = %d%%'%(sigma2), ha='left', va='center', 
#     transform=ax2.transAxes)
# mu3 = np.nanmean((af_mohegh_2019-af_mohegh_2010_2012)*100)
# ax3.text(0.02, 0.12,'$\mathregular{\mu}$ = %d%%\n'%(mu3), ha='left', 
#     va='center', transform=ax3.transAxes)
# mu4, sigma4 = np.nanmean(af_larkin*100), np.nanstd(af_larkin*100)
# ax4.text(0.02, 0.12,'$\mathregular{\mu}$ = %d%%\n'%(mu4)+
#     '$\mathregular{\sigma}$ = %d%%'%(sigma4), ha='left', va='center', 
#     transform=ax4.transAxes)
# mu5, sigma5 = np.nanmean(af_larkin_scaled*100), np.nanstd(af_larkin_scaled*100)
# ax5.text(0.02, 0.12,'$\mathregular{\mu}$ = %d%%\n'%(mu5)+
#     '$\mathregular{\sigma}$ = %d%%'%(sigma5), ha='left', va='center', 
#     transform=ax5.transAxes)
# mu6 = np.nanmean((af_larkin_scaled-af_larkin)*100)
# ax6.text(0.02, 0.12,'$\mathregular{\mu}$ = %d%%\n'%(mu6), ha='left', 
#     va='center', transform=ax6.transAxes)
# # Add colorbars
# fig.subplots_adjust(bottom=0.15, top=0.95, wspace=0.1)
# cbar_ax = fig.add_axes([0.225, 0.1, 0.3, 0.03])
# fig.colorbar(mb, extend='max', cax=cbar_ax, orientation='horizontal', 
#     label='Attributable fraction [%]')
# fig.subplots_adjust(bottom=0.15, top=0.95, wspace=0.1)
# cbar_ax = fig.add_axes([0.7, 0.1, 0.15, 0.03])
# fig.colorbar(mb2, extend='both', cax=cbar_ax, orientation='horizontal', 
#     label='$\mathregular{\Delta}$ Attributable fraction [%]')
# plt.savefig(DIR_FIG+'af_oakland.png', dpi=500)




