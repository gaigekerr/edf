#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 22:02:09 2021

@author: ghkerr
"""
DIR_ROOT = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'
DIR_AF = DIR_ROOT+'af/'
DIR_CENSUS = DIR_ROOT+'acs/'
DIR_CROSS = DIR_ROOT
DIR_GEO = DIR_ROOT+'tigerline/'
DIR_OUT = DIR_ROOT+'harmonizedtables/'

import math
import time
from datetime import datetime
import numpy as np   
from scipy import stats
# Create output text file for print statements (i.e., crosswalk checks)
f = open(DIR_OUT+'harmonize_afacs_%s.txt'%(
    datetime.now().strftime('%Y-%m-%d-%H%M')), 'a')

def geo_idx(dd, dd_array):
    """Function searches for nearest decimal degree in an array of decimal 
    degrees and returns the index. np.argmin returns the indices of minimum 
    value along an axis. So subtract dd from all values in dd_array, take 
    absolute value and find index of minimum. n.b. Function edited on 
    5 Dec 2018 to calculate the resolution using the mode. Before this it 
    used the simple difference which could be erroraneous because of 
    longitudes in (-180-180) coordinates (the "jump" from -180 to 180 yielded 
    a large resolution).
    
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
    geo_idx = (np.abs(dd_array - dd)).argmin()
    # if distance from closest cell to intended value is 2x the value of the
    # spatial resolution, raise error 
    res = np.abs(stats.mode(np.diff(dd_array))[0][0])
    if np.abs(dd_array[geo_idx] - dd) > (2 * res):
        print('Closet index far from intended value!', file=f)
        return 
    return geo_idx

def harvesine(lon1, lat1, lon2, lat2):
    """Distance calculation, degree to km (Haversine method)

    Parameters
    ----------
    lon1 : float
        Longitude of point A
    lat1 : float
        Latitude of point A
    lon2 : float
        Longitude of point B
    lat2 : float
        Latitude of point A

    Returns
    -------
    d : float
        Distance between points A and B, units of degrees
    """
    rad = math.pi / 180  # degree to radian
    R = 6378.1  # earth average radius at equador (km)
    dlon = (lon2 - lon1) * rad
    dlat = (lat2 - lat1) * rad
    a = (math.sin(dlat / 2)) ** 2 + math.cos(lat1 * rad) * \
        math.cos(lat2 * rad) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d

def idwr(x, y, z, xi, yi):
    """Inverse distance weighting for interpolating missing TROPOMI NO2 
    column densities for census tracts that are too small to intersect with
    the regridded NO2 fields.

    Parameters
    ----------
    x : list
        Longitude
    y : list
        Latitudes
    z : list
        NO2 column densities
    xi : list
        Unknown longitude
    yi : TYPE
        Unknown latitude

    Returns
    -------
    lstxyzi : list
        List comprised of unknown longitude, latitude, and interpolated 
        TROPOMI column density for small census tract 
    """
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.sum(sumsup)
        sumsup = np.sum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        xyzi = [xi[p], yi[p], u]
        lstxyzi.append(xyzi)
    return lstxyzi

def harmonize_afacs(vintage, statefips):
    """For a given year and state, function harmonizes gridded (~1 km x 1 km) 
    maps of NO2-attributable paediatric asthma fractions to U.S. census tracts 
    and fetches tract-level ACS demographic information. Note that the 
    demographic information represents the ACS estimate vintage for the five 
    year period ending with the year of interest (e.g., vintage='2014-2018' 
    will be harmonized with attributable fractions for 2018). 
                          
    Parameters
    ----------
    vintage : str
        Historical 5-year ACS vintages (note that we have 2005-2009 to 2015-
        2019). 
    statefips : str
        State FIPS code ()

    Returns
    -------
    None
    """
    import netCDF4 as nc
    import pandas as pd
    import shapefile
    from shapely.geometry import shape, Point
    # # # # Open 2019 American Community Survey: 5-Year Data (2015-2019) 
    # from NHGIS
    #----------------------
    acs2 = pd.read_csv(DIR_CENSUS+'acs%s/'%vintage+'acs%sb.csv'%vintage, 
        sep=',', header=0, skiprows=[1], engine='python')      
    statename = acs2.loc[acs2['STATEA']==int(statefips)]['STATE'].values[0]
    print('HANDLING %s FOR %s:'%(statename.upper(), vintage[-4:]), file=f)
    print('----------------------------------------', file=f)
    acs1 = pd.read_csv(DIR_CENSUS+'acs%s/'%vintage+'acs%sa.csv'%vintage, 
        sep=',', header=0, skiprows=[1], engine='python')     
    # Merge; adapted from https://stackoverflow.com/questions/19125091/
    # pandas-merge-how-to-avoid-duplicating-columns
    acs = acs1.merge(acs2, left_index=True, right_index=True,
        how='outer', suffixes=('', '_y'))
    acs.drop(acs.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)
    # Columns to drop from ACS data
    tokeep = ['GISJOIN', # GIS Join Match Code
        'YEAR', # Data File Year
        'STATE', # State Name
        'STATEA', # State Code
        'COUNTYA', # County Code
        'TRACTA' # Census Tract Code
        ]
    print('# # # # U.S. Census Bureau/ACS data loaded!', file=f)
    
    # # # # Load crosswalk to link ACS parameter codes to unified names
    #----------------------
    crosswalk = pd.read_csv(DIR_CROSS+'crosswalk.csv', sep=',', header=0, 
        engine='python')
    # Select vintage from crosswalk file
    crosswalk = crosswalk.loc[crosswalk['vintage']==('acs'+vintage)]
    print('# # # # Crosswalk file loaded!', file=f)
    
    # # # # Open TIGER/Line shapefiles; note that for ACS estimate vintages
    # post 2009, the 2019 TIGER/Line shapefiles are used. For 2009, TIGER/Line
    # shapefiles from 2009 are used
    #----------------------
    if vintage == '2005-2009':
        fname = DIR_GEO+'tract_2000/tl_2009_%s_tract00/'%statefips
        r = shapefile.Reader(fname+'tl_2009_%s_tract00.shp'%statefips)        
    else: 
        fname = DIR_GEO+'tract_2010/tl_2019_%s_tract/'%statefips
        r = shapefile.Reader(fname+'tl_2019_%s_tract.shp'%statefips)
    # Get shapes, records
    tracts = r.shapes()
    records = r.records()
    print('# # # # TIGER/Line shapefiles loaded!', file=f)
    
    # # # # Read attributable fraction estimates
    #----------------------
    # Note that the AF dataset loaded corresponds to the final year of the 
    # 5-year ACS estimate
    af = nc.Dataset(DIR_AF+'af_anenbergmoheghno2_usa_%s.nc'%vintage[-4:])
    afyear = af.title[35:39] # Kludgey!
    lat_af = af.variables['latitude'][:].data
    lng_af = af.variables['longitude'][:].data
    fill_value = af.variables['AF'][:].fill_value
    af = af.variables['AF'][:].data
    af[af==fill_value] = np.nan
    print('# # # # Attributable fraction data loaded!', file=f)
    # Put to sleep for a hot sec so it doesn't screw up the progress bar
    time.sleep(2)
    
    # # # # Loop through tracts and find attributable fractions in tract
    # and demographic information
    #----------------------
    df = []
    for tract in np.arange(0, len(tracts), 1):
        record = records[tract]  
        # Extract GEOID of record
        if vintage=='2005-2009':
            geoid = record['CTIDFP00']
        else: 
            geoid = record['GEOID']  
        # Build a shapely polygon from shape
        tract = shape(tracts[tract]) 
        # Fetch demographic information for tract; note that GEOIDs aren't
        # available for all vintages, so we'll have to construct the GISJOIN 
        # code.
        # GISJOIN identifiers match the identifiers used in NHGIS data 
        # tables and boundary files. A block GISJOIN concatenates these codes:
        #    "G" prefix         This prevents applications from automatically 
        #                       reading the identifier as a number and, in 
        #                       effect, dropping important leading zeros
        #    State NHGIS code	3 digits (FIPS + "0"). NHGIS adds a zero to 
        #                       state FIPS codes to differentiate current 
        #                       states from historical territories.
        #    County NHGIS code	4 digits (FIPS + "0"). NHGIS adds a zero to 
        #                       county FIPS codes to differentiate current 
        #                       counties from historical counties.
        #    Census tract code	6 digits for 2000 and 2010 tracts. 1990 tract 
        #                       codes use either 4 or 6 digits.
        #    Census block code	4 digits for 2000 and 2010 blocks. 1990 block
        #                       codes use either 3 or 4 digits.
        # GEOID identifiers correspond to the codes used in most current 
        # Census sources (American FactFinder, TIGER/Line, Relationship Files, 
        # etc.). A block GEOID concatenates these codes:
        #    State FIPS code	2 digits
        #    County FIPS code	3 digits
        #    Census tract code	6 digits. 1990 tract codes that were 
        #                       originally 4 digits (as in NHGIS files) are 
        #                       extended to 6 with an appended "00" (as in 
        #                       Census Relationship Files).
        #    Census block code	4 digits for 2000 and 2010 blocks. 1990 block 
        #                       codes use either 3 or 4 digits.
        geoid_2_gisjoin = 'G'+geoid[:2]+'0'+geoid[2:5]+'0'+geoid[5:]
        acs_tract = acs.loc[acs['GISJOIN']==geoid_2_gisjoin]
        # Centroid of tract 
        lat_tract = tract.centroid.y
        lng_tract = tract.centroid.x
        # Subset latitude, longitude, and attributable fraction maps  
        upper = geo_idx(lat_tract-0.75, lat_af)
        lower = geo_idx(lat_tract+0.75, lat_af)
        left = geo_idx(lng_tract-0.75, lng_af)
        right = geo_idx(lng_tract+0.75, lng_af)
        lat_subset = lat_af[lower:upper]
        lng_subset = lng_af[left:right]
        af_subset = af[lower:upper, left:right]
        # List will be filled with point(s) from attributable fractions grid 
        # inside tract
        af_inside = []
        interpflag = []
        # Fetch coordinates within tracts (if they exist)
        for i, ilat in enumerate(lat_subset):
            for j, jlng in enumerate(lng_subset): 
                point = Point(jlng, ilat)
                if tract.contains(point) is True:
                    # Fill lists with indices in grid within polygon
                    af_inside.append(af_subset[i,j])
                    interpflag.append(0.)
        # Otherwise, interpolate using inverse distance weighting 
        # https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
        if len(af_inside)==0:
            idx_latnear = geo_idx(lat_tract, lat_subset)
            idx_lngnear = geo_idx(lng_tract, lng_subset)
            # Indices for 8 nearby points
            lng_idx = [idx_lngnear-1, idx_lngnear, idx_lngnear+1, 
                idx_lngnear-1, idx_lngnear+1, idx_lngnear-1, idx_lngnear, 
                idx_lngnear+1]
            lat_idx = [idx_latnear+1, idx_latnear+1, idx_latnear+1, 
                idx_latnear, idx_latnear, idx_latnear-1, idx_latnear-1, 
                idx_latnear-1]
            # Known coordinates and attributable fractions
            x = lng_subset[lng_idx]
            y = lat_subset[lat_idx]
            z = af_subset[lat_idx, lng_idx]
            af_inside.append(idwr(x,y,z,[lng_tract], [lat_tract])[0][-1])
            interpflag.append(1.)
        dicttemp = {'GEOID':geoid, 
            'AF':np.nanmean(af_inside),
            'INTERPFLAG':np.nanmean(interpflag),
            'LAT_CENTROID':lat_tract,
            'LNG_CENTROID':lng_tract}
        for var in tokeep+list(crosswalk['code_acs']):
            if var in list(crosswalk['code_acs']):
                var_unified = crosswalk.loc[crosswalk['code_acs']==
                    var]['code_edf'].values[0]
                dicttemp[var_unified] = acs_tract[var].values[0]                    
            else: 
                dicttemp[var] = acs_tract[var].values[0]        
        df.append(dicttemp)   
        del dicttemp
    df = pd.DataFrame(df)
    # Set GEOID as column 
    df = df.set_index('GEOID')
    print('# # # # AF/census harmonized at tract level!', file=f)
        
    # # # # Check to ensure that ACS unified names via crosswalk worked out
    #----------------------
    print('# # # # Checking unified demographic variables from crosswalk...', 
        file=f)
    # Race/ethnicity
    race = (df.filter(like='race_').sum(axis=1)-(2*df['race_tot'])-
        df['race_nh']-df['race_h']-df['race_h_othergt1a']-
        df['race_h_othergt1b']-df['race_nh_othergt1a']-
        df['race_nh_othergt1b'])
    if (np.mean(race)==0.) & (np.max(race)==0.) & (np.min(race)==0.):
        print('racial categories sum to 0!', file=f)
    else: 
        print('racial categories DO NOT sum to 0!', file=f)
    # Housing/vehicle ownership
    housing = (df.filter(like='housing_').sum(axis=1)-(2*df['housing_tot'])-
        df['housing_own']-df['housing_rent'])    
    if (np.mean(housing)==0.) & (np.max(housing)==0.) & (np.min(housing)==0.):
        print('housing categories sum to 0!', file=f)
    else: 
        print('housing categories DO NOT sum to 0!', file=f)
    # Education
    if int(vintage[-4:]) > 2011:
        education = (df.filter(like='education_').sum(axis=1)-
            (2*df['education_tot']))
    else: 
        education = (df.filter(like='education_').sum(axis=1)-
            (2*df['education_tot'])-df['education_m']-df['education_f'])        
    if (np.mean(education)==0.) & (np.max(education)==0.) & \
        (np.min(education)==0.):
        print('education categories sum to 0!', file=f)
    else: 
        print('education categories DO NOT sum to 0!', file=f)
    # Nativity 
    nativity = df['nativity_tot']-df['nativity_foreign']-df['nativity_native']
    if (np.mean(nativity)==0.) & (np.max(nativity)==0.) & \
        (np.min(nativity)==0.):
        print('nativity categories sum to 0!', file=f)
    else: 
        print('nativity categories DO NOT sum to 0!', file=f)
    # Age
    pop = (df.filter(like='pop_').sum(axis=1)-(2*df['pop_tot'])-
        df['pop_m']-df['pop_f'])
    if (np.mean(pop)==0.) & (np.max(pop)==0.) & (np.min(pop)==0.):
        print('population/age categories sum to 0!', file=f)
    else: 
        print('population/age categories DO NOT sum to 0!', file=f)
    
    # Save DataFrame; output files should contain the state FIPS code, AF year,
    # ACS vintage, and version  
    #----------------------
    df = df.replace('NaN', '', regex=True)
    df.to_csv(DIR_OUT+'asthma_af%s_acs%s_%s_v1.csv'%(afyear, vintage, 
        statefips), sep = ',')
    print('# # # # Output file written!\n', file=f)
    return 

vintages = ['2005-2009', '2006-2010', '2007-2011', '2008-2012', 
    '2009-2013', '2010-2014', '2011-2015', '2012-2016', '2013-2017', 
    '2014-2018', '2015-2019']    
fips = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', 
    '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
    '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
    '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', 
    '48', '49', '50', '51', '53', '54', '55', '56', '72']    
# for vintage in vintages: 
    # for statefips in fips: 
        # harmonize_afacs(vintage, statefips)
harmonize_afacs('2011-2015', '11')
f.close()        