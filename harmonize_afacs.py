#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmonize 1 x 1 km NO2 dataset from Anenberg, Mohegh, et al. (2021) with 
census tract geometries and calculate pediatric PAF for states/territories in 
the U.S. and Puerto Rico. 

The version number, whose descriptions/differences are denoted below, are 
indicated in the output harmonized tables' filenames.

v1 - original version
v2 - Calculate PAF using tract-averaged NO2 rather than using a preformed 
     gridded PAF dataset and averaging grid cell PAFs values over tracts.
   - Change the grid for which NO2 is subsampled to the grid of 
     NO2; in the earlier version we had been indexing with the grid of PAF. 
     These two grids have the same dimensions, so no error was thrown, but 
     the latitude coordinates differ slightly. This resulted in slightly off
     NO2 concentrations. 
   - For 2015-2019 vintage, calculate tract-averaged TROPOMI NO2 from 2019

Created on 25 May 2021
"""
__author__ = "Gaige Hunter Kerr"
__maintainer__ = "Kerr"
__email__ = "gaigekerr@gwu.edu"

import math
import time
from datetime import datetime
import numpy as np   

# DIR_ROOT = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'
# DIR_GBD = DIR_ROOT+'gbd/'
# DIR_CENSUS = DIR_ROOT+'acs/'
# DIR_TROPOMI = '/GWSPH/groups/anenberggrp/ghkerr/data/tropomi/'
# DIR_CROSS = DIR_ROOT
# DIR_NO2 = DIR_ROOT+'no2/'
# DIR_GEO = DIR_ROOT+'tigerline/'
# DIR_OUT = DIR_ROOT+'harmonizedtables/'
DIR_TROPOMI = '/Users/ghkerr/GW/data/tropomi/'
DIR_GBD = '/Users/ghkerr/GW/data/gbd/'
DIR_CENSUS = '/Users/ghkerr/GW/data/demographics/'
DIR_CROSS = '/Users/ghkerr/GW/data/demographics/'
DIR_NO2 = '/Users/ghkerr/GW/data/anenberg_mohegh_no2/no2/'
DIR_GEO = '/Users/ghkerr/GW/data/geography/tigerline/'
DIR_OUT = '/Users/ghkerr/Desktop/'

def pixel2coord(col, row, a, b, c, d, e, f):
    """Returns global coordinates to pixel center using base-0 raster 
    index. Adapted from https://gis.stackexchange.com/questions/53617/
    how-to-find-lat-lon-values-for-every-pixel-in-a-geotiff-file"""
    xp = a* col + b * row + a * 0.5 + b * 0.5 + c
    yp = d* col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)

def open_no2pop_tif(fname, fill): 
    """Open TIF dataset for the United States containing 
    and extract coordinate information for the specified domain. 
    
    Parameters
    ----------
    fname : str
        Path to and name of TIF file
    fill : float
        Missing value

    Returns
    -------
    lng : numpy.ndarray
        Longitude array for Larkin dataset, units of degrees, [lng,]
    lat : numpy.ndarray
        Latitude array for Larkin dataset, units of degrees, [lat,]    
    larkin : numpy.ndarray
        Larkin surface-level NO2, units of ppbv, [lat, lng]    
    """
    from osgeo import gdal
    import numpy as np
    ds = gdal.Open('%s.tif'%fname)
    band = ds.GetRasterBand(1)
    no2 = band.ReadAsArray()
    c, a, b, f, d, e = ds.GetGeoTransform()
    col = ds.RasterXSize
    row = ds.RasterYSize
    # Fetch latitudes
    lat = []
    for ri in range(row):
        coord = pixel2coord(0, ri, a, b, c, d, e, f) # Can substitute 
        # whatever for 0 and it should yield the name answer. 
        lat.append(coord[1])
    lat = np.array(lat)
    # Fetch longitudes
    lng = []
    for ci in range(col):
        coord = pixel2coord(ci, 0, a, b, c, d, e, f)
        lng.append(coord[0])
    lng = np.array(lng)
    # Convert from uint8 to float
    no2 = no2.astype(np.float)
    # Replace fill value with NaN
    no2[no2==fill]=np.nan
    return lng, lat, no2

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
    geo_idx = (np.abs(dd_array - dd)).argmin()
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
        State FIPS code (https://www.nrcs.usda.gov/wps/portal/nrcs/detail/
        ?cid=nrcs143_013696)

    Returns
    -------
    None
    """
    import netCDF4 as nc
    import pandas as pd
    import shapefile
    from shapely.geometry import shape, Point
    # For subsetting maps for NO2 harmonization 
    searchrad = 0.75
    
    # # # # Open American Community Survey: 5-Year Data from NHGIS
    #----------------------
    acs2 = pd.read_csv(DIR_CENSUS+'acs%s/'%vintage+'acs%sb.csv'%vintage, 
        sep=',', header=0, engine='python')      
    statename = acs2.loc[acs2['STATEA']==int(statefips)]['STATE'].values[0]
    print('HANDLING %s FOR %s:'%(statename.upper(), vintage[-4:]), file=f)
    print('----------------------------------------', file=f)
    acs1 = pd.read_csv(DIR_CENSUS+'acs%s/'%vintage+'acs%sa.csv'%vintage, 
        sep=',', header=0,  engine='python')     
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
    
    # # # # Open GBD estimates of NO2-attributable pediatric asthma RR and 
    # calculate RR from Khreis et al. (2017) for PAF calculations
    #----------------------
    beta = np.log(1.26)/10.
    betaupper = np.log(1.37)/10.
    betalower = np.log(1.10)/10.
    betagbd = pd.read_csv(DIR_GBD+'no2_rr_draws_summary.csv', sep=',',
        engine='python')
    print('# # # # GBD/Khreis RR calculated!', file=f)    
    
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
    
    # # # # Read NO2
    #----------------------
    no2year = vintage[-4:]
    no2file = DIR_NO2+'%s_final_1km_usa'%no2year
    # Note that the latitude and longitude coordinates should match the 
    # attributable fraction dataset
    lng_no2, lat_no2, no2 = open_no2pop_tif(no2file, -999.)
    print('# # # # NO2 data loaded!', file=f)
    # Put to sleep for a hot sec so it doesn't screw up the progress bar
    time.sleep(2)
    
    
    # # # # Read TROPOMI tropospheric column NO2 for 2015-2019 vintage
    #----------------------
    if vintage == '2015-2019':
        tropomi = nc.Dataset(DIR_TROPOMI+
            'Tropomi_NO2_griddedon0.01grid_2019_QA75.ncf', 'r')
        tropomi = tropomi.variables['NO2'][:].data
        lnglat_tropomi = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf', 'r')
        lng_tropomi = lnglat_tropomi.variables['LON'][:].data
        lat_tropomi = lnglat_tropomi.variables['LAT'][:].data
        print('# # # # TROPOMI NO2 loaded!', file=f)
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
        nestedtract = 0.
        # There are a small number of tracts that don't have data in ACS
        # (for example, see GISJOIN = G0100070010100; GEOID = 01007010100 for 
        # vintage 2005-2009). However, they *appear* to have a number of census 
        # tracts associated with them. Using the example from above, 
        # the following tracks assocated with this GISJOIN code: G0100070010001, 
        # G0100070010002, G0100070010003, G0100070010004
        if acs_tract.shape[0]==0:
            gisjoin_temp = geoid_2_gisjoin[:-4]
            acs_tract = acs.loc[acs['GISJOIN'].str.startswith(gisjoin_temp)]
            nestedtract = 1.
        # Rename columns to unifed names 
        for var in list(crosswalk['code_acs']):
            var_unified = crosswalk.loc[crosswalk['code_acs']==
                var]['code_edf'].values[0]
            acs_tract = acs_tract.rename(columns={var:var_unified})
        # If census tracts shapes associated with >1 ACS entry, sum over 
        # all columns besides Gini and income
        if acs_tract.shape[0]>1: 
            ginitemp = np.nanmean(acs_tract['gini'])
            incometemp = np.nanmean(acs_tract['income'])
            yeartemp = acs_tract['YEAR'].values[0]
            statetemp = acs_tract['STATE'].values[0]
            stateatemp = acs_tract['STATEA'].values[0]
            countyatemp = acs_tract['COUNTYA'].values[0]
            tractatemp = acs_tract['TRACTA'].values[0]
            acs_tract = pd.DataFrame(acs_tract.mean(axis=0)).T
            acs_tract['GISJOIN'] = geoid_2_gisjoin
            acs_tract['YEAR'] = yeartemp
            acs_tract['STATE'] = statetemp
            acs_tract['STATEA'] = stateatemp
            acs_tract['COUNTYA'] = countyatemp
            acs_tract['TRACTA'] = tractatemp
        # Centroid of tract 
        lat_tract = tract.centroid.y
        lng_tract = tract.centroid.x
        # Subset latitude, longitude, and attributable fraction maps  
        upper = geo_idx(lat_tract-searchrad, lat_no2)
        lower = geo_idx(lat_tract+searchrad, lat_no2)
        left = geo_idx(lng_tract-searchrad, lng_no2)
        right = geo_idx(lng_tract+searchrad, lng_no2)
        lat_subset = lat_no2[lower:upper]
        lng_subset = lng_no2[left:right]
        no2_subset = no2[lower:upper, left:right]
        if vintage == '2015-2019':
            uppert = geo_idx(lat_tract-searchrad, lat_tropomi)
            lowert = geo_idx(lat_tract+searchrad, lat_tropomi)
            leftt = geo_idx(lng_tract-searchrad, lng_tropomi)
            rightt = geo_idx(lng_tract+searchrad, lng_tropomi)
            lat_tropomi_subset = lat_tropomi[uppert:lowert]
            lng_tropomi_subset = lng_tropomi[leftt:rightt]            
            tropomi_subset = tropomi[uppert:lowert, leftt:rightt]
            tropomi_inside = []
        # List will be filled with point(s) from attributable fractions and
        # NO2 grid inside tract
        no2_inside = []
        interpflag = []
        # # # # Fetch coordinates within tracts (if they exist) for NO2 dataset
        for i, ilat in enumerate(lat_subset):
            for j, jlng in enumerate(lng_subset): 
                point = Point(jlng, ilat)
                if tract.contains(point) is True:
                    no2_inside.append(no2_subset[i,j])
                    interpflag.append(0.)
        # Otherwise, interpolate using inverse distance weighting 
        # https://rafatieppo.github.io/post/2018_07_27_idw2pyr/
        if len(no2_inside)==0:
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
            z = no2_subset[lat_idx, lng_idx]
            no2_inside.append(idwr(x,y,z,[lng_tract], [lat_tract])[0][-1])
            interpflag.append(1.)
        # # # # Fetch coordinates within tracts for TROPOMI dataset
        if vintage == '2015-2019':            
            for i, ilat in enumerate(lat_tropomi_subset):
                for j, jlng in enumerate(lng_tropomi_subset): 
                    point = Point(jlng, ilat)
                    if tract.contains(point) is True:
                        tropomi_inside.append(tropomi_subset[i,j])
            if len(tropomi_inside)==0:
                idx_latnear = geo_idx(lat_tract, lat_tropomi_subset)
                idx_lngnear = geo_idx(lng_tract, lng_tropomi_subset)
                lng_idx = [idx_lngnear-1, idx_lngnear, idx_lngnear+1, 
                    idx_lngnear-1, idx_lngnear+1, idx_lngnear-1, idx_lngnear, 
                    idx_lngnear+1]
                lat_idx = [idx_latnear+1, idx_latnear+1, idx_latnear+1, 
                    idx_latnear, idx_latnear, idx_latnear-1, idx_latnear-1, 
                    idx_latnear-1]
                x = lng_tropomi_subset[lng_idx]
                y = lat_tropomi_subset[lat_idx]
                z = tropomi_subset[lat_idx, lng_idx]
                tropomi_inside.append(idwr(x,y,z,[lng_tract],
                    [lat_tract])[0][-1])
        # Mean NO2 within tract
        no2_inside = np.nanmean(no2_inside)
        # # # # Calculate attributable fraction based on tract-averaged NO2. 
        # The first method is using the concentration-response factor of 1.26 
        # (1.10 - 1.37) per 10 ppb is used in Achakulwisut et al. (2019) and 
        # taken from Khreis et al. (2017). Note that this "log-linear" 
        # relationship comes from epidemiological studies that log-transform 
        # concentration before regressing with incidence of health outcome 
        # (where log is the natural logarithm). Additional details can be 
        # found in Anenberg et al. (2010)
        af = (1-np.exp(-beta*no2_inside))
        afupper = (1-np.exp(-betaupper*no2_inside))
        aflower = (1-np.exp(-betalower*no2_inside))        
        # The second method is using the GBD RRs. These RR predictions are for 
        # a range of exposures between 0 and 100 ppb. These predictions are 
        # log-transformed, and there are files for 1000 draws and for a 
        # summary only (mean, median, 95% UI bounds). The TMREL used for our 
        # PAFs is a uniform distribution between 4.545 and 6.190 ppb.
        exposureclosest_index = betagbd['exposure'].sub(no2_inside
            ).abs().idxmin()
        # Closest exposure to tract-averaged NO2
        ec = betagbd.iloc[exposureclosest_index]
        afgbdmean = 1-np.exp(-ec['mean'])
        afgbdmed = 1-np.exp(-ec['median'])
        afgbdupper = 1-np.exp(-ec['upper'])
        afgbdlower  = 1-np.exp(-ec['lower'])
        dicttemp = {'GEOID':geoid, 
            'NO2':no2_inside,
            'AF':af,
            'AFUPPER':afupper,
            'AFLOWER':aflower,
            'AFMEAN_GBD':afgbdmean,
            'AFMEDIAN_GBD':afgbdmed,
            'AFUPPER_GBD':afgbdupper,            
            'AFLOWER_GBD':afgbdlower,            
            'INTERPFLAG':np.nanmean(interpflag),
            'NESTEDTRACTFLAG':nestedtract,
            'LAT_CENTROID':lat_tract,
            'LNG_CENTROID':lng_tract}
        if vintage=='2015-2019':
            dicttemp['TROPOMINO2']=np.nanmean(tropomi_inside)
            
        # There are some census tracts records that simply don't have an ACS 
        # entry associated with them (even after correcting for the nesting 
        # issue above). For example, GEOID=02158000100/GISJOIN=G0201580000100 
        # in Alaska has no ACS entry in the 2006-2010 vintage. Set these tracts
        # to NaN and flag
        if acs_tract.shape[0]==0:
            for var in tokeep+list(crosswalk['code_edf']):
                dicttemp[var] = np.nan
            dicttemp['MISSINGTRACTFLAG']=1.
        else: 
            for var in tokeep+list(crosswalk['code_edf']):
                dicttemp[var] = acs_tract[var].values[0]                
            dicttemp['MISSINGTRACTFLAG']=0.    
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
    # Deal with really small differences (e.g., -2e-12)
    race = race.round(decimals=1)
    if (np.mean(race)==0.) & (np.max(race)==0.) & (np.min(race)==0.):
        print('racial categories sum to 0!', file=f)
    else: 
        print('racial categories DO NOT sum to 0!', file=f)
    # Housing/vehicle ownership
    housing = (df.filter(like='housing_').sum(axis=1)-(2*df['housing_tot'])-
        df['housing_own']-df['housing_rent'])
    housing = housing.round(decimals=1)
    if (np.mean(housing)==0.) & (np.max(housing)==0.) & (np.min(housing)==0.):
        print('housing categories sum to 0!', file=f)
    else: 
        print('housing categories DO NOT sum to 0!', file=f)
    # Education
    if int(vintage[-4:]) > 2011:
        education = (df.filter(like='education_').sum(axis=1)-
            (2*df['education_tot']))
        education = education.round(decimals=1)
    else: 
        education = (df.filter(like='education_').sum(axis=1)-
            (2*df['education_tot'])-df['education_m']-df['education_f']) 
        education = education.round(decimals=1)        
    if (np.mean(education)==0.) & (np.max(education)==0.) & \
        (np.min(education)==0.):
        print('education categories sum to 0!', file=f)
    else: 
        print('education categories DO NOT sum to 0!', file=f)
    # Nativity 
    nativity = df['nativity_tot']-df['nativity_foreign']-df['nativity_native']
    nativity = nativity.round(decimals=1)
    if (np.mean(nativity)==0.) & (np.max(nativity)==0.) & \
        (np.min(nativity)==0.):
        print('nativity categories sum to 0!', file=f)
    else: 
        print('nativity categories DO NOT sum to 0!', file=f)
    # Age
    pop = (df.filter(like='pop_').sum(axis=1)-(2*df['pop_tot'])-
        df['pop_m']-df['pop_f'])
    pop = pop.round(decimals=1)
    if (np.mean(pop)==0.) & (np.max(pop)==0.) & (np.min(pop)==0.):
        print('population/age categories sum to 0!', file=f)
    else: 
        print('population/age categories DO NOT sum to 0!', file=f)
    
    # Save DataFrame; output files should contain the state FIPS code, AF year,
    # ACS vintage, and version  
    #----------------------
    df = df.replace('NaN', '', regex=True)
    df.to_csv(DIR_OUT+'asthma_af%s_acs%s_%s_v2.csv'%(no2year, vintage, 
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
for vintage in vintages: 
    # Create output text file for print statements (i.e., crosswalk checks)
    f = open(DIR_OUT+'harmonize_afacs%s_%s.txt'%(vintage,
        datetime.now().strftime('%Y-%m-%d-%H%M')), 'a')    
    for statefips in fips: 
        harmonize_afacs(vintage, statefips)
    f.close()