#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harmonize 1 x 1 km NO2 dataset from Anenberg, Mohegh, et al. (2022) with 
and 1 x 1 km PM25 dataset from van Donkelaar et al. (2021) with census tract 
geometries and calculate PAFs for NO2-attributable pediatric asthma and 
PM25-attributable premature mortality for states/territories in the U.S. and 
Puerto Rico. 

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
v3 - Add PM2.5 and its components to tract-level averaging computation. 
   - Fix summation over tracts that have > 1 ACS entry associated with them. 
   - Fix IDW regression function to handle missing values.
v4 - Add TROPOMI NO2 over Hawaii and Puerto Rico 
   - Calculate PAFs when WHO and EPA targets are met (sensitivity simulation)
v5 - Replace V4.NA.03 PM2.5 with V5.GL.02 PM2.5
   - Get rid of PM2.5 compositional estimates
v6 - Update GBD RR curves to 2020 curves, which no longer have stratification 
     for different ages
   - Add columns for AF calculate from lower and upper bounds of RR curves
   
Original version created on 25 May 2021
"""
__author__ = "Gaige Hunter Kerr"
__maintainer__ = "Kerr"
__email__ = "gaigekerr@gwu.edu"

import math
import time
from datetime import datetime
import numpy as np   

DIR_ROOT = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'
DIR_GBD = DIR_ROOT+'gbd/'
DIR_CENSUS = DIR_ROOT+'acs/'
DIR_TROPOMI = '/GWSPH/groups/anenberggrp/ghkerr/data/tropomi/'
DIR_CROSS = DIR_ROOT
DIR_NO2 = DIR_ROOT+'no2/'
DIR_PM25 = '/GWSPH/groups/anenberggrp/ghkerr/data/pm25/PM25/'
DIR_FIG = DIR_ROOT
DIR_GEO = DIR_ROOT+'tigerline/'
DIR_OUT = DIR_ROOT+'harmonizedtables/'

# DIR_ROOT = '/Users/ghkerr/GW/data/edf/'
# DIR_TROPOMI = '/Users/ghkerr/GW/data/tropomi/'
# DIR_GBD = '/Users/ghkerr/GW/data/gbd/'
# DIR_CENSUS = '/Users/ghkerr/GW/data/demographics/'
# DIR_CROSS = '/Users/ghkerr/GW/data/demographics/'
# DIR_NO2 = '/Users/ghkerr/GW/data/anenberg_mohegh_no2/no2/'
# DIR_PM25 = '/Users/ghkerr/Downloads/'
# DIR_GEO = '/Users/ghkerr/GW/data/geography/tigerline/'
# DIR_OUT = '/Users/ghkerr/Desktop/'

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
    """Inverse distance weighting for interpolating gridded fields to census 
    tracts that are too small to intersect with the grid. 

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
        value for small census tract 
    """
    lstxyzi = []
    for p in range(len(xi)):
        lstdist = []
        for s in range(len(x)):
            d = (harvesine(x[s], y[s], xi[p], yi[p]))
            lstdist.append(d)
        sumsup = list((1 / np.power(lstdist, 2)))
        suminf = np.nansum(sumsup)
        # The original configuration of this function had the following 
        # line of code
        # sumsup = np.nansum(np.array(sumsup) * np.array(z))
        # However, there were issues with arrays with missing data and NaNs, 
        # so it was changed to the following: 
        sumsup = np.nansum(np.array(sumsup) * np.array(z))
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
    # calculate RR from Khreis et al. (2017) for PAF calculations for NO2
    #----------------------
    beta = np.log(1.26)/10.
    betaupper = np.log(1.37)/10.
    betalower = np.log(1.10)/10.
    betagbd = pd.read_csv(DIR_GBD+'no2_rr_draws_summary.csv', sep=',',
        engine='python')
    # Load PM2.5-attributable relative risk for various health endpoints from 
    # the GBD meta-regression with Bayesian priors, Regularization and Trimming
    # (MR-BRT) files. Note that the key to interpret file names can be found
    # here: 
    # http://ghdx.healthdata.org/sites/default/files/record-attached-files/
    # IHME_GBD_2019_PM_RISK_INFO_SHEET_Y2021M01D06.PDF
    # RR for PM2.5-attributable ischemic heart disease
    rrpmihd = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd.csv')
    # RR for PM2.5-attributable stroke
    rrpmst = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke.csv')
    # RR for PM2.5-attributable chronic obstructive pulmonary disease
    rrpmcopd = pd.read_csv(DIR_GBD+'mrbrt_summary/resp_copd.csv')
    # RR for PM2.5-attributable tracheal, bronchus, and lung cancer
    rrpmlc = pd.read_csv(DIR_GBD+'mrbrt_summary/neo_lung.csv')
    # RR for PM2.5-attributable diabetes mellitus type 2
    rrpmdm = pd.read_csv(DIR_GBD+'mrbrt_summary/t2_dm.csv')
    # RR for PM2.5 attributable lower respiratory infections
    rrpmlri = pd.read_csv(DIR_GBD+'mrbrt_summary/lri.csv')
    
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
    
    # # # # Read PM25
    #----------------------
    pm25 = nc.Dataset(DIR_PM25+
        'V5GL02.HybridPM25.Global.%s01-%s12.nc'%(no2year,no2year), 'r')
    lat_pm = pm25.variables['lat'][:]
    lng_pm = pm25.variables['lon'][:]
    total = pm25.variables['GWRPM25'][:]
    print('# # # # PM2.5 data loaded!', file=f)
    time.sleep(2)
    
    # # # # Read TROPOMI tropospheric column NO2 for 2015-2019 vintage
    # note that if 
    #----------------------
    if vintage == '2015-2019':
        # Hawaii
        if statefips=='15':
            tropomi = nc.Dataset(DIR_TROPOMI+
                'Tropomi_NO2_griddedon0.01x0.01grid_Hawaii_2019_QA0.75.ncf', 
                'r')
            tropomi = tropomi.variables['NO2'][:].data
            lnglat_tropomi = nc.Dataset(DIR_TROPOMI+'LatLonGrid_Hawaii.ncf', 
                'r')
            lng_tropomi = lnglat_tropomi.variables['LON'][:].data
            lat_tropomi = lnglat_tropomi.variables['LAT'][:].data
            print('# # # # TROPOMI NO2 loaded!', file=f)         
        # Puerto Rico
        elif statefips=='72':
            tropomi = nc.Dataset(DIR_TROPOMI+
                'Tropomi_NO2_griddedon0.01x0.01grid_PuertoRico_2019_QA0.75.ncf', 
                'r')
            tropomi = tropomi.variables['NO2'][:].data
            lnglat_tropomi = nc.Dataset(DIR_TROPOMI+'LatLonGrid_PuertoRico.ncf', 
                'r')
            lng_tropomi = lnglat_tropomi.variables['LON'][:].data
            lat_tropomi = lnglat_tropomi.variables['LAT'][:].data
            print('# # # # TROPOMI NO2 loaded!', file=f)
        # All other states
        else: 
            tropomi = nc.Dataset(DIR_TROPOMI+
                'Tropomi_NO2_griddedon0.01grid_2019_QA75.ncf', 'r')
            tropomi = tropomi.variables['NO2'][:].data
            lnglat_tropomi = nc.Dataset(DIR_TROPOMI+'LatLonGrid.ncf', 'r')
            lng_tropomi = lnglat_tropomi.variables['LON'][:].data
            lat_tropomi = lnglat_tropomi.variables['LAT'][:].data
            print('# # # # TROPOMI NO2 loaded!', file=f)
        time.sleep(2)
    
    # # # # Loop through tracts and find attributable fractions in each tract
    # and demographic information
    #----------------------
    df = []
    for tract in np.arange(0, len(tracts), 1):
        record = records[tract]  
        # In older iterations of this script, I employed a 0.75˚ search radius
        # to find intersecting netCDF grid cells within the tract. This 
        # approach might not be perfect for very large tracts, such as those 
        # in the Western U.S.; however, it is very slow. The new approach 
        # defines a different search radius for each tract based on its total 
        # area (in meters squared) of land and water
        area = (record['ALAND']+record['AWATER'])/(1000*1000) # convert to km2
        # Translate square kilometers to the length of each tract boundary 
        # (assuming a square tract) 
        searchrad = np.sqrt(area)
        # Convert side (km) to degrees, assuming 1 deg = 110 km; and create
        # buffer (20x the side length of tract) to account for irregularly-
        # shaped tracts
        searchrad = (searchrad/110.) * 15.
        # For incredibly small block groups, this side approach is still too
        # small to pick up the 8 surrounding grid cells for inverse distance
        # weighting. In this case, artificially force side to be ~1 km
        if searchrad < 0.05:
            searchrad = 0.05
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
            acs_tract = pd.DataFrame(acs_tract.sum(axis=0)).T
            acs_tract['GISJOIN'] = geoid_2_gisjoin
            acs_tract['gini'] = ginitemp
            acs_tract['income'] = incometemp 
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
        # If there are no intersecting grid cells centers with the tract, 
        # interpolate using inverse distance weighting; adapted from
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
        # If there are issues/questions about the interpolation, examining 
        # the following chunk of code (denoted with ~***~) for a favorite
        # census tract might shed some light on the issue
        # ~***~
        # plt.pcolormesh(lng_subset, lat_subset, no2_subset, vmin=6, vmax=15, 
        #     shading='nearest', edgecolors='k')
        # plt.colorbar()
        # plt.xlim([x0, x1])
        # plt.ylim([y0, y1])
        # plt.scatter(lng_tract, lat_tract, c='k')
        # plt.plot(tract.boundary.xy[0], tract.boundary.xy[1], 'ko')
        # for i, ilat in enumerate(lat_subset):
        #     for j, jlng in enumerate(lng_subset): 
        #         point = Point(jlng, ilat)
        #         plt.plot(point.xy[0], point.xy[1], 'ro') 
        #         if tract.contains(point) is True:
        #             no2_inside.append(no2_subset[i,j])
        #             interpflag.append(0.)
        #             plt.plot(point.xy[0], point.xy[1], 'r*')                    
        # ~***~                    
        # # # # NO2 health impact assessment
        no2_inside = np.nanmean(no2_inside)
        if np.isnan(no2_inside)!=True:
            # For sensitivity simulations: if NO2 > attainment thresholds, 
            # force to comply with threshold. 
            # WHO Recommendation NO2
            # Interim target 40 ug/m3 --> 21.276595 ppbv
            # Interim target 30 ug/m3 --> 15.95744680 ppbv
            # Interim target 20 ug/m3 --> 10.63829787 ppbv
            # AQG level	10 ug/m3 --> 5.319148936 ppbv
            no2_insidewho40 = no2_inside
            no2_insidewho30 = no2_inside
            no2_insidewho20 = no2_inside
            no2_insidewho10 = no2_inside
            if no2_insidewho40 > 21.276595:
                no2_insidewho40 = 21.276595
            if no2_insidewho30 > 15.95744680:
                no2_insidewho30 = 15.95744680
            if no2_insidewho20 > 10.63829787:
                no2_insidewho20 = 10.63829787
            if no2_insidewho10 > 5.319148936:
                no2_insidewho10 = 5.319148936                
            # Calculate attributable fraction based on tract-averaged NO2 
            # using the GBD RRs. These RR predictions are for a range of 
            # exposures between 0 and 100 ppb. These predictions are 
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
            exposureclosest_index = betagbd['exposure'].sub(no2_insidewho40
                ).abs().idxmin()
            afgbdmeanwho40 = 1-np.exp(-ec['mean'])
            afgbdmedwho40 = 1-np.exp(-ec['median'])
            afgbdupperwho40 = 1-np.exp(-ec['upper'])
            afgbdlowerwho40  = 1-np.exp(-ec['lower'])
            exposureclosest_index = betagbd['exposure'].sub(no2_insidewho30
                ).abs().idxmin()
            ec = betagbd.iloc[exposureclosest_index]
            afgbdmeanwho30 = 1-np.exp(-ec['mean'])
            afgbdmedwho30 = 1-np.exp(-ec['median'])
            afgbdupperwho30 = 1-np.exp(-ec['upper'])
            afgbdlowerwho30  = 1-np.exp(-ec['lower'])
            exposureclosest_index = betagbd['exposure'].sub(no2_insidewho20
                ).abs().idxmin()
            ec = betagbd.iloc[exposureclosest_index]
            afgbdmeanwho20 = 1-np.exp(-ec['mean'])
            afgbdmedwho20 = 1-np.exp(-ec['median'])
            afgbdupperwho20 = 1-np.exp(-ec['upper'])
            afgbdlowerwho20  = 1-np.exp(-ec['lower'])    
            exposureclosest_index = betagbd['exposure'].sub(no2_insidewho10
                ).abs().idxmin()
            ec = betagbd.iloc[exposureclosest_index]
            afgbdmeanwho10 = 1-np.exp(-ec['mean'])
            afgbdmedwho10 = 1-np.exp(-ec['median'])
            afgbdupperwho10 = 1-np.exp(-ec['upper'])
            afgbdlowerwho10  = 1-np.exp(-ec['lower'])            
        else: 
            no2_insidewho40 = np.nan
            no2_insidewho30 = np.nan
            no2_insidewho20 = np.nan
            no2_insidewho10 = np.nan
            afgbdmean = np.nan
            afgbdmed = np.nan
            afgbdupper = np.nan
            afgbdlower  = np.nan
            afgbdmeanwho40 = np.nan
            afgbdmedwho40 = np.nan
            afgbdupperwho40 = np.nan
            afgbdlowerwho40  = np.nan
            afgbdmeanwho30 = np.nan
            afgbdmedwho30 = np.nan
            afgbdupperwho30 = np.nan
            afgbdlowerwho30  = np.nan
            afgbdmeanwho20 = np.nan
            afgbdmedwho20 = np.nan
            afgbdupperwho20 = np.nan
            afgbdlowerwho20  = np.nan
            afgbdmeanwho10 = np.nan
            afgbdmedwho10 = np.nan
            afgbdupperwho10 = np.nan
            afgbdlowerwho10  = np.nan            
        dicttemp = {'GEOID':geoid, 
            'NO2':no2_inside,
            'NO2WHO40':no2_insidewho40,
            'NO2WHO30':no2_insidewho30,
            'NO2WHO20':no2_insidewho20,
            'NO2WHO10':no2_insidewho10, 
            'AFPAMEAN_GBD':afgbdmean,
            'AFPAMEDIAN_GBD':afgbdmed,
            'AFPAUPPER_GBD':afgbdupper,            
            'AFPALOWER_GBD':afgbdlower,         
            'AFPAMEAN_GBDWHO40':afgbdmeanwho40,
            'AFPAMEDIAN_GBDWHO40':afgbdmedwho40,
            'AFPAUPPER_GBDWHO40':afgbdupperwho40,            
            'AFPALOWER_GBDWHO40':afgbdlowerwho40,         
            'AFPAMEAN_GBDWHO30':afgbdmeanwho30,
            'AFPAMEDIAN_GBDWHO30':afgbdmedwho30,
            'AFPAUPPER_GBDWHO30':afgbdupperwho30,            
            'AFPALOWER_GBDWHO30':afgbdlowerwho30,         
            'AFPAMEAN_GBDWHO20':afgbdmeanwho20,
            'AFPAMEDIAN_GBDWHO20':afgbdmedwho20,
            'AFPAUPPER_GBDWHO20':afgbdupperwho20,            
            'AFPALOWER_GBDWHO20':afgbdlowerwho20,         
            'AFPAMEAN_GBDWHO10':afgbdmeanwho10,
            'AFPAMEDIAN_GBDWHO10':afgbdmedwho10,
            'AFPAUPPER_GBDWHO10':afgbdupperwho10,            
            'AFPALOWER_GBDWHO10':afgbdlowerwho10,                     
            'INTERPFLAG':np.nanmean(interpflag),
            'NESTEDTRACTFLAG':nestedtract,
            'LAT_CENTROID':lat_tract,
            'LNG_CENTROID':lng_tract}
        # # # # Fetch tract-level PM2.5 estimates
        # Subset
        upperp = geo_idx(lat_tract+searchrad, lat_pm)
        lowerp = geo_idx(lat_tract-searchrad, lat_pm)
        leftp = geo_idx(lng_tract-searchrad, lng_pm)
        rightp = geo_idx(lng_tract+searchrad, lng_pm)
        lat_pm_subset = lat_pm[lowerp:upperp]
        lng_pm_subset = lng_pm[leftp:rightp] 
        pm_subset = total[lowerp:upperp, leftp:rightp]
        # Find PM25 in tracts
        pm_inside = []
        pminterpflag = []
        for i, ilat in enumerate(lat_pm_subset):
            for j, jlng in enumerate(lng_pm_subset): 
                point = Point(jlng, ilat)
                if tract.contains(point) is True:
                    pm_inside.append(pm_subset[i,j])
                    pminterpflag.append(0.)
        if len(pm_inside)==0:
            if (statefips!='02') and (statefips!='15') and (statefips!='72'): 
                idx_latnear = geo_idx(lat_tract, lat_pm_subset)
                idx_lngnear = geo_idx(lng_tract, lng_pm_subset)
                lng_idx = [idx_lngnear-1, idx_lngnear, idx_lngnear+1, 
                    idx_lngnear-1, idx_lngnear+1, idx_lngnear-1, idx_lngnear, 
                    idx_lngnear+1]
                lat_idx = [idx_latnear+1, idx_latnear+1, idx_latnear+1, 
                    idx_latnear, idx_latnear, idx_latnear-1, idx_latnear-1, 
                    idx_latnear-1]
                x = lng_pm_subset[lng_idx]
                y = lat_pm_subset[lat_idx]
                z = pm_subset[lat_idx, lng_idx]
                pm_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                pminterpflag.append(1.)
        # # # # PM2.5 health impact assessment                    
        pm_inside = np.nanmean(pm_inside)
        if np.isnan(pm_inside)!=True:
            # For sensitivity simulations; 
            # WHO/NAAQS Recommendation PM25
            # Interim target 15 ug/m3
            # NAAQS 12 ug/m3
            # Interim target 10 ug/m3
            # AQG levels 5 ug/m3
            pm_insidewho15 = pm_inside
            pm_insidenaaqs12 = pm_inside 
            pm_insidewho10 = pm_inside
            pm_insidewho5 = pm_inside       
            if pm_insidewho15 > 15.:
                pm_insidewho15 = 15.
            if pm_insidenaaqs12 > 12.:
                pm_insidenaaqs12 = 12.
            if pm_insidewho10 > 10.:
                pm_insidewho10 = 10.
            if pm_insidewho5 > 5.:
                pm_insidewho5 = 5.                
            # Length of all GBD PM2.5-attributable RR is the same, so 
            # this should work 
            ci = rrpmst['exposure'].sub(pm_inside).abs().idxmin()
            # Closest exposure to tract-averaged PM25 for mean, lower, and 
            # upper estimates
            cirrpmihd = rrpmihd.iloc[ci]['mean']
            cirrpmst = rrpmst.iloc[ci]['mean']
            cirrpmcopd = rrpmcopd.iloc[ci]['mean']
            cirrpmlc = rrpmlc.iloc[ci]['mean']
            cirrpmdm = rrpmdm.iloc[ci]['mean']
            cirrpmlri = rrpmlri.iloc[ci]['mean']
            cirrpmihdlower = rrpmihd.iloc[ci]['lower']
            cirrpmstlower = rrpmst.iloc[ci]['lower']
            cirrpmcopdlower = rrpmcopd.iloc[ci]['lower']
            cirrpmlclower = rrpmlc.iloc[ci]['lower']
            cirrpmdmlower = rrpmdm.iloc[ci]['lower']
            cirrpmlrilower = rrpmlri.iloc[ci]['lower']
            cirrpmihdupper = rrpmihd.iloc[ci]['upper']
            cirrpmstupper = rrpmst.iloc[ci]['upper']
            cirrpmcopdupper = rrpmcopd.iloc[ci]['upper']
            cirrpmlcupper = rrpmlc.iloc[ci]['upper']
            cirrpmdmupper = rrpmdm.iloc[ci]['upper']
            cirrpmlriupper = rrpmlri.iloc[ci]['upper']            
            # Calculate PAF as (RR-1)/RR for mean, lower, and upper estimates
            afpmihd = (cirrpmihd-1.)/cirrpmihd
            afpmst = (cirrpmst-1.)/cirrpmst
            afpmcopd = (cirrpmcopd-1.)/cirrpmcopd
            afpmlc = (cirrpmlc-1.)/cirrpmlc
            afpmdm = (cirrpmdm-1.)/cirrpmdm
            afpmlri = (cirrpmlri-1.)/cirrpmlri 
            afpmihdlower = (cirrpmihdlower-1.)/cirrpmihdlower
            afpmstlower = (cirrpmstlower-1.)/cirrpmstlower
            afpmcopdlower = (cirrpmcopdlower-1.)/cirrpmcopdlower
            afpmlclower = (cirrpmlclower-1.)/cirrpmlclower
            afpmdmlower = (cirrpmdmlower-1.)/cirrpmdmlower
            afpmlrilower = (cirrpmlrilower-1.)/cirrpmlrilower
            afpmihdupper = (cirrpmihdupper-1.)/cirrpmihdupper
            afpmstupper = (cirrpmstupper-1.)/cirrpmstupper
            afpmcopdupper = (cirrpmcopdupper-1.)/cirrpmcopdupper
            afpmlcupper = (cirrpmlcupper-1.)/cirrpmlcupper
            afpmdmupper = (cirrpmdmupper-1.)/cirrpmdmupper
            afpmlriupper = (cirrpmlriupper-1.)/cirrpmlriupper             
            # WHO Interim 3
            ciwho15 = rrpmst['exposure'].sub(pm_insidewho15).abs().idxmin()
            cirrpmihd = rrpmihd.iloc[ciwho15]['mean']
            cirrpmst = rrpmst.iloc[ciwho15]['mean']
            cirrpmcopd = rrpmcopd.iloc[ciwho15]['mean']
            cirrpmlc = rrpmlc.iloc[ciwho15]['mean']
            cirrpmdm = rrpmdm.iloc[ciwho15]['mean']
            cirrpmlri = rrpmlri.iloc[ciwho15]['mean']
            cirrpmihdlower = rrpmihd.iloc[ciwho15]['lower']
            cirrpmstlower = rrpmst.iloc[ciwho15]['lower']
            cirrpmcopdlower = rrpmcopd.iloc[ciwho15]['lower']
            cirrpmlclower = rrpmlc.iloc[ciwho15]['lower']
            cirrpmdmlower = rrpmdm.iloc[ciwho15]['lower']
            cirrpmlrilower = rrpmlri.iloc[ciwho15]['lower']
            cirrpmihdupper = rrpmihd.iloc[ciwho15]['upper']
            cirrpmstupper = rrpmst.iloc[ciwho15]['upper']
            cirrpmcopdupper = rrpmcopd.iloc[ciwho15]['upper']
            cirrpmlcupper = rrpmlc.iloc[ciwho15]['upper']
            cirrpmdmupper = rrpmdm.iloc[ciwho15]['upper']
            cirrpmlriupper = rrpmlri.iloc[ciwho15]['upper']            
            afpmihdwho15 = (cirrpmihd-1.)/cirrpmihd
            afpmstwho15 = (cirrpmst-1.)/cirrpmst
            afpmcopdwho15 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcwho15 = (cirrpmlc-1.)/cirrpmlc
            afpmdmwho15 = (cirrpmdm-1.)/cirrpmdm
            afpmlriwho15 = (cirrpmlri-1.)/cirrpmlri
            afpmihdwho15lower = (cirrpmihdlower-1.)/cirrpmihdlower
            afpmstwho15lower = (cirrpmstlower-1.)/cirrpmstlower
            afpmcopdwho15lower = (cirrpmcopdlower-1.)/cirrpmcopdlower
            afpmlcwho15lower = (cirrpmlclower-1.)/cirrpmlclower
            afpmdmwho15lower = (cirrpmdmlower-1.)/cirrpmdmlower
            afpmlriwho15lower = (cirrpmlrilower-1.)/cirrpmlrilower             
            afpmihdwho15upper = (cirrpmihdupper-1.)/cirrpmihdupper
            afpmstwho15upper = (cirrpmstupper-1.)/cirrpmstupper
            afpmcopdwho15upper = (cirrpmcopdupper-1.)/cirrpmcopdupper
            afpmlcwho15upper = (cirrpmlcupper-1.)/cirrpmlcupper
            afpmdmwho15upper = (cirrpmdmupper-1.)/cirrpmdmupper
            afpmlriwho15upper = (cirrpmlriupper-1.)/cirrpmlriupper 
            # EPA NAAQS
            cinaaqs12 = rrpmst['exposure'].sub(pm_insidenaaqs12).abs().idxmin()
            cirrpmihd = rrpmihd.iloc[cinaaqs12]['mean']
            cirrpmst = rrpmst.iloc[cinaaqs12]['mean']
            cirrpmcopd = rrpmcopd.iloc[cinaaqs12]['mean']
            cirrpmlc = rrpmlc.iloc[cinaaqs12]['mean']
            cirrpmdm = rrpmdm.iloc[cinaaqs12]['mean']
            cirrpmlri = rrpmlri.iloc[cinaaqs12]['mean']
            cirrpmihdlower = rrpmihd.iloc[cinaaqs12]['lower']
            cirrpmstlower = rrpmst.iloc[cinaaqs12]['lower']
            cirrpmcopdlower = rrpmcopd.iloc[cinaaqs12]['lower']
            cirrpmlclower = rrpmlc.iloc[cinaaqs12]['lower']
            cirrpmdmlower = rrpmdm.iloc[cinaaqs12]['lower']
            cirrpmlrilower = rrpmlri.iloc[cinaaqs12]['lower']
            cirrpmihdupper = rrpmihd.iloc[cinaaqs12]['upper']
            cirrpmstupper = rrpmst.iloc[cinaaqs12]['upper']
            cirrpmcopdupper = rrpmcopd.iloc[cinaaqs12]['upper']
            cirrpmlcupper = rrpmlc.iloc[cinaaqs12]['upper']
            cirrpmdmupper = rrpmdm.iloc[cinaaqs12]['upper']
            cirrpmlriupper = rrpmlri.iloc[cinaaqs12]['upper']
            afpmihdnaaqs12 = (cirrpmihd-1.)/cirrpmihd
            afpmstnaaqs12 = (cirrpmst-1.)/cirrpmst
            afpmcopdnaaqs12 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcnaaqs12 = (cirrpmlc-1.)/cirrpmlc
            afpmdmnaaqs12 = (cirrpmdm-1.)/cirrpmdm
            afpmlrinaaqs12 = (cirrpmlri-1.)/cirrpmlri
            afpmihdnaaqs12lower = (cirrpmihdlower-1.)/cirrpmihdlower
            afpmstnaaqs12lower = (cirrpmstlower-1.)/cirrpmstlower
            afpmcopdnaaqs12lower = (cirrpmcopdlower-1.)/cirrpmcopdlower
            afpmlcnaaqs12lower = (cirrpmlclower-1.)/cirrpmlclower
            afpmdmnaaqs12lower = (cirrpmdmlower-1.)/cirrpmdmlower
            afpmlrinaaqs12lower = (cirrpmlrilower-1.)/cirrpmlrilower
            afpmihdnaaqs12upper = (cirrpmihdupper-1.)/cirrpmihdupper
            afpmstnaaqs12upper = (cirrpmstupper-1.)/cirrpmstupper
            afpmcopdnaaqs12upper = (cirrpmcopdupper-1.)/cirrpmcopdupper
            afpmlcnaaqs12upper = (cirrpmlcupper-1.)/cirrpmlcupper
            afpmdmnaaqs12upper = (cirrpmdmupper-1.)/cirrpmdmupper
            afpmlrinaaqs12upper = (cirrpmlriupper-1.)/cirrpmlriupper
            # WHO Interim 4                
            ciwho10 = rrpmst['exposure'].sub(pm_insidewho10).abs().idxmin()
            cirrpmihd = rrpmihd.iloc[ciwho10]['mean']
            cirrpmst = rrpmst.iloc[ciwho10]['mean']
            cirrpmcopd = rrpmcopd.iloc[ciwho10]['mean']
            cirrpmlc = rrpmlc.iloc[ciwho10]['mean']
            cirrpmdm = rrpmdm.iloc[ciwho10]['mean']
            cirrpmlri = rrpmlri.iloc[ciwho10]['mean']
            cirrpmihdlower = rrpmihd.iloc[ciwho10]['lower']
            cirrpmstlower = rrpmst.iloc[ciwho10]['lower']
            cirrpmcopdlower = rrpmcopd.iloc[ciwho10]['lower']
            cirrpmlclower = rrpmlc.iloc[ciwho10]['lower']
            cirrpmdmlower = rrpmdm.iloc[ciwho10]['lower']
            cirrpmlrilower = rrpmlri.iloc[ciwho10]['lower']
            cirrpmihdupper = rrpmihd.iloc[ciwho10]['upper']
            cirrpmstupper = rrpmst.iloc[ciwho10]['upper']
            cirrpmcopdupper = rrpmcopd.iloc[ciwho10]['upper']
            cirrpmlcupper = rrpmlc.iloc[ciwho10]['upper']
            cirrpmdmupper = rrpmdm.iloc[ciwho10]['upper']
            cirrpmlriupper = rrpmlri.iloc[ciwho10]['upper']            
            afpmihdwho10 = (cirrpmihd-1.)/cirrpmihd
            afpmstwho10 = (cirrpmst-1.)/cirrpmst
            afpmcopdwho10 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcwho10 = (cirrpmlc-1.)/cirrpmlc
            afpmdmwho10 = (cirrpmdm-1.)/cirrpmdm
            afpmlriwho10 = (cirrpmlri-1.)/cirrpmlri
            afpmihdwho10lower = (cirrpmihdlower-1.)/cirrpmihdlower
            afpmstwho10lower = (cirrpmstlower-1.)/cirrpmstlower
            afpmcopdwho10lower = (cirrpmcopdlower-1.)/cirrpmcopdlower
            afpmlcwho10lower = (cirrpmlclower-1.)/cirrpmlclower
            afpmdmwho10lower = (cirrpmdmlower-1.)/cirrpmdmlower
            afpmlriwho10lower = (cirrpmlrilower-1.)/cirrpmlrilower
            afpmihdwho10upper = (cirrpmihdupper-1.)/cirrpmihdupper
            afpmstwho10upper = (cirrpmstupper-1.)/cirrpmstupper
            afpmcopdwho10upper = (cirrpmcopdupper-1.)/cirrpmcopdupper
            afpmlcwho10upper = (cirrpmlcupper-1.)/cirrpmlcupper
            afpmdmwho10upper = (cirrpmdmupper-1.)/cirrpmdmupper
            afpmlriwho10upper = (cirrpmlriupper-1.)/cirrpmlriupper
            # WHO AQG
            ciwho5 = rrpmst['exposure'].sub(pm_insidewho5).abs().idxmin()
            cirrpmihd = rrpmihd.iloc[ciwho5]['mean']
            cirrpmst = rrpmst.iloc[ciwho5]['mean']
            cirrpmcopd = rrpmcopd.iloc[ciwho5]['mean']
            cirrpmlc = rrpmlc.iloc[ciwho5]['mean']
            cirrpmdm = rrpmdm.iloc[ciwho5]['mean']
            cirrpmlri = rrpmlri.iloc[ciwho5]['mean']
            cirrpmihdlower = rrpmihd.iloc[ciwho5]['lower']
            cirrpmstlower = rrpmst.iloc[ciwho5]['lower']
            cirrpmcopdlower = rrpmcopd.iloc[ciwho5]['lower']
            cirrpmlclower = rrpmlc.iloc[ciwho5]['lower']
            cirrpmdmlower = rrpmdm.iloc[ciwho5]['lower']
            cirrpmlrilower = rrpmlri.iloc[ciwho5]['lower']
            cirrpmihdupper = rrpmihd.iloc[ciwho5]['upper']
            cirrpmstupper = rrpmst.iloc[ciwho5]['upper']
            cirrpmcopdupper = rrpmcopd.iloc[ciwho5]['upper']
            cirrpmlcupper = rrpmlc.iloc[ciwho5]['upper']
            cirrpmdmupper = rrpmdm.iloc[ciwho5]['upper']
            cirrpmlriupper = rrpmlri.iloc[ciwho5]['upper']  
            afpmihdwho5 = (cirrpmihd-1.)/cirrpmihd
            afpmstwho5 = (cirrpmst-1.)/cirrpmst
            afpmcopdwho5 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcwho5 = (cirrpmlc-1.)/cirrpmlc
            afpmdmwho5 = (cirrpmdm-1.)/cirrpmdm
            afpmlriwho5 = (cirrpmlri-1.)/cirrpmlri
            afpmihdwho5lower = (cirrpmihdlower-1.)/cirrpmihdlower
            afpmstwho5lower = (cirrpmstlower-1.)/cirrpmstlower
            afpmcopdwho5lower = (cirrpmcopdlower-1.)/cirrpmcopdlower
            afpmlcwho5lower = (cirrpmlclower-1.)/cirrpmlclower
            afpmdmwho5lower = (cirrpmdmlower-1.)/cirrpmdmlower
            afpmlriwho5lower = (cirrpmlrilower-1.)/cirrpmlrilower  
            afpmihdwho5upper = (cirrpmihdupper-1.)/cirrpmihdupper
            afpmstwho5upper = (cirrpmstupper-1.)/cirrpmstupper
            afpmcopdwho5upper = (cirrpmcopdupper-1.)/cirrpmcopdupper
            afpmlcwho5upper = (cirrpmlcupper-1.)/cirrpmlcupper
            afpmdmwho5upper = (cirrpmdmupper-1.)/cirrpmdmupper
            afpmlriwho5upper = (cirrpmlriupper-1.)/cirrpmlriupper              
        else:
            pm_insidewho15, pm_insidenaaqs12 = np.nan, np.nan
            pm_insidewho10, pm_insidewho5 = np.nan, np.nan
            afpmihd, afpmst = np.nan, np.nan
            afpmcopd, afpmlc, afpmdm, afpmlri = np.nan, np.nan, np.nan, np.nan
            afpmihdwho15, afpmstwho15 = np.nan, np.nan
            afpmcopdwho15, afpmlcwho15 = np.nan, np.nan
            afpmdmwho15, afpmlriwho15 = np.nan, np.nan
            afpmihdnaaqs12, afpmstnaaqs12 = np.nan, np.nan
            afpmcopdnaaqs12, afpmlcnaaqs12 = np.nan, np.nan
            afpmdmnaaqs12, afpmlrinaaqs12 = np.nan, np.nan                
            afpmihdwho10, afpmstwho10 = np.nan, np.nan
            afpmcopdwho10, afpmlcwho10 = np.nan, np.nan
            afpmdmwho10, afpmlriwho10 = np.nan, np.nan
            afpmihdwho5, afpmstwho5 = np.nan, np.nan
            afpmcopdwho5, afpmlcwho5 = np.nan, np.nan
            afpmdmwho5, afpmlriwho5 = np.nan, np.nan                
        dicttemp['PM25'] = pm_inside
        dicttemp['PM25WHO15'] = pm_insidewho15
        dicttemp['PM25NAAQS12'] = pm_insidenaaqs12
        dicttemp['PM25WHO10'] = pm_insidewho10
        dicttemp['PM25WHO5'] = pm_insidewho5
        dicttemp['AFIHD'] = afpmihd
        dicttemp['AFIHDLOWER'] = afpmihdlower
        dicttemp['AFIHDUPPER'] = afpmihdupper
        dicttemp['AFST'] = afpmst
        dicttemp['AFSTLOWER'] = afpmstlower
        dicttemp['AFSTUPPER'] = afpmstupper  
        dicttemp['AFCOPD'] = afpmcopd
        dicttemp['AFCOPDLOWER'] = afpmcopdlower
        dicttemp['AFCOPDUPPER'] = afpmcopdupper
        dicttemp['AFLC'] = afpmlc
        dicttemp['AFLCLOWER'] = afpmlclower
        dicttemp['AFLCUPPER'] = afpmlcupper
        dicttemp['AFDM'] = afpmdm
        dicttemp['AFDMLOWER'] = afpmdmlower
        dicttemp['AFDMUPPER'] = afpmdmupper
        dicttemp['AFLRI'] = afpmlri  
        dicttemp['AFLRILOWER'] = afpmlrilower
        dicttemp['AFLRIUPPER'] = afpmlriupper
        dicttemp['AFIHDWHO15'] = afpmihdwho15
        dicttemp['AFIHDWHO15LOWER'] = afpmihdwho15lower
        dicttemp['AFIHDWHO15UPPER'] = afpmihdwho15upper
        dicttemp['AFSTWHO15'] = afpmstwho15
        dicttemp['AFSTWHO15LOWER'] = afpmstwho15lower
        dicttemp['AFSTWHO15UPPER'] = afpmstwho15upper
        dicttemp['AFCOPDWHO15'] = afpmcopdwho15
        dicttemp['AFCOPDWHO15LOWER'] = afpmcopdwho15lower
        dicttemp['AFCOPDWHO15UPPER'] = afpmcopdwho15upper
        dicttemp['AFLCWHO15'] = afpmlcwho15
        dicttemp['AFLCWHO15LOWER'] = afpmlcwho15lower
        dicttemp['AFLCWHO15UPPER'] = afpmlcwho15upper
        dicttemp['AFDMWHO15'] = afpmdmwho15
        dicttemp['AFDMWHO15LOWER'] = afpmdmwho15lower
        dicttemp['AFDMWHO15UPPER'] = afpmdmwho15upper
        dicttemp['AFLRIWHO15'] = afpmlriwho15
        dicttemp['AFLRIWHO15LOWER'] = afpmlriwho15lower
        dicttemp['AFLRIWHO15UPPER'] = afpmlriwho15upper
        dicttemp['AFIHDNAAQS12'] = afpmihdnaaqs12
        dicttemp['AFIHDNAAQS12LOWER'] = afpmihdnaaqs12lower
        dicttemp['AFIHDNAAQS12UPPER'] = afpmihdnaaqs12upper
        dicttemp['AFSTNAAQS12'] = afpmstnaaqs12
        dicttemp['AFSTNAAQS12LOWER'] = afpmstnaaqs12lower
        dicttemp['AFSTNAAQS12UPPER'] = afpmstnaaqs12upper
        dicttemp['AFCOPDNAAQS12'] = afpmcopdnaaqs12
        dicttemp['AFCOPDNAAQS12LOWER'] = afpmcopdnaaqs12lower
        dicttemp['AFCOPDNAAQS12UPPER'] = afpmcopdnaaqs12upper
        dicttemp['AFLCNAAQS12'] = afpmlcnaaqs12
        dicttemp['AFLCNAAQS12LOWER'] = afpmlcnaaqs12lower
        dicttemp['AFLCNAAQS12UPPER'] = afpmlcnaaqs12upper
        dicttemp['AFDMNAAQS12'] = afpmdmnaaqs12
        dicttemp['AFDMNAAQS12LOWER'] = afpmdmnaaqs12lower
        dicttemp['AFDMNAAQS12UPPER'] = afpmdmnaaqs12upper
        dicttemp['AFLRINAAQS12'] = afpmlrinaaqs12
        dicttemp['AFLRINAAQS12LOWER'] = afpmlrinaaqs12lower
        dicttemp['AFLRINAAQS12UPPER'] = afpmlrinaaqs12upper
        dicttemp['AFIHDWHO10'] = afpmihdwho10
        dicttemp['AFIHDWHO10LOWER'] = afpmihdwho10lower
        dicttemp['AFIHDWHO10UPPER'] = afpmihdwho10upper
        dicttemp['AFSTWHO10'] = afpmstwho10
        dicttemp['AFSTWHO10LOWER'] = afpmstwho10lower
        dicttemp['AFSTWHO10UPPER'] = afpmstwho10upper
        dicttemp['AFCOPDWHO10'] = afpmcopdwho10
        dicttemp['AFCOPDWHO10LOWER'] = afpmcopdwho10lower
        dicttemp['AFCOPDWHO10UPPER'] = afpmcopdwho10upper
        dicttemp['AFLCWHO10'] = afpmlcwho10
        dicttemp['AFLCWHO10LOWER'] = afpmlcwho10lower
        dicttemp['AFLCWHO10UPPER'] = afpmlcwho10upper
        dicttemp['AFDMWHO10'] = afpmdmwho10
        dicttemp['AFDMWHO10LOWER'] = afpmdmwho10lower
        dicttemp['AFDMWHO10UPPER'] = afpmdmwho10upper
        dicttemp['AFLRIWHO10'] = afpmlriwho10     
        dicttemp['AFLRIWHO10LOWER'] = afpmlriwho10lower  
        dicttemp['AFLRIWHO10UPPER'] = afpmlriwho10upper   
        dicttemp['AFIHDWHO5'] = afpmihdwho5
        dicttemp['AFIHDWHO5LOWER'] = afpmihdwho5lower
        dicttemp['AFIHDWHO5UPPER'] = afpmihdwho5upper
        dicttemp['AFSTWHO5'] = afpmstwho5
        dicttemp['AFSTWHO5LOWER'] = afpmstwho5lower
        dicttemp['AFSTWHO5UPPER'] = afpmstwho5upper
        dicttemp['AFCOPDWHO5'] = afpmcopdwho5
        dicttemp['AFCOPDWHO5LOWER'] = afpmcopdwho5lower
        dicttemp['AFCOPDWHO5UPPER'] = afpmcopdwho5upper
        dicttemp['AFLCWHO5'] = afpmlcwho5
        dicttemp['AFLCWHO5LOWER'] = afpmlcwho5lower
        dicttemp['AFLCWHO5UPPER'] = afpmlcwho5upper
        dicttemp['AFDMWHO5'] = afpmdmwho5
        dicttemp['AFDMWHO5LOWER'] = afpmdmwho5lower
        dicttemp['AFDMWHO5UPPER'] = afpmdmwho5upper
        dicttemp['AFLRIWHO5'] = afpmlriwho5
        dicttemp['AFLRIWHO5LOWER'] = afpmlriwho5lower
        dicttemp['AFLRIWHO5UPPER'] = afpmlriwho5upper
        dicttemp['INTERPFLAG_PM25'] = np.nanmean(pminterpflag)
        # # # # Fetch coordinates within tracts for TROPOMI dataset
        if vintage == '2015-2019':
            for i, ilat in enumerate(lat_tropomi_subset):
                for j, jlng in enumerate(lng_tropomi_subset): 
                    point = Point(jlng, ilat)
                    if tract.contains(point) is True:
                        tropomi_inside.append(tropomi_subset[i,j])
            if len(tropomi_inside)==0:
                if (statefips!='02') and (statefips!='15') and (statefips!='72'): 
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
            dicttemp['TROPOMINO2'] = np.nanmean(tropomi_inside)      
            
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
    df.to_csv(DIR_OUT+'asthma_af%s_acs%s_%s_v6.csv'%(no2year, vintage, 
        statefips), sep = ',')
    print('# # # # Output file written!\n', file=f)
    return 

vintages = ['2015-2019', '2006-2010', '2014-2018', '2013-2017', '2012-2016', 
    '2011-2015', '2010-2014', '2009-2013', '2008-2012', '2007-2011']  
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