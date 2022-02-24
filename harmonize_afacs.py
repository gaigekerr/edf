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

Original version created on 25 May 2021
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
# DIR_PM25 = '/GWSPH/groups/anenberggrp/ghkerr/data/pm25/PM25/'
# DIR_FIG = DIR_ROOT
# DIR_GEO = DIR_ROOT+'tigerline/'
# DIR_OUT = DIR_ROOT+'harmonizedtables/'

# DIR_ROOT = '/Users/ghkerr/GW/data/edf/'
# DIR_TROPOMI = '/Users/ghkerr/GW/data/tropomi/'
# DIR_GBD = '/Users/ghkerr/GW/data/gbd/'
# DIR_CENSUS = '/Users/ghkerr/GW/data/demographics/'
# DIR_CROSS = '/Users/ghkerr/GW/data/demographics/'
# DIR_NO2 = '/Users/ghkerr/GW/data/anenberg_mohegh_no2/no2/'
# DIR_PM25 = '/Users/ghkerr/Downloads/Annual-2/'
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
        suminf = np.nansum(sumsup)
        sumsup = np.nansum(np.array(sumsup) * np.array(z))
        u = sumsup / suminf
        xyzi = [xi[p], yi[p], u]
        lstxyzi.append(xyzi)
    return lstxyzi

# def open_V4NAO3(year, checkplot=False):
#     """Open North American Regional Estimates (V4.NA.03) of ground-level fine 
#     particulate matter (PM2.5) total and compositional mass concentrations
#     for year of interest at 0.01˚ x 0.01˚; data can be found at 
#     https://sites.wustl.edu/acag/datasets/surface-pm2-5/
    
#     Parameters
#     ----------
#     year : int
#         Year of interest
        
#     Returns
#     -------
#     lat : numpy.ma.core.MaskedArray
#         Latitude, units of degrees north, [lat,]
#     lng : numpy.ma.core.MaskedArray
#         Longitude, units of degrees east, [lng,]
#     total : numpy.ma.core.MaskedArray
#         Total PM2.5 mass, units of µg m-3, [lat, lng]
#     bc : numpy.ma.core.MaskedArray
#         Contribution of black carbon, units of %, [lat, lng]
#     nh4 : numpy.ma.core.MaskedArray
#         Contribution of ammonium, units of %, [lat, lng]
#     nit : numpy.ma.core.MaskedArray
#          Contribution of nitrate, units of %, [lat, lng]
#     om : numpy.ma.core.MaskedArray
#         Contribution of organic matter, units of %, [lat, lng]
#     so4 : numpy.ma.core.MaskedArray
#         Contribution of sulfate, units of %, [lat, lng]
#     soil : numpy.ma.core.MaskedArray
#         Contribution of soil/crustal particulates, units of %, [lat, lng]
#     ss : numpy.ma.core.MaskedArray
#         Contribution of sea salt particulates, units of %, [lat, lng]
    
#     References
#     ----------
#     Hammer, M. S.; van Donkelaar, A.; Li, C.; Lyapustin, A.; Sayer, A. M.; Hsu, 
#         N. C.; Levy, R. C.; Garay, M. J.; Kalashnikova, O. V.; Kahn, R. A.; 
#         Brauer, M.; Apte, J. S.; Henze, D. K.; Zhang, L.; Zhang, Q.; Ford, B.; 
#         Pierce, J. R.; and Martin, R. V., Global Estimates and Long-Term Trends 
#         of Fine Particulate Matter Concentrations (1998-2018)., Environ. Sci. 
#         Technol, doi: 10.1021/acs.est.0c01764, 2020. 
#     van Donkelaar, A., R. V. Martin, et al. (2019). Regional Estimates of 
#         Chemical Composition of Fine Particulate Matter using a Combined 
#         Geoscience-Statistical Method with Information from Satellites, Models, 
#         and Monitors. Environmental Science & Technology, 2019, 
#         doi:10.1021/acs.est.8b06392.
#     """
#     def open_V4NA03_species(species, year):
#         """Open Dalhousie/WUSTL V4.NA.03 total or composition PM2.5 mass using 
#         Geographically Weighted Regression for year of interest; note that for total 
#         PM2.5 mass, species='PM25'. For individual component percentages/mass 
#         contribution species='BC','NH4','NIT','OM','SO4','SOIL', or 'SS'.
    
#         Parameters
#         ----------
#         species : str
#             PM2.5 component of interest
#         year : int
#             Year of interest
    
#         Returns
#         -------
#         lat : numpy.ma.core.MaskedArray
#             Latitude, units of degrees north, [lat,]
#         lng : numpy.ma.core.MaskedArray
#             Longitude, units of degrees east, [lng,]
#         pm25 : numpy.ma.core.MaskedArray
#             Total PM2.5 mass or percentage contribution from individual components 
#             to total mass, units of µg m-3 or %, [lat, lng]
#         """
#         import glob
#         import netCDF4 as nc
#         fname = glob.glob(DIR_PM25+'*%s*/*%s*%s*.nc'%(species.upper(), 
#             species.upper(), year))
#         pm25 = nc.Dataset(fname[0], 'r')
#         lat = pm25.variables['LAT'][:]
#         lng = pm25.variables['LON'][:]
#         pm25 = pm25.variables['%s'%species.upper()][:]
#         return lat, lng, pm25
#     # Load total PM2.5 mass and component contribution 
#     lat, lng, total = open_V4NA03_species('PM25', year)
#     lat, lng, bc = open_V4NA03_species('bc', year)
#     lat, lng, nh4 = open_V4NA03_species('nh4', year)
#     lat, lng, nit = open_V4NA03_species('nit', year)
#     lat, lng, om = open_V4NA03_species('om', year)
#     lat, lng, so4 = open_V4NA03_species('so4', year)
#     lat, lng, soil = open_V4NA03_species('soil', year)
#     lat, lng, ss = open_V4NA03_species('ss', year)
#     if checkplot==True:
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.axes_grid1 import make_axes_locatable
#         import cartopy.crs as ccrs
#         fig = plt.figure(figsize=(9,7))
#         ax1 = plt.subplot2grid((3,3),(0,0), projection=ccrs.LambertConformal())
#         ax2 = plt.subplot2grid((3,3),(0,1), projection=ccrs.LambertConformal())
#         ax3 = plt.subplot2grid((3,3),(0,2), projection=ccrs.LambertConformal())
#         ax4 = plt.subplot2grid((3,3),(1,0), projection=ccrs.LambertConformal())
#         ax5 = plt.subplot2grid((3,3),(1,1), projection=ccrs.LambertConformal())
#         ax6 = plt.subplot2grid((3,3),(1,2), projection=ccrs.LambertConformal())
#         ax7 = plt.subplot2grid((3,3),(2,0), projection=ccrs.LambertConformal())
#         ax8 = plt.subplot2grid((3,3),(2,1), projection=ccrs.LambertConformal())
#         ax9 = plt.subplot2grid((3,3),(2,2), projection=ccrs.LambertConformal())
#         # Total mass
#         p1 = ax1.pcolormesh(lng[::5], lat[::5], total[::5,::5], vmin=0, vmax=15)
#         ax1.set_title('Total mass [$\mu$g m$^{-3}$]')
#         # BC contribution
#         p2 = ax2.pcolormesh(lng[::5], lat[::5], bc[::5,::5], vmin=0, vmax=25)
#         ax2.set_title('BC [%]')
#         # NH4 contribution
#         p3 = ax3.pcolormesh(lng[::5], lat[::5], nh4[::5,::5], vmin=0, vmax=25)
#         ax3.set_title('NH4 [%]')
#         # NIT contribution
#         p4 = ax4.pcolormesh(lng[::5], lat[::5], nit[::5,::5], vmin=0, vmax=25)
#         ax4.set_title('NIT [%]')
#         # OM contribution
#         p5 = ax5.pcolormesh(lng[::5], lat[::5], om[::5,::5], vmin=0, vmax=25)
#         ax5.set_title('OM [%]')
#         # SO4 contribution
#         p6 = ax6.pcolormesh(lng[::5], lat[::5], so4[::5,::5], vmin=0, vmax=25)
#         ax6.set_title('SO4 [%]')
#         # SOIL contribution
#         p7 = ax7.pcolormesh(lng[::5], lat[::5], soil[::5,::5], vmin=0, vmax=25)
#         ax7.set_title('SOIL [%]')
#         # SS contribution
#         p8 = ax8.pcolormesh(lng[::5], lat[::5], ss[::5,::5], vmin=0, vmax=25)
#         ax8.set_title('SS [%]')
#         # Total contribution
#         p9 = ax9.pcolormesh(lng[::5], lat[::5], (bc+nh4+nit+om+so4+soil+ss
#             ).data[::5,::5], vmin=99, vmax=101, cmap=plt.get_cmap('bwr'))
#         ax9.set_title('Total contribution [%]')
#         # Add colorbars, set extent
#         for mb, ax in zip([p1, p2, p3, p4, p5, p6, p7, p8, p9], 
#             [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes('bottom', size='5%', pad=0.15, 
#                 axes_class=plt.Axes)
#             fig.colorbar(mb, cax=cax, spacing='proportional', 
#                 orientation='horizontal', extend='max')
#             ax.set_extent([-125, -66.5, 20, 50], ccrs.LambertConformal())
#             ax.coastlines()
#         plt.savefig(DIR_FIG+'checkplot_pm25na_wustl_%s.png'%year, dpi=500)
#         plt.show()
#     return lat, lng, total, bc, nh4, nit, om, so4, soil, ss

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

    # For subsetting maps for NO2/PM2.5 harmonization 
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
    # RR for PM2.5-attributable ischemic heart disease, ages 25 to 29 
    rrpmihd_25 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_25.csv')
    # ages 30 to 34 
    rrpmihd_30 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_30.csv')
    # ages 35 to 39 
    rrpmihd_35 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_35.csv')
    # ages 40 to 44
    rrpmihd_40 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_40.csv')
    # ages 45 to 49 
    rrpmihd_45 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_45.csv')
    # ages 50 to 54 
    rrpmihd_50 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_50.csv')
    # ages 55 to 59 
    rrpmihd_55 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_55.csv') 
    # ages 60 to 64 
    rrpmihd_60 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_60.csv')
    # ages 65 to 69 
    rrpmihd_65 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_65.csv')
    # ages 70 to 74 
    rrpmihd_70 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_70.csv')
    # ages 75 to 79
    rrpmihd_75 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_75.csv')
    # ages 80 to 84 
    rrpmihd_80 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_80.csv')
    # ages 85 to 89 
    rrpmihd_85 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_85.csv')
    # ages 90 to 94 
    rrpmihd_90 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_90.csv')
    # ages 95 to 99
    rrpmihd_95 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd_95.csv')
    
    # RR for PM2.5-attributable stroke, ages 25 to 29
    rrpmst_25 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_25.csv')
    # ages 30 to 34 
    rrpmst_30 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_30.csv')
    # ages 35 to 39 
    rrpmst_35 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_35.csv')
    # ages 40 to 44
    rrpmst_40 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_40.csv')
    # ages 45 to 49 
    rrpmst_45 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_45.csv')
    # ages 50 to 54 
    rrpmst_50 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_50.csv')
    # ages 55 to 59 
    rrpmst_55 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_55.csv') 
    # ages 60 to 64 
    rrpmst_60 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_60.csv')
    # ages 65 to 69 
    rrpmst_65 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_65.csv')
    # ages 70 to 74 
    rrpmst_70 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_70.csv')
    # ages 75 to 79
    rrpmst_75 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_75.csv')
    # ages 80 to 84 
    rrpmst_80 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_80.csv')
    # ages 85 to 89 
    rrpmst_85 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_85.csv')
    # ages 90 to 94 
    rrpmst_90 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_90.csv')
    # ages 95 to 99
    rrpmst_95 = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke_95.csv')
    # RR for PM2.5-attributable chronic obstructive pulmonary disease
    rrpmcopd = pd.read_csv(DIR_GBD+'mrbrt_summary/resp_copd.csv')
    # RR for PM2.5-attributable tracheal, bronchus, and lung cancer
    rrpmlc = pd.read_csv(DIR_GBD+'mrbrt_summary/neo_lung.csv')
    # RR for PM2.5-attributable diabetes mellitus type 2
    rrpmdm = pd.read_csv(DIR_GBD+'mrbrt_summary/t2_dm.csv')
    # RR for PM2.5 attributable lower respiratory infections
    rrpmlri = pd.read_csv(DIR_GBD+'mrbrt_summary/lri.csv')
    print('# # # # GBD RRs loaded!', file=f)    
    
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
            # Calculate attributable fraction based on tract-averaged NO2. 
            # The first method is using the concentration-response factor of 
            # 1.26 (1.10 - 1.37) per 10 ppb is used in Achakulwisut et al. 
            # (2019) and taken from Khreis et al. (2017). Note that this 
            # "log-linear" relationship comes from epidemiological studies that 
            # log-transform concentration before regressing with incidence of 
            # health outcome (where log is the natural logarithm). Additional 
            # details can be found in Anenberg et al. (2010)
            af = (1-np.exp(-beta*no2_inside))
            afupper = (1-np.exp(-betaupper*no2_inside))
            aflower = (1-np.exp(-betalower*no2_inside))
            afwho40 = (1-np.exp(-beta*no2_insidewho40))
            afupperwho40 = (1-np.exp(-betaupper*no2_insidewho40))
            aflowerwho40 = (1-np.exp(-betalower*no2_insidewho40))
            afwho30 = (1-np.exp(-beta*no2_insidewho30))
            afupperwho30 = (1-np.exp(-betaupper*no2_insidewho30))
            aflowerwho30 = (1-np.exp(-betalower*no2_insidewho30))
            afwho20 = (1-np.exp(-beta*no2_insidewho20))
            afupperwho20 = (1-np.exp(-betaupper*no2_insidewho20))
            aflowerwho20 = (1-np.exp(-betalower*no2_insidewho20))
            afwho10 = (1-np.exp(-beta*no2_insidewho10))
            afupperwho10 = (1-np.exp(-betaupper*no2_insidewho10))
            aflowerwho10 = (1-np.exp(-betalower*no2_insidewho10))            
            
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
            af = np.nan
            afupper = np.nan
            aflower = np.nan
            afgbdmean = np.nan
            afgbdmed = np.nan
            afgbdupper = np.nan
            afgbdlower  = np.nan
            afwho40 = np.nan
            afupperwho40 = np.nan
            aflowerwho40 = np.nan
            afgbdmeanwho40 = np.nan
            afgbdmedwho40 = np.nan
            afgbdupperwho40 = np.nan
            afgbdlowerwho40  = np.nan
            afwho30 = np.nan
            afupperwho30 = np.nan
            aflowerwho30 = np.nan
            afgbdmeanwho30 = np.nan
            afgbdmedwho30 = np.nan
            afgbdupperwho30 = np.nan
            afgbdlowerwho30  = np.nan
            afwho20 = np.nan
            afupperwho20 = np.nan
            aflowerwho20 = np.nan
            afgbdmeanwho20 = np.nan
            afgbdmedwho20 = np.nan
            afgbdupperwho20 = np.nan
            afgbdlowerwho20  = np.nan
            afwho10 = np.nan
            afupperwho10 = np.nan
            aflowerwho10 = np.nan
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
            'AFPA':af,
            'AFPAUPPER':afupper,
            'AFPALOWER':aflower,
            'AFPAWHO40':afwho40,
            'AFPAUPPERWHO40':afupperwho40,
            'AFPALOWERWHO40':aflowerwho40,
            'AFPAWHO30':afwho30,
            'AFPAUPPERWHO30':afupperwho30,
            'AFPALOWERWHO30':aflowerwho30,
            'AFPAWHO20':afwho20,
            'AFPAUPPERWHO20':afupperwho20,
            'AFPALOWERWHO20':aflowerwho20,
            'AFPAWHO10':afwho10,
            'AFPAUPPERWHO10':afupperwho10,
            'AFPALOWERWHO10':aflowerwho10,
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
            ci = rrpmst_25['exposure_spline'].sub(pm_inside
                ).abs().idxmin()
            # Closest exposure to tract-averaged PM25
            cirrpmihd_25 = rrpmihd_25.iloc[ci]['mean']
            cirrpmihd_30 = rrpmihd_30.iloc[ci]['mean']
            cirrpmihd_35 = rrpmihd_35.iloc[ci]['mean']
            cirrpmihd_40 = rrpmihd_40.iloc[ci]['mean']
            cirrpmihd_45 = rrpmihd_45.iloc[ci]['mean']
            cirrpmihd_50 = rrpmihd_50.iloc[ci]['mean']
            cirrpmihd_55 = rrpmihd_55.iloc[ci]['mean']
            cirrpmihd_60 = rrpmihd_60.iloc[ci]['mean']
            cirrpmihd_65 = rrpmihd_65.iloc[ci]['mean']
            cirrpmihd_70 = rrpmihd_70.iloc[ci]['mean']
            cirrpmihd_75 = rrpmihd_75.iloc[ci]['mean']
            cirrpmihd_80 = rrpmihd_80.iloc[ci]['mean']
            cirrpmihd_85 = rrpmihd_85.iloc[ci]['mean']
            cirrpmihd_90 = rrpmihd_90.iloc[ci]['mean']
            cirrpmihd_95 = rrpmihd_95.iloc[ci]['mean']
            cirrpmst_25 = rrpmst_25.iloc[ci]['mean']
            cirrpmst_30 = rrpmst_30.iloc[ci]['mean']
            cirrpmst_35 = rrpmst_35.iloc[ci]['mean']
            cirrpmst_40 = rrpmst_40.iloc[ci]['mean']
            cirrpmst_45 = rrpmst_45.iloc[ci]['mean']
            cirrpmst_50 = rrpmst_50.iloc[ci]['mean']
            cirrpmst_55 = rrpmst_55.iloc[ci]['mean']
            cirrpmst_60 = rrpmst_60.iloc[ci]['mean']
            cirrpmst_65 = rrpmst_65.iloc[ci]['mean']
            cirrpmst_70 = rrpmst_70.iloc[ci]['mean']
            cirrpmst_75 = rrpmst_75.iloc[ci]['mean']
            cirrpmst_80 = rrpmst_80.iloc[ci]['mean']
            cirrpmst_85 = rrpmst_85.iloc[ci]['mean']
            cirrpmst_90 = rrpmst_90.iloc[ci]['mean']
            cirrpmst_95 = rrpmst_95.iloc[ci]['mean']
            cirrpmcopd = rrpmcopd.iloc[ci]['mean']
            cirrpmlc = rrpmlc.iloc[ci]['mean']
            cirrpmdm = rrpmdm.iloc[ci]['mean']
            cirrpmlri = rrpmlri.iloc[ci]['mean']
            # Calculate PAF as (RR-1)/RR
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
            # WHO Interim 3
            ciwho15 = rrpmst_25['exposure_spline'].sub(pm_insidewho15
                ).abs().idxmin()
            cirrpmihd_25 = rrpmihd_25.iloc[ciwho15]['mean']
            cirrpmihd_30 = rrpmihd_30.iloc[ciwho15]['mean']
            cirrpmihd_35 = rrpmihd_35.iloc[ciwho15]['mean']
            cirrpmihd_40 = rrpmihd_40.iloc[ciwho15]['mean']
            cirrpmihd_45 = rrpmihd_45.iloc[ciwho15]['mean']
            cirrpmihd_50 = rrpmihd_50.iloc[ciwho15]['mean']
            cirrpmihd_55 = rrpmihd_55.iloc[ciwho15]['mean']
            cirrpmihd_60 = rrpmihd_60.iloc[ciwho15]['mean']
            cirrpmihd_65 = rrpmihd_65.iloc[ciwho15]['mean']
            cirrpmihd_70 = rrpmihd_70.iloc[ciwho15]['mean']
            cirrpmihd_75 = rrpmihd_75.iloc[ciwho15]['mean']
            cirrpmihd_80 = rrpmihd_80.iloc[ciwho15]['mean']
            cirrpmihd_85 = rrpmihd_85.iloc[ciwho15]['mean']
            cirrpmihd_90 = rrpmihd_90.iloc[ciwho15]['mean']
            cirrpmihd_95 = rrpmihd_95.iloc[ciwho15]['mean']
            cirrpmst_25 = rrpmst_25.iloc[ciwho15]['mean']
            cirrpmst_30 = rrpmst_30.iloc[ciwho15]['mean']
            cirrpmst_35 = rrpmst_35.iloc[ciwho15]['mean']
            cirrpmst_40 = rrpmst_40.iloc[ciwho15]['mean']
            cirrpmst_45 = rrpmst_45.iloc[ciwho15]['mean']
            cirrpmst_50 = rrpmst_50.iloc[ciwho15]['mean']
            cirrpmst_55 = rrpmst_55.iloc[ciwho15]['mean']
            cirrpmst_60 = rrpmst_60.iloc[ciwho15]['mean']
            cirrpmst_65 = rrpmst_65.iloc[ciwho15]['mean']
            cirrpmst_70 = rrpmst_70.iloc[ciwho15]['mean']
            cirrpmst_75 = rrpmst_75.iloc[ciwho15]['mean']
            cirrpmst_80 = rrpmst_80.iloc[ciwho15]['mean']
            cirrpmst_85 = rrpmst_85.iloc[ciwho15]['mean']
            cirrpmst_90 = rrpmst_90.iloc[ciwho15]['mean']
            cirrpmst_95 = rrpmst_95.iloc[ciwho15]['mean']
            cirrpmcopd = rrpmcopd.iloc[ciwho15]['mean']
            cirrpmlc = rrpmlc.iloc[ciwho15]['mean']
            cirrpmdm = rrpmdm.iloc[ciwho15]['mean']
            cirrpmlri = rrpmlri.iloc[ciwho15]['mean']
            afpmihd_25who15 = (cirrpmihd_25-1.)/cirrpmihd_25
            afpmihd_30who15 = (cirrpmihd_30-1.)/cirrpmihd_30
            afpmihd_35who15 = (cirrpmihd_35-1.)/cirrpmihd_35
            afpmihd_40who15 = (cirrpmihd_40-1.)/cirrpmihd_40
            afpmihd_45who15 = (cirrpmihd_45-1.)/cirrpmihd_45
            afpmihd_50who15 = (cirrpmihd_50-1.)/cirrpmihd_50
            afpmihd_55who15 = (cirrpmihd_55-1.)/cirrpmihd_55
            afpmihd_60who15 = (cirrpmihd_60-1.)/cirrpmihd_60
            afpmihd_65who15 = (cirrpmihd_65-1.)/cirrpmihd_65
            afpmihd_70who15 = (cirrpmihd_70-1.)/cirrpmihd_70
            afpmihd_75who15 = (cirrpmihd_75-1.)/cirrpmihd_75
            afpmihd_80who15 = (cirrpmihd_80-1.)/cirrpmihd_80
            afpmihd_85who15 = (cirrpmihd_85-1.)/cirrpmihd_85
            afpmihd_90who15 = (cirrpmihd_90-1.)/cirrpmihd_90
            afpmihd_95who15 = (cirrpmihd_95-1.)/cirrpmihd_95
            afpmst_25who15 = (cirrpmst_25-1.)/cirrpmst_25
            afpmst_30who15 = (cirrpmst_30-1.)/cirrpmst_30
            afpmst_35who15 = (cirrpmst_35-1.)/cirrpmst_35
            afpmst_40who15 = (cirrpmst_40-1.)/cirrpmst_40
            afpmst_45who15 = (cirrpmst_45-1.)/cirrpmst_45
            afpmst_50who15 = (cirrpmst_50-1.)/cirrpmst_50
            afpmst_55who15 = (cirrpmst_55-1.)/cirrpmst_55
            afpmst_60who15 = (cirrpmst_60-1.)/cirrpmst_60
            afpmst_65who15 = (cirrpmst_65-1.)/cirrpmst_65
            afpmst_70who15 = (cirrpmst_70-1.)/cirrpmst_70
            afpmst_75who15 = (cirrpmst_75-1.)/cirrpmst_75
            afpmst_80who15 = (cirrpmst_80-1.)/cirrpmst_80
            afpmst_85who15 = (cirrpmst_85-1.)/cirrpmst_85
            afpmst_90who15 = (cirrpmst_90-1.)/cirrpmst_90
            afpmst_95who15 = (cirrpmst_95-1.)/cirrpmst_95
            afpmcopdwho15 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcwho15 = (cirrpmlc-1.)/cirrpmlc
            afpmdmwho15 = (cirrpmdm-1.)/cirrpmdm
            afpmlriwho15 = (cirrpmlri-1.)/cirrpmlri               
            # EPA NAAQS
            cinaaqs12 = rrpmst_25['exposure_spline'].sub(pm_insidenaaqs12
                ).abs().idxmin()
            cirrpmihd_25 = rrpmihd_25.iloc[cinaaqs12]['mean']
            cirrpmihd_30 = rrpmihd_30.iloc[cinaaqs12]['mean']
            cirrpmihd_35 = rrpmihd_35.iloc[cinaaqs12]['mean']
            cirrpmihd_40 = rrpmihd_40.iloc[cinaaqs12]['mean']
            cirrpmihd_45 = rrpmihd_45.iloc[cinaaqs12]['mean']
            cirrpmihd_50 = rrpmihd_50.iloc[cinaaqs12]['mean']
            cirrpmihd_55 = rrpmihd_55.iloc[cinaaqs12]['mean']
            cirrpmihd_60 = rrpmihd_60.iloc[cinaaqs12]['mean']
            cirrpmihd_65 = rrpmihd_65.iloc[cinaaqs12]['mean']
            cirrpmihd_70 = rrpmihd_70.iloc[cinaaqs12]['mean']
            cirrpmihd_75 = rrpmihd_75.iloc[cinaaqs12]['mean']
            cirrpmihd_80 = rrpmihd_80.iloc[cinaaqs12]['mean']
            cirrpmihd_85 = rrpmihd_85.iloc[cinaaqs12]['mean']
            cirrpmihd_90 = rrpmihd_90.iloc[cinaaqs12]['mean']
            cirrpmihd_95 = rrpmihd_95.iloc[cinaaqs12]['mean']
            cirrpmst_25 = rrpmst_25.iloc[cinaaqs12]['mean']
            cirrpmst_30 = rrpmst_30.iloc[cinaaqs12]['mean']
            cirrpmst_35 = rrpmst_35.iloc[cinaaqs12]['mean']
            cirrpmst_40 = rrpmst_40.iloc[cinaaqs12]['mean']
            cirrpmst_45 = rrpmst_45.iloc[cinaaqs12]['mean']
            cirrpmst_50 = rrpmst_50.iloc[cinaaqs12]['mean']
            cirrpmst_55 = rrpmst_55.iloc[cinaaqs12]['mean']
            cirrpmst_60 = rrpmst_60.iloc[cinaaqs12]['mean']
            cirrpmst_65 = rrpmst_65.iloc[cinaaqs12]['mean']
            cirrpmst_70 = rrpmst_70.iloc[cinaaqs12]['mean']
            cirrpmst_75 = rrpmst_75.iloc[cinaaqs12]['mean']
            cirrpmst_80 = rrpmst_80.iloc[cinaaqs12]['mean']
            cirrpmst_85 = rrpmst_85.iloc[cinaaqs12]['mean']
            cirrpmst_90 = rrpmst_90.iloc[cinaaqs12]['mean']
            cirrpmst_95 = rrpmst_95.iloc[cinaaqs12]['mean']
            cirrpmcopd = rrpmcopd.iloc[cinaaqs12]['mean']
            cirrpmlc = rrpmlc.iloc[cinaaqs12]['mean']
            cirrpmdm = rrpmdm.iloc[cinaaqs12]['mean']
            cirrpmlri = rrpmlri.iloc[cinaaqs12]['mean']
            afpmihd_25naaqs12 = (cirrpmihd_25-1.)/cirrpmihd_25
            afpmihd_30naaqs12 = (cirrpmihd_30-1.)/cirrpmihd_30
            afpmihd_35naaqs12 = (cirrpmihd_35-1.)/cirrpmihd_35
            afpmihd_40naaqs12 = (cirrpmihd_40-1.)/cirrpmihd_40
            afpmihd_45naaqs12 = (cirrpmihd_45-1.)/cirrpmihd_45
            afpmihd_50naaqs12 = (cirrpmihd_50-1.)/cirrpmihd_50
            afpmihd_55naaqs12 = (cirrpmihd_55-1.)/cirrpmihd_55
            afpmihd_60naaqs12 = (cirrpmihd_60-1.)/cirrpmihd_60
            afpmihd_65naaqs12 = (cirrpmihd_65-1.)/cirrpmihd_65
            afpmihd_70naaqs12 = (cirrpmihd_70-1.)/cirrpmihd_70
            afpmihd_75naaqs12 = (cirrpmihd_75-1.)/cirrpmihd_75
            afpmihd_80naaqs12 = (cirrpmihd_80-1.)/cirrpmihd_80
            afpmihd_85naaqs12 = (cirrpmihd_85-1.)/cirrpmihd_85
            afpmihd_90naaqs12 = (cirrpmihd_90-1.)/cirrpmihd_90
            afpmihd_95naaqs12 = (cirrpmihd_95-1.)/cirrpmihd_95
            afpmst_25naaqs12 = (cirrpmst_25-1.)/cirrpmst_25
            afpmst_30naaqs12 = (cirrpmst_30-1.)/cirrpmst_30
            afpmst_35naaqs12 = (cirrpmst_35-1.)/cirrpmst_35
            afpmst_40naaqs12 = (cirrpmst_40-1.)/cirrpmst_40
            afpmst_45naaqs12 = (cirrpmst_45-1.)/cirrpmst_45
            afpmst_50naaqs12 = (cirrpmst_50-1.)/cirrpmst_50
            afpmst_55naaqs12 = (cirrpmst_55-1.)/cirrpmst_55
            afpmst_60naaqs12 = (cirrpmst_60-1.)/cirrpmst_60
            afpmst_65naaqs12 = (cirrpmst_65-1.)/cirrpmst_65
            afpmst_70naaqs12 = (cirrpmst_70-1.)/cirrpmst_70
            afpmst_75naaqs12 = (cirrpmst_75-1.)/cirrpmst_75
            afpmst_80naaqs12 = (cirrpmst_80-1.)/cirrpmst_80
            afpmst_85naaqs12 = (cirrpmst_85-1.)/cirrpmst_85
            afpmst_90naaqs12 = (cirrpmst_90-1.)/cirrpmst_90
            afpmst_95naaqs12 = (cirrpmst_95-1.)/cirrpmst_95
            afpmcopdnaaqs12 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcnaaqs12 = (cirrpmlc-1.)/cirrpmlc
            afpmdmnaaqs12 = (cirrpmdm-1.)/cirrpmdm
            afpmlrinaaqs12 = (cirrpmlri-1.)/cirrpmlri
            # WHO Interim 4                
            ciwho10 = rrpmst_25['exposure_spline'].sub(pm_insidewho10
                ).abs().idxmin()
            cirrpmihd_25 = rrpmihd_25.iloc[ciwho10]['mean']
            cirrpmihd_30 = rrpmihd_30.iloc[ciwho10]['mean']
            cirrpmihd_35 = rrpmihd_35.iloc[ciwho10]['mean']
            cirrpmihd_40 = rrpmihd_40.iloc[ciwho10]['mean']
            cirrpmihd_45 = rrpmihd_45.iloc[ciwho10]['mean']
            cirrpmihd_50 = rrpmihd_50.iloc[ciwho10]['mean']
            cirrpmihd_55 = rrpmihd_55.iloc[ciwho10]['mean']
            cirrpmihd_60 = rrpmihd_60.iloc[ciwho10]['mean']
            cirrpmihd_65 = rrpmihd_65.iloc[ciwho10]['mean']
            cirrpmihd_70 = rrpmihd_70.iloc[ciwho10]['mean']
            cirrpmihd_75 = rrpmihd_75.iloc[ciwho10]['mean']
            cirrpmihd_80 = rrpmihd_80.iloc[ciwho10]['mean']
            cirrpmihd_85 = rrpmihd_85.iloc[ciwho10]['mean']
            cirrpmihd_90 = rrpmihd_90.iloc[ciwho10]['mean']
            cirrpmihd_95 = rrpmihd_95.iloc[ciwho10]['mean']
            cirrpmst_25 = rrpmst_25.iloc[ciwho10]['mean']
            cirrpmst_30 = rrpmst_30.iloc[ciwho10]['mean']
            cirrpmst_35 = rrpmst_35.iloc[ciwho10]['mean']
            cirrpmst_40 = rrpmst_40.iloc[ciwho10]['mean']
            cirrpmst_45 = rrpmst_45.iloc[ciwho10]['mean']
            cirrpmst_50 = rrpmst_50.iloc[ciwho10]['mean']
            cirrpmst_55 = rrpmst_55.iloc[ciwho10]['mean']
            cirrpmst_60 = rrpmst_60.iloc[ciwho10]['mean']
            cirrpmst_65 = rrpmst_65.iloc[ciwho10]['mean']
            cirrpmst_70 = rrpmst_70.iloc[ciwho10]['mean']
            cirrpmst_75 = rrpmst_75.iloc[ciwho10]['mean']
            cirrpmst_80 = rrpmst_80.iloc[ciwho10]['mean']
            cirrpmst_85 = rrpmst_85.iloc[ciwho10]['mean']
            cirrpmst_90 = rrpmst_90.iloc[ciwho10]['mean']
            cirrpmst_95 = rrpmst_95.iloc[ciwho10]['mean']
            cirrpmcopd = rrpmcopd.iloc[ciwho10]['mean']
            cirrpmlc = rrpmlc.iloc[ciwho10]['mean']
            cirrpmdm = rrpmdm.iloc[ciwho10]['mean']
            cirrpmlri = rrpmlri.iloc[ciwho10]['mean']
            afpmihd_25who10 = (cirrpmihd_25-1.)/cirrpmihd_25
            afpmihd_30who10 = (cirrpmihd_30-1.)/cirrpmihd_30
            afpmihd_35who10 = (cirrpmihd_35-1.)/cirrpmihd_35
            afpmihd_40who10 = (cirrpmihd_40-1.)/cirrpmihd_40
            afpmihd_45who10 = (cirrpmihd_45-1.)/cirrpmihd_45
            afpmihd_50who10 = (cirrpmihd_50-1.)/cirrpmihd_50
            afpmihd_55who10 = (cirrpmihd_55-1.)/cirrpmihd_55
            afpmihd_60who10 = (cirrpmihd_60-1.)/cirrpmihd_60
            afpmihd_65who10 = (cirrpmihd_65-1.)/cirrpmihd_65
            afpmihd_70who10 = (cirrpmihd_70-1.)/cirrpmihd_70
            afpmihd_75who10 = (cirrpmihd_75-1.)/cirrpmihd_75
            afpmihd_80who10 = (cirrpmihd_80-1.)/cirrpmihd_80
            afpmihd_85who10 = (cirrpmihd_85-1.)/cirrpmihd_85
            afpmihd_90who10 = (cirrpmihd_90-1.)/cirrpmihd_90
            afpmihd_95who10 = (cirrpmihd_95-1.)/cirrpmihd_95
            afpmst_25who10 = (cirrpmst_25-1.)/cirrpmst_25
            afpmst_30who10 = (cirrpmst_30-1.)/cirrpmst_30
            afpmst_35who10 = (cirrpmst_35-1.)/cirrpmst_35
            afpmst_40who10 = (cirrpmst_40-1.)/cirrpmst_40
            afpmst_45who10 = (cirrpmst_45-1.)/cirrpmst_45
            afpmst_50who10 = (cirrpmst_50-1.)/cirrpmst_50
            afpmst_55who10 = (cirrpmst_55-1.)/cirrpmst_55
            afpmst_60who10 = (cirrpmst_60-1.)/cirrpmst_60
            afpmst_65who10 = (cirrpmst_65-1.)/cirrpmst_65
            afpmst_70who10 = (cirrpmst_70-1.)/cirrpmst_70
            afpmst_75who10 = (cirrpmst_75-1.)/cirrpmst_75
            afpmst_80who10 = (cirrpmst_80-1.)/cirrpmst_80
            afpmst_85who10 = (cirrpmst_85-1.)/cirrpmst_85
            afpmst_90who10 = (cirrpmst_90-1.)/cirrpmst_90
            afpmst_95who10 = (cirrpmst_95-1.)/cirrpmst_95
            afpmcopdwho10 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcwho10 = (cirrpmlc-1.)/cirrpmlc
            afpmdmwho10 = (cirrpmdm-1.)/cirrpmdm
            afpmlriwho10 = (cirrpmlri-1.)/cirrpmlri
            # WHO AQG
            ciwho5 = rrpmst_25['exposure_spline'].sub(pm_insidewho5
                ).abs().idxmin()
            cirrpmihd_25 = rrpmihd_25.iloc[ciwho5]['mean']
            cirrpmihd_30 = rrpmihd_30.iloc[ciwho5]['mean']
            cirrpmihd_35 = rrpmihd_35.iloc[ciwho5]['mean']
            cirrpmihd_40 = rrpmihd_40.iloc[ciwho5]['mean']
            cirrpmihd_45 = rrpmihd_45.iloc[ciwho5]['mean']
            cirrpmihd_50 = rrpmihd_50.iloc[ciwho5]['mean']
            cirrpmihd_55 = rrpmihd_55.iloc[ciwho5]['mean']
            cirrpmihd_60 = rrpmihd_60.iloc[ciwho5]['mean']
            cirrpmihd_65 = rrpmihd_65.iloc[ciwho5]['mean']
            cirrpmihd_70 = rrpmihd_70.iloc[ciwho5]['mean']
            cirrpmihd_75 = rrpmihd_75.iloc[ciwho5]['mean']
            cirrpmihd_80 = rrpmihd_80.iloc[ciwho5]['mean']
            cirrpmihd_85 = rrpmihd_85.iloc[ciwho5]['mean']
            cirrpmihd_90 = rrpmihd_90.iloc[ciwho5]['mean']
            cirrpmihd_95 = rrpmihd_95.iloc[ciwho5]['mean']
            cirrpmst_25 = rrpmst_25.iloc[ciwho5]['mean']
            cirrpmst_30 = rrpmst_30.iloc[ciwho5]['mean']
            cirrpmst_35 = rrpmst_35.iloc[ciwho5]['mean']
            cirrpmst_40 = rrpmst_40.iloc[ciwho5]['mean']
            cirrpmst_45 = rrpmst_45.iloc[ciwho5]['mean']
            cirrpmst_50 = rrpmst_50.iloc[ciwho5]['mean']
            cirrpmst_55 = rrpmst_55.iloc[ciwho5]['mean']
            cirrpmst_60 = rrpmst_60.iloc[ciwho5]['mean']
            cirrpmst_65 = rrpmst_65.iloc[ciwho5]['mean']
            cirrpmst_70 = rrpmst_70.iloc[ciwho5]['mean']
            cirrpmst_75 = rrpmst_75.iloc[ciwho5]['mean']
            cirrpmst_80 = rrpmst_80.iloc[ciwho5]['mean']
            cirrpmst_85 = rrpmst_85.iloc[ciwho5]['mean']
            cirrpmst_90 = rrpmst_90.iloc[ciwho5]['mean']
            cirrpmst_95 = rrpmst_95.iloc[ciwho5]['mean']
            cirrpmcopd = rrpmcopd.iloc[ciwho5]['mean']
            cirrpmlc = rrpmlc.iloc[ciwho5]['mean']
            cirrpmdm = rrpmdm.iloc[ciwho5]['mean']
            cirrpmlri = rrpmlri.iloc[ciwho5]['mean']
            afpmihd_25who5 = (cirrpmihd_25-1.)/cirrpmihd_25
            afpmihd_30who5 = (cirrpmihd_30-1.)/cirrpmihd_30
            afpmihd_35who5 = (cirrpmihd_35-1.)/cirrpmihd_35
            afpmihd_40who5 = (cirrpmihd_40-1.)/cirrpmihd_40
            afpmihd_45who5 = (cirrpmihd_45-1.)/cirrpmihd_45
            afpmihd_50who5 = (cirrpmihd_50-1.)/cirrpmihd_50
            afpmihd_55who5 = (cirrpmihd_55-1.)/cirrpmihd_55
            afpmihd_60who5 = (cirrpmihd_60-1.)/cirrpmihd_60
            afpmihd_65who5 = (cirrpmihd_65-1.)/cirrpmihd_65
            afpmihd_70who5 = (cirrpmihd_70-1.)/cirrpmihd_70
            afpmihd_75who5 = (cirrpmihd_75-1.)/cirrpmihd_75
            afpmihd_80who5 = (cirrpmihd_80-1.)/cirrpmihd_80
            afpmihd_85who5 = (cirrpmihd_85-1.)/cirrpmihd_85
            afpmihd_90who5 = (cirrpmihd_90-1.)/cirrpmihd_90
            afpmihd_95who5 = (cirrpmihd_95-1.)/cirrpmihd_95
            afpmst_25who5 = (cirrpmst_25-1.)/cirrpmst_25
            afpmst_30who5 = (cirrpmst_30-1.)/cirrpmst_30
            afpmst_35who5 = (cirrpmst_35-1.)/cirrpmst_35
            afpmst_40who5 = (cirrpmst_40-1.)/cirrpmst_40
            afpmst_45who5 = (cirrpmst_45-1.)/cirrpmst_45
            afpmst_50who5 = (cirrpmst_50-1.)/cirrpmst_50
            afpmst_55who5 = (cirrpmst_55-1.)/cirrpmst_55
            afpmst_60who5 = (cirrpmst_60-1.)/cirrpmst_60
            afpmst_65who5 = (cirrpmst_65-1.)/cirrpmst_65
            afpmst_70who5 = (cirrpmst_70-1.)/cirrpmst_70
            afpmst_75who5 = (cirrpmst_75-1.)/cirrpmst_75
            afpmst_80who5 = (cirrpmst_80-1.)/cirrpmst_80
            afpmst_85who5 = (cirrpmst_85-1.)/cirrpmst_85
            afpmst_90who5 = (cirrpmst_90-1.)/cirrpmst_90
            afpmst_95who5 = (cirrpmst_95-1.)/cirrpmst_95
            afpmcopdwho5 = (cirrpmcopd-1.)/cirrpmcopd
            afpmlcwho5 = (cirrpmlc-1.)/cirrpmlc
            afpmdmwho5 = (cirrpmdm-1.)/cirrpmdm
            afpmlriwho5 = (cirrpmlri-1.)/cirrpmlri                
        else:
            pm_insidewho15, pm_insidenaaqs12 = np.nan, np.nan
            pm_insidewho10, pm_insidewho5 = np.nan, np.nan
            afpmihd_25, afpmihd_30, afpmihd_35 = np.nan, np.nan, np.nan
            afpmihd_40, afpmihd_45, afpmihd_50 = np.nan, np.nan, np.nan
            afpmihd_55, afpmihd_60, afpmihd_65 = np.nan, np.nan, np.nan
            afpmihd_70, afpmihd_75, afpmihd_80 = np.nan, np.nan, np.nan
            afpmihd_85, afpmihd_90, afpmihd_95 = np.nan, np.nan, np.nan
            afpmst_25, afpmst_30, afpmst_35 = np.nan, np.nan, np.nan 
            afpmst_40, afpmst_45, afpmst_50 = np.nan, np.nan, np.nan 
            afpmst_55, afpmst_60, afpmst_65 = np.nan, np.nan, np.nan  
            afpmst_70, afpmst_75, afpmst_80 = np.nan, np.nan, np.nan 
            afpmst_85, afpmst_90, afpmst_95 = np.nan, np.nan, np.nan 
            afpmcopd, afpmlc, afpmdm, afpmlri = np.nan, np.nan, np.nan, np.nan
            afpmihd_25who15, afpmihd_30who15 = np.nan, np.nan
            afpmihd_35who15, afpmihd_40who15 = np.nan, np.nan
            afpmihd_45who15, afpmihd_50who15 = np.nan, np.nan
            afpmihd_55who15, afpmihd_60who15 = np.nan, np.nan
            afpmihd_65who15, afpmihd_70who15 = np.nan, np.nan
            afpmihd_75who15, afpmihd_80who15 = np.nan, np.nan
            afpmihd_85who15, afpmihd_90who15 = np.nan, np.nan
            afpmihd_95who15, afpmst_25who15 = np.nan, np.nan
            afpmst_30who15, afpmst_35who15 = np.nan, np.nan
            afpmst_40who15, afpmst_45who15 = np.nan, np.nan
            afpmst_50who15, afpmst_55who15 = np.nan, np.nan
            afpmst_60who15, afpmst_65who15 = np.nan, np.nan
            afpmst_70who15, afpmst_75who15 = np.nan, np.nan
            afpmst_80who15, afpmst_85who15 = np.nan, np.nan
            afpmst_90who15, afpmst_95who15 = np.nan, np.nan
            afpmcopdwho15, afpmlcwho15 = np.nan, np.nan
            afpmdmwho15, afpmlriwho15 = np.nan, np.nan
            afpmihd_25naaqs12, afpmihd_30naaqs12 = np.nan, np.nan
            afpmihd_35naaqs12, afpmihd_40naaqs12 = np.nan, np.nan
            afpmihd_45naaqs12, afpmihd_50naaqs12 = np.nan, np.nan
            afpmihd_55naaqs12, afpmihd_60naaqs12 = np.nan, np.nan
            afpmihd_65naaqs12, afpmihd_70naaqs12 = np.nan, np.nan
            afpmihd_75naaqs12, afpmihd_80naaqs12 = np.nan, np.nan
            afpmihd_85naaqs12, afpmihd_90naaqs12 = np.nan, np.nan
            afpmihd_95naaqs12, afpmst_25naaqs12 = np.nan, np.nan
            afpmst_30naaqs12, afpmst_35naaqs12 = np.nan, np.nan
            afpmst_40naaqs12, afpmst_45naaqs12 = np.nan, np.nan
            afpmst_50naaqs12, afpmst_55naaqs12 = np.nan, np.nan
            afpmst_60naaqs12, afpmst_65naaqs12 = np.nan, np.nan
            afpmst_70naaqs12, afpmst_75naaqs12 = np.nan, np.nan
            afpmst_80naaqs12, afpmst_85naaqs12 = np.nan, np.nan
            afpmst_90naaqs12, afpmst_95naaqs12 = np.nan, np.nan
            afpmcopdnaaqs12, afpmlcnaaqs12 = np.nan, np.nan
            afpmdmnaaqs12, afpmlrinaaqs12 = np.nan, np.nan                
            afpmihd_25who10, afpmihd_30who10 = np.nan, np.nan
            afpmihd_35who10, afpmihd_40who10 = np.nan, np.nan
            afpmihd_45who10, afpmihd_50who10 = np.nan, np.nan
            afpmihd_55who10, afpmihd_60who10 = np.nan, np.nan
            afpmihd_65who10, afpmihd_70who10 = np.nan, np.nan
            afpmihd_75who10, afpmihd_80who10 = np.nan, np.nan
            afpmihd_85who10, afpmihd_90who10 = np.nan, np.nan
            afpmihd_95who10, afpmst_25who10 = np.nan, np.nan
            afpmst_30who10, afpmst_35who10 = np.nan, np.nan
            afpmst_40who10, afpmst_45who10 = np.nan, np.nan
            afpmst_50who10, afpmst_55who10 = np.nan, np.nan
            afpmst_60who10, afpmst_65who10 = np.nan, np.nan
            afpmst_70who10, afpmst_75who10 = np.nan, np.nan
            afpmst_80who10, afpmst_85who10 = np.nan, np.nan
            afpmst_90who10, afpmst_95who10 = np.nan, np.nan
            afpmcopdwho10, afpmlcwho10 = np.nan, np.nan
            afpmdmwho10, afpmlriwho10 = np.nan, np.nan
            afpmihd_25who5, afpmihd_30who5 = np.nan, np.nan
            afpmihd_35who5, afpmihd_40who5 = np.nan, np.nan
            afpmihd_45who5, afpmihd_50who5 = np.nan, np.nan
            afpmihd_55who5, afpmihd_60who5 = np.nan, np.nan
            afpmihd_65who5, afpmihd_70who5 = np.nan, np.nan
            afpmihd_75who5, afpmihd_80who5 = np.nan, np.nan
            afpmihd_85who5, afpmihd_90who5 = np.nan, np.nan
            afpmihd_95who5, afpmst_25who5 = np.nan, np.nan
            afpmst_30who5, afpmst_35who5 = np.nan, np.nan
            afpmst_40who5, afpmst_45who5 = np.nan, np.nan
            afpmst_50who5, afpmst_55who5 = np.nan, np.nan
            afpmst_60who5, afpmst_65who5 = np.nan, np.nan
            afpmst_70who5, afpmst_75who5 = np.nan, np.nan
            afpmst_80who5, afpmst_85who5 = np.nan, np.nan
            afpmst_90who5, afpmst_95who5 = np.nan, np.nan
            afpmcopdwho5, afpmlcwho5 = np.nan, np.nan
            afpmdmwho5, afpmlriwho5 = np.nan, np.nan                
        dicttemp['PM25'] = pm_inside
        dicttemp['PM25WHO15'] = pm_insidewho15
        dicttemp['PM25NAAQS12'] = pm_insidenaaqs12
        dicttemp['PM25WHO10'] = pm_insidewho10
        dicttemp['PM25WHO5'] = pm_insidewho5
        dicttemp['AFIHD_25'] = afpmihd_25
        dicttemp['AFIHD_30'] = afpmihd_30
        dicttemp['AFIHD_35'] = afpmihd_35
        dicttemp['AFIHD_40'] = afpmihd_40
        dicttemp['AFIHD_45'] = afpmihd_45
        dicttemp['AFIHD_50'] = afpmihd_50
        dicttemp['AFIHD_55'] = afpmihd_55
        dicttemp['AFIHD_60'] = afpmihd_60
        dicttemp['AFIHD_65'] = afpmihd_65
        dicttemp['AFIHD_70'] = afpmihd_70
        dicttemp['AFIHD_75'] = afpmihd_75
        dicttemp['AFIHD_80'] = afpmihd_80
        dicttemp['AFIHD_85'] = afpmihd_85
        dicttemp['AFIHD_90'] = afpmihd_90
        dicttemp['AFIHD_95'] = afpmihd_95
        dicttemp['AFST_25'] = afpmst_25
        dicttemp['AFST_30'] = afpmst_30
        dicttemp['AFST_35'] = afpmst_35
        dicttemp['AFST_40'] = afpmst_40
        dicttemp['AFST_45'] = afpmst_45
        dicttemp['AFST_50'] = afpmst_50
        dicttemp['AFST_55'] = afpmst_55
        dicttemp['AFST_60'] = afpmst_60
        dicttemp['AFST_65'] = afpmst_65
        dicttemp['AFST_70'] = afpmst_70
        dicttemp['AFST_75'] = afpmst_75
        dicttemp['AFST_80'] = afpmst_80
        dicttemp['AFST_85'] = afpmst_85
        dicttemp['AFST_90'] = afpmst_90
        dicttemp['AFST_95'] = afpmst_95
        dicttemp['AFCOPD'] = afpmcopd
        dicttemp['AFLC'] = afpmlc
        dicttemp['AFDM'] = afpmdm
        dicttemp['AFLRI'] = afpmlri  
        dicttemp['AFIHD_25WHO15'] = afpmihd_25who15
        dicttemp['AFIHD_30WHO15'] = afpmihd_30who15
        dicttemp['AFIHD_35WHO15'] = afpmihd_35who15
        dicttemp['AFIHD_40WHO15'] = afpmihd_40who15
        dicttemp['AFIHD_45WHO15'] = afpmihd_45who15
        dicttemp['AFIHD_50WHO15'] = afpmihd_50who15
        dicttemp['AFIHD_55WHO15'] = afpmihd_55who15
        dicttemp['AFIHD_60WHO15'] = afpmihd_60who15
        dicttemp['AFIHD_65WHO15'] = afpmihd_65who15
        dicttemp['AFIHD_70WHO15'] = afpmihd_70who15
        dicttemp['AFIHD_75WHO15'] = afpmihd_75who15
        dicttemp['AFIHD_80WHO15'] = afpmihd_80who15
        dicttemp['AFIHD_85WHO15'] = afpmihd_85who15
        dicttemp['AFIHD_90WHO15'] = afpmihd_90who15
        dicttemp['AFIHD_95WHO15'] = afpmihd_95who15
        dicttemp['AFST_25WHO15'] = afpmst_25who15
        dicttemp['AFST_30WHO15'] = afpmst_30who15
        dicttemp['AFST_35WHO15'] = afpmst_35who15
        dicttemp['AFST_40WHO15'] = afpmst_40who15
        dicttemp['AFST_45WHO15'] = afpmst_45who15
        dicttemp['AFST_50WHO15'] = afpmst_50who15
        dicttemp['AFST_55WHO15'] = afpmst_55who15
        dicttemp['AFST_60WHO15'] = afpmst_60who15
        dicttemp['AFST_65WHO15'] = afpmst_65who15
        dicttemp['AFST_70WHO15'] = afpmst_70who15
        dicttemp['AFST_75WHO15'] = afpmst_75who15
        dicttemp['AFST_80WHO15'] = afpmst_80who15
        dicttemp['AFST_85WHO15'] = afpmst_85who15
        dicttemp['AFST_90WHO15'] = afpmst_90who15
        dicttemp['AFST_95WHO15'] = afpmst_95who15
        dicttemp['AFCOPDWHO15'] = afpmcopdwho15
        dicttemp['AFLCWHO15'] = afpmlcwho15
        dicttemp['AFDMWHO15'] = afpmdmwho15
        dicttemp['AFLRIWHO15'] = afpmlriwho15    
        dicttemp['AFIHD_25NAAQS12'] = afpmihd_25naaqs12
        dicttemp['AFIHD_30NAAQS12'] = afpmihd_30naaqs12
        dicttemp['AFIHD_35NAAQS12'] = afpmihd_35naaqs12
        dicttemp['AFIHD_40NAAQS12'] = afpmihd_40naaqs12
        dicttemp['AFIHD_45NAAQS12'] = afpmihd_45naaqs12
        dicttemp['AFIHD_50NAAQS12'] = afpmihd_50naaqs12
        dicttemp['AFIHD_55NAAQS12'] = afpmihd_55naaqs12
        dicttemp['AFIHD_60NAAQS12'] = afpmihd_60naaqs12
        dicttemp['AFIHD_65NAAQS12'] = afpmihd_65naaqs12
        dicttemp['AFIHD_70NAAQS12'] = afpmihd_70naaqs12
        dicttemp['AFIHD_75NAAQS12'] = afpmihd_75naaqs12
        dicttemp['AFIHD_80NAAQS12'] = afpmihd_80naaqs12
        dicttemp['AFIHD_85NAAQS12'] = afpmihd_85naaqs12
        dicttemp['AFIHD_90NAAQS12'] = afpmihd_90naaqs12
        dicttemp['AFIHD_95NAAQS12'] = afpmihd_95naaqs12
        dicttemp['AFST_25NAAQS12'] = afpmst_25naaqs12
        dicttemp['AFST_30NAAQS12'] = afpmst_30naaqs12
        dicttemp['AFST_35NAAQS12'] = afpmst_35naaqs12
        dicttemp['AFST_40NAAQS12'] = afpmst_40naaqs12
        dicttemp['AFST_45NAAQS12'] = afpmst_45naaqs12
        dicttemp['AFST_50NAAQS12'] = afpmst_50naaqs12
        dicttemp['AFST_55NAAQS12'] = afpmst_55naaqs12
        dicttemp['AFST_60NAAQS12'] = afpmst_60naaqs12
        dicttemp['AFST_65NAAQS12'] = afpmst_65naaqs12
        dicttemp['AFST_70NAAQS12'] = afpmst_70naaqs12
        dicttemp['AFST_75NAAQS12'] = afpmst_75naaqs12
        dicttemp['AFST_80NAAQS12'] = afpmst_80naaqs12
        dicttemp['AFST_85NAAQS12'] = afpmst_85naaqs12
        dicttemp['AFST_90NAAQS12'] = afpmst_90naaqs12
        dicttemp['AFST_95NAAQS12'] = afpmst_95naaqs12
        dicttemp['AFCOPDNAAQS12'] = afpmcopdnaaqs12
        dicttemp['AFLCNAAQS12'] = afpmlcnaaqs12
        dicttemp['AFDMNAAQS12'] = afpmdmnaaqs12
        dicttemp['AFLRINAAQS12'] = afpmlrinaaqs12
        dicttemp['AFIHD_25WHO10'] = afpmihd_25who10
        dicttemp['AFIHD_30WHO10'] = afpmihd_30who10
        dicttemp['AFIHD_35WHO10'] = afpmihd_35who10
        dicttemp['AFIHD_40WHO10'] = afpmihd_40who10
        dicttemp['AFIHD_45WHO10'] = afpmihd_45who10
        dicttemp['AFIHD_50WHO10'] = afpmihd_50who10
        dicttemp['AFIHD_55WHO10'] = afpmihd_55who10
        dicttemp['AFIHD_60WHO10'] = afpmihd_60who10
        dicttemp['AFIHD_65WHO10'] = afpmihd_65who10
        dicttemp['AFIHD_70WHO10'] = afpmihd_70who10
        dicttemp['AFIHD_75WHO10'] = afpmihd_75who10
        dicttemp['AFIHD_80WHO10'] = afpmihd_80who10
        dicttemp['AFIHD_85WHO10'] = afpmihd_85who10
        dicttemp['AFIHD_90WHO10'] = afpmihd_90who10
        dicttemp['AFIHD_95WHO10'] = afpmihd_95who10
        dicttemp['AFST_25WHO10'] = afpmst_25who10
        dicttemp['AFST_30WHO10'] = afpmst_30who10
        dicttemp['AFST_35WHO10'] = afpmst_35who10
        dicttemp['AFST_40WHO10'] = afpmst_40who10
        dicttemp['AFST_45WHO10'] = afpmst_45who10
        dicttemp['AFST_50WHO10'] = afpmst_50who10
        dicttemp['AFST_55WHO10'] = afpmst_55who10
        dicttemp['AFST_60WHO10'] = afpmst_60who10
        dicttemp['AFST_65WHO10'] = afpmst_65who10
        dicttemp['AFST_70WHO10'] = afpmst_70who10
        dicttemp['AFST_75WHO10'] = afpmst_75who10
        dicttemp['AFST_80WHO10'] = afpmst_80who10
        dicttemp['AFST_85WHO10'] = afpmst_85who10
        dicttemp['AFST_90WHO10'] = afpmst_90who10
        dicttemp['AFST_95WHO10'] = afpmst_95who10
        dicttemp['AFCOPDWHO10'] = afpmcopdwho10
        dicttemp['AFLCWHO10'] = afpmlcwho10
        dicttemp['AFDMWHO10'] = afpmdmwho10
        dicttemp['AFLRIWHO10'] = afpmlriwho10     
        dicttemp['AFIHD_25WHO5'] = afpmihd_25who5
        dicttemp['AFIHD_30WHO5'] = afpmihd_30who5
        dicttemp['AFIHD_35WHO5'] = afpmihd_35who5
        dicttemp['AFIHD_40WHO5'] = afpmihd_40who5
        dicttemp['AFIHD_45WHO5'] = afpmihd_45who5
        dicttemp['AFIHD_50WHO5'] = afpmihd_50who5
        dicttemp['AFIHD_55WHO5'] = afpmihd_55who5
        dicttemp['AFIHD_60WHO5'] = afpmihd_60who5
        dicttemp['AFIHD_65WHO5'] = afpmihd_65who5
        dicttemp['AFIHD_70WHO5'] = afpmihd_70who5
        dicttemp['AFIHD_75WHO5'] = afpmihd_75who5
        dicttemp['AFIHD_80WHO5'] = afpmihd_80who5
        dicttemp['AFIHD_85WHO5'] = afpmihd_85who5
        dicttemp['AFIHD_90WHO5'] = afpmihd_90who5
        dicttemp['AFIHD_95WHO5'] = afpmihd_95who5
        dicttemp['AFST_25WHO5'] = afpmst_25who5
        dicttemp['AFST_30WHO5'] = afpmst_30who5
        dicttemp['AFST_35WHO5'] = afpmst_35who5
        dicttemp['AFST_40WHO5'] = afpmst_40who5
        dicttemp['AFST_45WHO5'] = afpmst_45who5
        dicttemp['AFST_50WHO5'] = afpmst_50who5
        dicttemp['AFST_55WHO5'] = afpmst_55who5
        dicttemp['AFST_60WHO5'] = afpmst_60who5
        dicttemp['AFST_65WHO5'] = afpmst_65who5
        dicttemp['AFST_70WHO5'] = afpmst_70who5
        dicttemp['AFST_75WHO5'] = afpmst_75who5
        dicttemp['AFST_80WHO5'] = afpmst_80who5
        dicttemp['AFST_85WHO5'] = afpmst_85who5
        dicttemp['AFST_90WHO5'] = afpmst_90who5
        dicttemp['AFST_95WHO5'] = afpmst_95who5
        dicttemp['AFCOPDWHO5'] = afpmcopdwho5
        dicttemp['AFLCWHO5'] = afpmlcwho5
        dicttemp['AFDMWHO5'] = afpmdmwho5
        dicttemp['AFLRIWHO5'] = afpmlriwho5
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
    df.to_csv(DIR_OUT+'asthma_af%s_acs%s_%s_v5.csv'%(no2year, vintage, 
        statefips), sep = ',')
    print('# # # # Output file written!\n', file=f)
    return 

vintages = ['2015-2019', '2013-2017', '2006-2010', '2007-2011', '2008-2012', 
    '2009-2013', '2010-2014', '2011-2015', '2012-2016', '2014-2018']  
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