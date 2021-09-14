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
v3 - Add PM2.5 and its components to tract-level averaging computation. 
   - Fix summation over tracts that have > 1 ACS entry associated with them. 
   - Fix IDW regression function to handle missing values.

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
DIR_PM25 = '/GWSPH/groups/anenberggrp/ghkerr/data/pm25/'
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

def open_V4NAO3(year, checkplot=False):
    """Open North American Regional Estimates (V4.NA.03) of ground-level fine 
    particulate matter (PM2.5) total and compositional mass concentrations
    for year of interest at 0.01˚ x 0.01˚; data can be found at 
    https://sites.wustl.edu/acag/datasets/surface-pm2-5/
    
    Parameters
    ----------
    year : int
        Year of interest
        
    Returns
    -------
    lat : numpy.ma.core.MaskedArray
        Latitude, units of degrees north, [lat,]
    lng : numpy.ma.core.MaskedArray
        Longitude, units of degrees east, [lng,]
    total : numpy.ma.core.MaskedArray
        Total PM2.5 mass, units of µg m-3, [lat, lng]
    bc : numpy.ma.core.MaskedArray
        Contribution of black carbon, units of %, [lat, lng]
    nh4 : numpy.ma.core.MaskedArray
        Contribution of ammonium, units of %, [lat, lng]
    nit : numpy.ma.core.MaskedArray
         Contribution of nitrate, units of %, [lat, lng]
    om : numpy.ma.core.MaskedArray
        Contribution of organic matter, units of %, [lat, lng]
    so4 : numpy.ma.core.MaskedArray
        Contribution of sulfate, units of %, [lat, lng]
    soil : numpy.ma.core.MaskedArray
        Contribution of soil/crustal particulates, units of %, [lat, lng]
    ss : numpy.ma.core.MaskedArray
        Contribution of sea salt particulates, units of %, [lat, lng]
    
    References
    ----------
    Hammer, M. S.; van Donkelaar, A.; Li, C.; Lyapustin, A.; Sayer, A. M.; Hsu, 
        N. C.; Levy, R. C.; Garay, M. J.; Kalashnikova, O. V.; Kahn, R. A.; 
        Brauer, M.; Apte, J. S.; Henze, D. K.; Zhang, L.; Zhang, Q.; Ford, B.; 
        Pierce, J. R.; and Martin, R. V., Global Estimates and Long-Term Trends 
        of Fine Particulate Matter Concentrations (1998-2018)., Environ. Sci. 
        Technol, doi: 10.1021/acs.est.0c01764, 2020. 
    van Donkelaar, A., R. V. Martin, et al. (2019). Regional Estimates of 
        Chemical Composition of Fine Particulate Matter using a Combined 
        Geoscience-Statistical Method with Information from Satellites, Models, 
        and Monitors. Environmental Science & Technology, 2019, 
        doi:10.1021/acs.est.8b06392.
    """
    def open_V4NA03_species(species, year):
        """Open Dalhousie/WUSTL V4.NA.03 total or composition PM2.5 mass using 
        Geographically Weighted Regression for year of interest; note that for total 
        PM2.5 mass, species='PM25'. For individual component percentages/mass 
        contribution species='BC','NH4','NIT','OM','SO4','SOIL', or 'SS'.
    
        Parameters
        ----------
        species : str
            PM2.5 component of interest
        year : int
            Year of interest
    
        Returns
        -------
        lat : numpy.ma.core.MaskedArray
            Latitude, units of degrees north, [lat,]
        lng : numpy.ma.core.MaskedArray
            Longitude, units of degrees east, [lng,]
        pm25 : numpy.ma.core.MaskedArray
            Total PM2.5 mass or percentage contribution from individual components 
            to total mass, units of µg m-3 or %, [lat, lng]
        """
        import glob
        import netCDF4 as nc
        fname = glob.glob(DIR_PM25+'*%s*/*%s*%s*.nc'%(species.upper(), 
            species.upper(), year))
        pm25 = nc.Dataset(fname[0], 'r')
        lat = pm25.variables['LAT'][:]
        lng = pm25.variables['LON'][:]
        pm25 = pm25.variables['%s'%species.upper()][:]
        return lat, lng, pm25
    # Load total PM2.5 mass and component contribution 
    lat, lng, total = open_V4NA03_species('PM25', year)
    lat, lng, bc = open_V4NA03_species('bc', year)
    lat, lng, nh4 = open_V4NA03_species('nh4', year)
    lat, lng, nit = open_V4NA03_species('nit', year)
    lat, lng, om = open_V4NA03_species('om', year)
    lat, lng, so4 = open_V4NA03_species('so4', year)
    lat, lng, soil = open_V4NA03_species('soil', year)
    lat, lng, ss = open_V4NA03_species('ss', year)
    if checkplot==True:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import cartopy.crs as ccrs
        fig = plt.figure(figsize=(9,7))
        ax1 = plt.subplot2grid((3,3),(0,0), projection=ccrs.LambertConformal())
        ax2 = plt.subplot2grid((3,3),(0,1), projection=ccrs.LambertConformal())
        ax3 = plt.subplot2grid((3,3),(0,2), projection=ccrs.LambertConformal())
        ax4 = plt.subplot2grid((3,3),(1,0), projection=ccrs.LambertConformal())
        ax5 = plt.subplot2grid((3,3),(1,1), projection=ccrs.LambertConformal())
        ax6 = plt.subplot2grid((3,3),(1,2), projection=ccrs.LambertConformal())
        ax7 = plt.subplot2grid((3,3),(2,0), projection=ccrs.LambertConformal())
        ax8 = plt.subplot2grid((3,3),(2,1), projection=ccrs.LambertConformal())
        ax9 = plt.subplot2grid((3,3),(2,2), projection=ccrs.LambertConformal())
        # Total mass
        p1 = ax1.pcolormesh(lng[::5], lat[::5], total[::5,::5], vmin=0, vmax=15)
        ax1.set_title('Total mass [$\mu$g m$^{-3}$]')
        # BC contribution
        p2 = ax2.pcolormesh(lng[::5], lat[::5], bc[::5,::5], vmin=0, vmax=25)
        ax2.set_title('BC [%]')
        # NH4 contribution
        p3 = ax3.pcolormesh(lng[::5], lat[::5], nh4[::5,::5], vmin=0, vmax=25)
        ax3.set_title('NH4 [%]')
        # NIT contribution
        p4 = ax4.pcolormesh(lng[::5], lat[::5], nit[::5,::5], vmin=0, vmax=25)
        ax4.set_title('NIT [%]')
        # OM contribution
        p5 = ax5.pcolormesh(lng[::5], lat[::5], om[::5,::5], vmin=0, vmax=25)
        ax5.set_title('OM [%]')
        # SO4 contribution
        p6 = ax6.pcolormesh(lng[::5], lat[::5], so4[::5,::5], vmin=0, vmax=25)
        ax6.set_title('SO4 [%]')
        # SOIL contribution
        p7 = ax7.pcolormesh(lng[::5], lat[::5], soil[::5,::5], vmin=0, vmax=25)
        ax7.set_title('SOIL [%]')
        # SS contribution
        p8 = ax8.pcolormesh(lng[::5], lat[::5], ss[::5,::5], vmin=0, vmax=25)
        ax8.set_title('SS [%]')
        # Total contribution
        p9 = ax9.pcolormesh(lng[::5], lat[::5], (bc+nh4+nit+om+so4+soil+ss
            ).data[::5,::5], vmin=99, vmax=101, cmap=plt.get_cmap('bwr'))
        ax9.set_title('Total contribution [%]')
        # Add colorbars, set extent
        for mb, ax in zip([p1, p2, p3, p4, p5, p6, p7, p8, p9], 
            [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=0.15, 
                axes_class=plt.Axes)
            fig.colorbar(mb, cax=cax, spacing='proportional', 
                orientation='horizontal', extend='max')
            ax.set_extent([-125, -66.5, 20, 50], ccrs.LambertConformal())
            ax.coastlines()
        plt.savefig(DIR_FIG+'checkplot_pm25na_wustl_%s.png'%year, dpi=500)
        plt.show()
    return lat, lng, total, bc, nh4, nit, om, so4, soil, ss

def harmonize_afacs(vintage, statefips, formpm=False):
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
    formpm : bool, optional
        Flag to trigger computation of tract level PM2.5 (and contribution of 
        components to total PM2.5 mass) 

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
    if formpm==True: 
        (lat_pm, lng_pm, total, bc, nh4, nit, om, so4, soil, ss) = \
            open_V4NAO3(no2year)
        print('# # # # PM2.5 data loaded!', file=f)
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
        else: 
            af = np.nan
            afupper = np.nan
            aflower = np.nan
            afgbdmean = np.nan
            afgbdmed = np.nan
            afgbdupper = np.nan
            afgbdlower  = np.nan
        dicttemp = {'GEOID':geoid, 
            'NO2':no2_inside,
            'AFPA':af,
            'AFPAUPPER':afupper,
            'AFPALOWER':aflower,
            'AFPAMEAN_GBD':afgbdmean,
            'AFPAMEDIAN_GBD':afgbdmed,
            'AFPAUPPER_GBD':afgbdupper,            
            'AFPALOWER_GBD':afgbdlower,            
            'INTERPFLAG':np.nanmean(interpflag),
            'NESTEDTRACTFLAG':nestedtract,
            'LAT_CENTROID':lat_tract,
            'LNG_CENTROID':lng_tract}
        
        # # # # If specified, fetch tract-level PM2.5 estimates
        if formpm==True: 
            # Subset
            upperp = geo_idx(lat_tract-searchrad, lat_pm)
            lowerp = geo_idx(lat_tract+searchrad, lat_pm)
            leftp = geo_idx(lng_tract-searchrad, lng_pm)
            rightp = geo_idx(lng_tract+searchrad, lng_pm)
            lat_pm_subset = lat_pm[lowerp:upperp]
            lng_pm_subset = lng_pm[leftp:rightp] 
            pm_subset = total[lowerp:upperp, leftp:rightp]
            bc_subset = bc[lowerp:upperp, leftp:rightp]
            nh4_subset = nh4[lowerp:upperp, leftp:rightp]
            nit_subset = nit[lowerp:upperp, leftp:rightp]
            om_subset = om[lowerp:upperp, leftp:rightp]
            so4_subset = so4[lowerp:upperp, leftp:rightp]
            soil_subset = soil[lowerp:upperp, leftp:rightp]
            ss_subset = ss[lowerp:upperp, leftp:rightp] 
            # Find PM25 and its components in tracts
            pm_inside, bc_inside, nh4_inside, nit_inside = [], [], [], []
            om_inside, so4_inside, soil_inside, ss_inside = [], [], [], []
            pminterpflag = []
            for i, ilat in enumerate(lat_pm_subset):
                for j, jlng in enumerate(lng_pm_subset): 
                    point = Point(jlng, ilat)
                    if tract.contains(point) is True:
                        pm_inside.append(pm_subset[i,j])
                        bc_inside.append(bc_subset[i,j])
                        nh4_inside.append(nh4_subset[i,j])
                        nit_inside.append(nit_subset[i,j])
                        om_inside.append(om_subset[i,j])
                        so4_inside.append(so4_subset[i,j])
                        soil_inside.append(soil_subset[i,j])
                        ss_inside.append(ss_subset[i,j])
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
                    z = bc_subset[lat_idx, lng_idx]
                    bc_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                    z = nh4_subset[lat_idx, lng_idx]
                    nh4_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                    z = nit_subset[lat_idx, lng_idx]
                    nit_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                    z = om_subset[lat_idx, lng_idx]
                    om_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                    z = so4_subset[lat_idx, lng_idx]
                    so4_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                    z = soil_subset[lat_idx, lng_idx]
                    soil_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                    z = ss_subset[lat_idx, lng_idx]
                    ss_inside.append(idwr(x,y,z,[lng_tract],[lat_tract])[0][-1])
                    pminterpflag.append(1.)
            # # # # PM2.5 health impact assessment                    
            pm_inside = np.nanmean(pm_inside)
            bc_inside = np.nanmean(bc_inside)
            nh4_inside = np.nanmean(nh4_inside) 
            nit_inside = np.nanmean(nit_inside)
            om_inside = np.nanmean(om_inside)
            so4_inside = np.nanmean(so4_inside) 
            soil_inside = np.nanmean(soil_inside) 
            ss_inside = np.nanmean(ss_inside) 
            if np.isnan(pm_inside)!=True:
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
            else: 
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
            dicttemp['PM25'] = pm_inside
            dicttemp['PM25_BC'] = bc_inside
            dicttemp['PM25_NH4'] = nh4_inside
            dicttemp['PM25_NIT'] = nit_inside
            dicttemp['PM25_OM'] = om_inside
            dicttemp['PM25_SO4'] = so4_inside
            dicttemp['PM25_SOIL'] = soil_inside
            dicttemp['PM25_SS'] = ss_inside
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
    df.to_csv(DIR_OUT+'asthma_af%s_acs%s_%s_v3.csv'%(no2year, vintage, 
        statefips), sep = ',')
    print('# # # # Output file written!\n', file=f)
    return 

vintages = ['2006-2010', '2007-2011', '2008-2012', 
    '2009-2013', '2010-2014', '2011-2015', '2012-2016', '2013-2017', 
    '2014-2018', '2015-2019']    
fips = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', 
    '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
    '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
    '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', 
    '48', '49', '50', '51', '53', '54', '55', '56', '72']
for vintage in vintages: 
    # Conduct PM2.5-based health impact assessment for years with PM2.5 data
    if int(vintage[-4:])<=2017:
        formpm = True
    else: 
        formpm = False
    # Create output text file for print statements (i.e., crosswalk checks)
    f = open(DIR_OUT+'harmonize_afacs%s_%s.txt'%(vintage,
        datetime.now().strftime('%Y-%m-%d-%H%M')), 'a')    
    for statefips in fips: 
        harmonize_afacs(vintage, statefips, formpm=formpm)
    f.close()