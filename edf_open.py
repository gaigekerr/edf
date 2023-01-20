#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:42:12 2021

@author: ghkerr
"""
# Local environment
DIR = '/Users/ghkerr/GW/data/'
DIR_NO2 = DIR+'data/no2/'
DIR_PM25 = '/Users/ghkerr/Downloads/'
# DIR_POP = DIR+'data/population/'
DIR_AQS = '/Users/ghkerr/GW/data/aq/aqs/'
DIR_TROPOMI = '/Users/ghkerr/GW/data/'
DIR_HARM = DIR+'anenberg_mohegh_no2/harmonizedtables/'
DIR_FIG = '/Users/ghkerr/Desktop/'
# # Pegasus
# DIR = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'
# DIR_NO2 = DIR+'no2/'
# DIR_POP = DIR+'population/'
# DIR_FIG = '/GWSPH/groups/anenberggrp/ghkerr/data/edf/'

def pixel2coord(col, row, a, b, c, d, e, f):
    """Returns global coordinates to pixel center using base-0 raster 
    index. Adapted from https://gis.stackexchange.com/questions/53617/
    how-to-find-lat-lon-values-for-every-pixel-in-a-geotiff-file"""
    xp = a* col + b * row + a * 0.5 + b * 0.5 + c
    yp = d* col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)

# def open_no2pop_tif(fname, fill, ftype, clip=None): 
#     """Open TIF dataset for the United States containing 
#     and extract coordinate information for the specified domain. 
    
#     Parameters
#     ----------
#     fname : str
#     fill : float
#     ftype : str
#         "NO2" or "pop"
#     clip : list, optional
#         Coordinates (x0, x1, y0, y1) to which the Larkin NO2 dataset will 
#         be clipped.

#     Returns
#     -------
#     lng : numpy.ndarray
#         Longitude array for Larkin dataset, units of degrees, [lng,]
#     lat : numpy.ndarray
#         Latitude array for Larkin dataset, units of degrees, [lat,]    
#     larkin : numpy.ndarray
#         Larkin surface-level NO2, units of ppbv, [lat, lng]    
#     """
#     from osgeo import gdal
#     import numpy as np
#     if ftype=='NO2':
#         DIR = DIR_NO2
#     if ftype=='pop':
#         DIR = DIR_POP
#     # For multiple years/files
#     if type(fname) is list:
#         no2 = []
#         for f in fname: 
#             ds = gdal.Open(DIR+'%s.tif'%f)
#             # Note GetRasterBand() takes band no. starting from 1 not 0
#             band = ds.GetRasterBand(1)
#             no2f = band.ReadAsArray()
#             # Unravel GDAL affine transform parameters
#             c, a, b, f, d, e = ds.GetGeoTransform()
#             # Dimensions
#             col = ds.RasterXSize
#             row = ds.RasterYSize        
#             no2.append(no2f)
#     no2 = np.stack(no2)
#     # For single year/files
#     else: 
#         ds = gdal.Open(DIR+'%s.tif'%fname)
#         band = ds.GetRasterBand(1)
#         no2 = band.ReadAsArray()
#         c, a, b, f, d, e = ds.GetGeoTransform()
#         col = ds.RasterXSize
#         row = ds.RasterYSize
#     # Fetch latitudes
#     lat = []
#     for ri in range(row):
#         coord = pixel2coord(0, ri, a, b, c, d, e, f) # Can substitute 
#         # whatever for 0 and it should yield the name answer. 
#         lat.append(coord[1])
#     lat = np.array(lat)
#     # Fetch longitudes
#     lng = []
#     for ci in range(col):
#         coord = pixel2coord(ci, 0, a, b, c, d, e, f)
#         lng.append(coord[0])
#     lng = np.array(lng)
#     if clip is not None: 
#         latb_down = np.abs(lat-clip[3]).argmin() # Note that since the raster
#         # is reflected/reserved, the "top latitude" is actually lower than
#         # the bottom latitude
#         latb_top = np.abs(lat-clip[2]).argmin()
#         lngb_left = np.abs(lng-clip[0]).argmin()
#         lngb_right = np.abs(lng-clip[1]).argmin()
#         # Clip domain and coordinates
#         lat = lat[latb_down:latb_top+1]
#         lng = lng[lngb_left:lngb_right+1]
#         if type(fname) is list: 
#             no2 = no2[:, latb_down:latb_top+1,lngb_left:lngb_right+1]
#         else: 
#             no2 = no2[latb_down:latb_top+1,lngb_left:lngb_right+1]
#     # Convert from uint8 to float
#     no2 = no2.astype(np.float)
#     # Replace fill value with NaN
#     no2[no2==fill]=np.nan
#     if type(fname) is list:
#         no2 = np.nanmean(no2, axis=0)
#     return lng, lat, no2

def open_cooperno2(clip=None):
    """Open surface-level concentrations of NO2 from Cooper et al. (2020) 
    dataset for the specified domain. 

    Parameters
    ----------
    clip : list, optional
        Coordinates (x0, x1, y0, y1) to which the Larkin NO2 dataset will 
        be clipped.

    Returns
    -------
    lng : numpy.ndarray
        Longitude, units of degrees, [lng,]
    lat : numpy.ndarray
        Latitude, units of degrees, [lat,]    
    no2 : numpy.ndarray
        Early afternoon (13:00-15:00 hours local time) surface-level NO2, 
        units of ppbv, [lat, lng]    

    References
    ----------
    Cooper, M. J., et al 2020. Inferring ground-level nitrogen dioxide 
    concentrations at fine spatial resolution applied to the TROPOMI satellite 
    instrument. Environ. Res. Lett. in press. 
    https://doi.org/10.1088/1748-9326/aba3a5
    
    """
    import numpy as np
    import h5py
    no2 = h5py.File(DIR_TROPOMI+
        'tropomi_surface_no2_0p025x0p03125_northamerica.h5', 'r')
    lat = no2['LAT_CENTER'][0,:]
    lng = no2['LON_CENTER'][:,0]
    no2 = no2['surface_NO2_2019'][:]
    # The x,y coordinates of Cooper et al. (2020) are a little wonky, so 
    # rotate and flip
    no2 = np.flipud(np.rot90((no2)))
    if clip is not None: 
        latb_top = np.abs(lat-clip[3]).argmin()
        latb_down = np.abs(lat-clip[2]).argmin()
        lngb_left = np.abs(lng-clip[0]).argmin()
        lngb_right = np.abs(lng-clip[1]).argmin()
        # Clip domain and coordinates
        lat = lat[latb_down:latb_top+1]
        lng = lng[lngb_left:lngb_right+1]
        no2 = no2[latb_down:latb_top+1,lngb_left:lngb_right+1]
    return lng, lat, no2
            
def read_aqs_amean(year):
    """Open annual mean NO2 concentrations at AQS monitors.

    Parameters
    ----------
    year : int
        Year of interest (note that as of 27 April 2021, only annual 
        concentrations by monitors had been downloaded for 2019. Additional 
        years can be downloaded at under "Annual Summary Data" at 
        https://aqs.epa.gov/aqsweb/airdata/download_files.html).

    Returns
    -------
    aqs : pandas.core.frame.DataFrame
        Annual mean NO2 concentrations at AQS sites including metadata on
        siting/geographic indicators.
    """
    import pandas as pd
    aqs = pd.read_csv(DIR_AQS+'annual_conc_by_monitor_%d.csv'%year, sep=',',
        engine='python')
    # Select NO2 monitors (Parameter Code = 42602)
    aqs = aqs.loc[aqs['Parameter Code']==42602]
    # Selecting the parameter code associated with NO2 above will yield
    # two different values for each site. One has "Pollutant Standard" = 
    # "NO2 1-hour" and the other has "Pollutant Standard" = "NO2 Annual
    # 1971". Select the "NO2 Annual 1971" values as they appear to be 
    # representing the annual average 
    aqs = aqs.loc[aqs['Pollutant Standard']=='NO2 Annual 1971']
    # Strip off only columns of interest. Note that all measurements are in
    # ppbv (so no need to include 'Units of Measure'). Gaige note: it might be
    # useful to look at 'Method Name' to see if the Chemiluminescence issues 
    # with molybdenum show up
    keep = ['State Code', 'County Code', 'Latitude', 'Longitude', 
        'Year', 'Observation Percent', 'Method Name',
        'Arithmetic Mean', 'State Name', 'County Name', 'City Name']
    aqs = aqs[keep]
    # Exclude sites in Alaska, Puerto Rico 
    aqs = aqs.loc[~aqs['State Name'].isin(['Alaska', 'Puerto Rico', 
        'Hawaii'])]
    return aqs

def read_aqs_hourly(year):
    """Read hourly AQS NO2 concentrations and subsample during the early 
    afternoon to be consistent with NO2 dataset detailed in Cooper et al. 
    (2020). 
    
    Parameters
    ----------
    year : int
        Year of interest    

    Returns
    -------
    aqs : pandas.core.frame.DataFrame
        Annual mean early afternoon (13-15 hours local time) NO2 concentrations 
        at AQS sites including metadata on siting/geographic indicators.    
        
    References
    ----------
    Cooper, M. J., et al 2020. Inferring ground-level nitrogen dioxide 
    concentrations at fine spatial resolution applied to the TROPOMI satellite 
    instrument. Environ. Res. Lett. in press. 
    https://doi.org/10.1088/1748-9326/aba3a5
    """
    import pandas as pd
    aqs = pd.read_csv(DIR_AQS+'hourly_42602_%s.csv'%year, sep=',', 
        engine='python')
    # Drop unneeded columns; drop latitude/longitude coordinates for 
    # temperature observations as the merging of the O3 and temperature 
    # DataFrames will supply these coordinates 
    to_drop = ['Parameter Code', 'POC', 'Datum', 'Parameter Name',
        'Date GMT', 'Time GMT', 'Units of Measure', 'MDL', 'Uncertainty', 
        'Qualifier', 'Method Type', 'Method Code', 'Date of Last Change',
        'Site Num']
    aqs = aqs.drop(to_drop, axis=1)
    # Select months in measuring period (can finetune dates if the whole
    # year isn't desired)
    date_start, date_end = '2019-01-01', '2019-12-31'
    aqs = aqs.loc[pd.to_datetime(aqs['Date Local']).isin(
        pd.date_range(date_start,date_end))]
    # Extract the early afternoon times of 13:00-15:00 hours from Cooper et 
    # al. (2020), which correspond to the time of satellite overpass
    aqs = aqs.loc[pd.to_datetime(aqs['Time Local']).isin(['13:00',
        '14:00','15:00'])]
    # Exclude sites in Alaska, Puerto Rico 
    aqs = aqs.loc[~aqs['State Name'].isin(['Alaska', 'Puerto Rico', 
        'Hawaii'])]
    aqs = aqs.groupby(['Latitude', 'Longitude']).mean()
    aqs = aqs.reset_index()
    return aqs

def load_vintageharmonized(vintage):
    """For a given ACS 5-year estimate, load harmonized demographic-attributable
    fraction files for all states in the U.S. and the District of Columbia and 
    Puerto Rico. 

    Parameters
    ----------
    vintage : str
        Years corresponding to the 5-year ACS estimates.

    Returns
    ------
    harm : pandas.core.frame.DataFrame
        Harmonized demographic-attributable fractions for a given 5-year ACS
        estimate.
    """
    import os
    import pandas as pd
    vintage_files = [DIR_HARM+f for f in os.listdir(DIR_HARM) if 
        vintage in f]
    def load_files(filenames):
        """code adapted from https://pandasninja.com/2019/04/
        how-to-read-lots-of-csv-files-easily-into-pandas/
        """
        for filename in filenames:
            yield pd.read_csv(filename)
    harm = pd.concat(load_files(vintage_files))
    # For states with FIPS codes 0-9, there is no leading zero in their 
    # GEOID row, so add one to ensure GEOIDs are identical lengths
    harm['GEOID'] = harm['GEOID'].map(lambda x: f'{x:0>11}')
    # Make GEOID a string and index row 
    harm = harm.set_index('GEOID')
    # Remove tracts where MISSINGTRACTFLAG == 1; these tracts have no 
    # demographic information (i.e., population) and will thus cause an error
    # when calculating asthma burdens
    harm.drop(harm[harm['MISSINGTRACTFLAG']==1.].index, inplace = True)
    # Also remove tracts where NESTEDTRACTFLAG == 1
    harm.drop(harm[harm['NESTEDTRACTFLAG']==1.].index, inplace = True)
    harm.index = harm.index.map(str)
    harm.loc[harm['STATE']=='District Of Columbia', 'STATE'] = \
        'District of Columbia'
    return harm