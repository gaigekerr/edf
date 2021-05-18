#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 22:26:10 2021

@author: ghkerr
"""
DIR = '/Users/ghkerr/GW/'
DIR_AQ = DIR+'data/aq/'
DIR_LAND = '/Users/ghkerr/GW/edf/data/population/GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K_V2_0/'
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
    
def pixel2coord(col, row, a, b, c, d, e, f):
    """Returns global coordinates to pixel center using base-0 raster 
    index. Adapted from https://gis.stackexchange.com/questions/53617/
    how-to-find-lat-lon-values-for-every-pixel-in-a-geotiff-file"""
    xp = a* col + b * row + a * 0.5 + b * 0.5 + c
    yp = d* col + e * row + d * 0.5 + e * 0.5 + f
    return(xp, yp)

# "*DESCRIPTION"
# "*1HR"
# "*8HR"
# "*24HR"
# "*DMAX1HR"
# "*DAILYMEAN"
# "********************************************DESCRIPTION*********************************************"
# "File generated on",2020-11-30
# ,
# "Annual Summary Report",
# "Pollutant: NO2",
# "Units: ppb",
# "Day: from midnight to midnight.",
# "All data reported in standard local time",
# ,
# "Reporting Criteria for 1hr report:",
# "Minimum, Maximum and Percentiles are reported regardless of data completeness",
# "Monthly averages are reported if at least 50% of the hours during the month have valid measurements",
# "Annual average and standard deviation are reported if 50% of hours in year are valid and each quarter has at least 2 valid months",
# ,
# "Reporting Criteria for 8hr report:",
# "Running 8 hour averages are calculated with reporting hour being end hour (eg. 8pm = average of 1pm to 8pm)",
# "8 hour averages are valid if at least 6 of the 8 hours are valid",
# "Minimum, Maximum and Percentiles are reported regardless of data completeness",
# "Monthly averages are reported if at least 50% of the hours during the month have valid running 8-hour average",
# "Annual average and standard deviation are reported if 50% of running 8-hour averages are valid and each quarter has at least 2 valid months",
# ,
# "Reporting Criteria for 24hr report:",
# "Running 24 hour averages are calculated with reporting hour being end hour (eg. 8pm = 24 hour average from 9pm previous day to 8pm)",
# "24 hour averages are valid if at least 18 of the 24 hours are valid",
# "Minimum, Maximum and Percentiles are reported regardless of data completeness",
# "Monthly averages are reported if at least 50% of the hours during the month have valid running 24-hour average",
# "Annual average and standard deviation are reported if 50% of running 24-hour averages are valid and each quarter has at least 2 valid months",
# ,
# "Reporting Criteria for DMax1hr report:",
# "Daily max 1-hour only valid if at least 18 of 24 hours have valid data",
# "Minimum, Maximum and Percentiles are reported regardless of data completeness",
# "Monthly averages are reported if at least 50% days have a valid daily 1-hr max",
# "Annual average and standard deviation are reported if 50% of days have valid daily 1-hr max and each quarter has at least 2 valid months",
# ,
# "Reporting Criteria for DailyMean report:",
# "Daily means are valid if at least 18 of the 24 hours are valid",
# "Minimum, Maximum and Percentiles are reported regardless of data completeness",
# "Monthly averages are reported if at least 50% days have a valid daily mean",
# "Annual average and standard deviation are reported if 50% of days have daily mean and each quarter has at least 2 valid months",
# "************************************************1HR*************************************************"

import numpy as np
import pandas as pd
import sys
sys.path.append(DIR+'edf/')
import edf_open
sys.path.append('/Users/ghkerr/phd/utils/')
from geo_idx import geo_idx

# # # # Open Anenberg, Mohegh et al 1 km NO2 for 2019
lng_am, lat_am, no2_am_2019 = edf_open.open_no2pop_tif(
    'arash_native/2019_final_1km', -999., 'NO2')

# # # # Open GHS SMOD data from  https://ghsl.jrc.ec.europa.eu/
# download.php?ds=smod. Note that the raw data were transformed to a
# different projection (i.e., the projection of Arash's native NO2 dataset)
# using instructions at the following: https://gis.stackexchange.com/
# questions/239013/reprojecting-a-raster-to-match-another-raster-using-gdal/239016
# or listed here: 
# gdalsrsinfo -o wkt 2019_final_1km.tif > target.wkt
# gdalwarp -t_srs target.wkt GHS_SMOD_POP2015_GLOBE_R2019A_54009_1K_V2_0.tif trial.tif
from osgeo import gdal, osr
ds = gdal.Open(DIR_LAND+'SMOD_reproj.tif')
band = ds.GetRasterBand(1)
smod = band.ReadAsArray()
c, a, b, f, d, e = ds.GetGeoTransform()
col = ds.RasterXSize
row = ds.RasterYSize
# Fetch latitudes
lat_smod = []
for ri in range(row):
    coord = pixel2coord(0, ri, a, b, c, d, e, f) # Can substitute 
    # whatever for 0 and it should yield the name answer. 
    lat_smod.append(coord[1])
lat_smod = np.array(lat_smod)
# Fetch longitudes
lng_smod = []
for ci in range(col):
    coord = pixel2coord(ci, 0, a, b, c, d, e, f)
    lng_smod.append(coord[0])
lng_smod = np.array(lng_smod)
# Convert from uint8 to float
smod = smod.astype(np.float)
# Replace fill value with NaN
smod[smod==-200.]=np.nan

# # # # Open NAPS (date accessed 11 May 2021)
# To download NAPS, visit https://data-donnees.ec.gc.ca/data/air/monitor/
# national-air-pollution-surveillance-naps-program/?lang=en
# Then click on the following: Data-Donnees -> 2019 -> 
# ContinuousData-DonneesContinu -> AnnualSummaries-SommairesAnnuels ->
# 2019_AnnualNO2_EN.csv (note that the output was copy and pasted to a 
# text file and the preamble text was removed) 
naps_2019 = pd.read_csv(DIR_AQ+'naps/naps_2019.csv', sep=',', engine='python')
# Find and subset daily mean entries
wheredm = '*********************************************DAILYMEAN'+\
    '**********************************************'
wheredm = naps_2019.loc[naps_2019['NAPS ID']==wheredm].index[0]
naps_2019 = naps_2019.loc[wheredm+2:]
# Change relevant column values from strings to floats
naps_2019['Latitude'] = naps_2019['Latitude'].astype(float)
naps_2019['Longitude'] = naps_2019['Longitude'].astype(float)
naps_2019['Mean'] = naps_2019['Mean'].astype(float)

# # # # Open AQS (date accessed 24 November 2020)
aqs_2019 = edf_open.read_aqs_amean(2019)

# # # # Open EEA (date accessed 11 May 2021)
# To download EEA, visit https://www.eea.europa.eu/data-and-maps/data/
# aqereporting-8. Then click on “Download file” under 
# “Air quality annual statistics (AIDE F)“. Note that Air quality annual 
# statistics calculated by the EEA. The annual aggregated air quality 
# values have been calculated by EEA based on the primary observations 
# (time series) uploaded by countries into CDR and successfully tested 
# by automated QC. For a shortcut to the data, see the following: 
# http://aidef.apps.eea.europa.eu/?source=%7B%22query%22%3A%7B%22match
# _all%22%3A%7B%7D%7D%2C%22display_type%22%3A%22tabular%22%7D) 
eea_2019 = pd.read_csv(DIR_AQ+'eea/data.csv')
# Select NO2 observations for year of interest
eea_2019 = eea_2019.loc[
    (eea_2019['Pollutant']=='Nitrogen dioxide (air)') & 
    (eea_2019['BeginPosition']=='2019-01-01') & 
    (eea_2019['EndPosition']=='2020-01-01') &
    (eea_2019['AggregationType']=='Annual mean / 1 calendar year')]
# There are a small number of stations (~30) that have two observations 
# at a given latitude/longitue coordinate (e.g., see StationLocalId = 
# STA.DE_DENW034). Average over these observations. 
eea_2019 = eea_2019.groupby(by=['SamplingPoint_Longitude', 
    'SamplingPoint_Latitude']).mean().reset_index()
# Convert μg/m3 to ppb. From https://www2.dmu.dk/AtmosphericEnvironment/
# Expost/database/docs/PPM_conversion.pdf
# NO2 1 ppb = 1.88 μg/m3 
eea_2019['AQValue'] = eea_2019['AQValue']/1.88
# Remove outliers (there is one station where NO2 is 3037 ppb)
eea_2019.loc[eea_2019['AQValue']>500, 'AQValue'] == np.nan

# # # # Find co-located Anenberg, Mohegh, et al. NO2 for each station 
# location
aqs_2019[['AM NO2', 'SMOD']] = np.nan
for index, row in aqs_2019.iterrows():
    lat_aqs = row['Latitude']
    lng_aqs = row['Longitude']
    # Co-located grid cell for NO2
    lat_idx = (np.abs(lat_am-lat_aqs)).argmin()
    lng_idx = (np.abs(lng_am-lng_aqs)).argmin()
    no2_ataqs = no2_am_2019[lat_idx, lng_idx]
    aqs_2019.loc[index, 'AM NO2'] = no2_ataqs
    # Co-located grid cell for SMOD
    lat_idx = (np.abs(lat_smod-lat_aqs)).argmin()
    lng_idx = (np.abs(lng_smod-lng_aqs)).argmin()
    smod_ataqs = smod[lat_idx, lng_idx]
    aqs_2019.loc[index, 'SMOD'] = smod_ataqs
naps_2019[['AM NO2', 'SMOD']] = np.nan
for index, row in naps_2019.iterrows():
    lat_naps = row['Latitude']
    lng_naps = row['Longitude']
    lat_idx = (np.abs(lat_am-lat_naps)).argmin()
    lng_idx = (np.abs(lng_am-lng_naps)).argmin()
    no2_atnaps = no2_am_2019[lat_idx, lng_idx]
    naps_2019.loc[index, 'AM NO2'] = no2_atnaps
    lat_idx = (np.abs(lat_smod-lat_naps)).argmin()
    lng_idx = (np.abs(lng_smod-lng_naps)).argmin()
    smod_atnaps = smod[lat_idx, lng_idx]
    naps_2019.loc[index, 'SMOD'] = smod_atnaps    
eea_2019[['AM NO2', 'SMOD']] = np.nan
for index, row in eea_2019.iterrows():
    lat_eea = row['SamplingPoint_Latitude']
    lng_eea = row['SamplingPoint_Longitude']
    lat_idx = (np.abs(lat_am-lat_eea)).argmin()
    lng_idx = (np.abs(lng_am-lng_eea)).argmin()
    no2_ateea = no2_am_2019[lat_idx, lng_idx]
    eea_2019.loc[index, 'AM NO2'] = no2_ateea
    lat_idx = (np.abs(lat_smod-lat_eea)).argmin()
    lng_idx = (np.abs(lng_smod-lng_eea)).argmin()
    smod_ateea = smod[lat_idx, lng_idx]
    eea_2019.loc[index, 'SMOD'] = smod_ateea    


import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
from matplotlib.lines import Line2D

# Projection, colormaps, etc. 
proj = ccrs.PlateCarree(central_longitude=0.0)
cmap = plt.cm.get_cmap('magma_r')
levels = np.arange(0,26+2,2)
col_rural = '#66c2a5'
col_urban = '#fc8d62'
# Initialize figure, axes
fig = plt.figure(figsize=(10,8))
ax1 = plt.subplot2grid((3,13),(0,0), aspect='auto', adjustable='box', 
    colspan=6, projection=proj) 
ax2 = plt.subplot2grid((3,13),(0,7), colspan=3) 
ax3 = plt.subplot2grid((3,13),(0,10), colspan=3) 
ax4 = plt.subplot2grid((3,13),(1,0), aspect='auto', adjustable='box', 
    colspan=6, projection=proj)
ax5 = plt.subplot2grid((3,13),(1,7), colspan=3) 
ax6 = plt.subplot2grid((3,13),(1,10), colspan=3) 
ax7 = plt.subplot2grid((3,13),(2,0), aspect='auto', adjustable='box', 
    colspan=6, projection=proj) 
ax8 = plt.subplot2grid((3,13),(2,7), colspan=3) 
ax9 = plt.subplot2grid((3,13),(2,10), colspan=3) 
# Map location of monitors and their mean concentrations
mb = ax1.scatter(naps_2019['Longitude'], naps_2019['Latitude'], 
    c=naps_2019['Mean'], s=2, cmap=cmap, transform=ccrs.PlateCarree(), 
    norm=mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False), 
    zorder=12)
ax4.scatter(aqs_2019['Longitude'], aqs_2019['Latitude'], 
    c=aqs_2019['Arithmetic Mean'], s=2, cmap=cmap, 
    transform=ccrs.PlateCarree(), norm=mpl.colors.BoundaryNorm(levels, 
    ncolors=cmap.N, clip=False), zorder=12)
ax7.scatter(eea_2019['SamplingPoint_Longitude'], eea_2019['SamplingPoint_Latitude'], 
    c=eea_2019['AQValue'], s=2, cmap=cmap, transform=ccrs.PlateCarree(), 
    norm=mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=False), 
    zorder=12)
# Jazz up maps
for ax in [ax1, ax4, ax7]:
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey', zorder=9)
    ax.set_aspect('auto')
    for k, spine in ax.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(10)
for ax in [ax1, ax4]:
    state_borders = cfeature.NaturalEarthFeature(category='cultural', 
        name='admin_1_states_provinces_lakes', scale='10m', 
        facecolor='None', zorder=10)
    ax.add_feature(state_borders, edgecolor='black', lw=0.1)  
ax7.add_feature(cfeature.BORDERS, edgecolor='black', lw=0.1)  
# Map extents
ax1.set_extent([-141.4,-50.8,41.1,60.7])
ax4.set_extent([-126,-66.6,24.5,46])
ax7.set_extent([-11.4,46.7,34.6,69.7])
# All NAPS
ax2.plot(naps_2019['Mean'], naps_2019['AM NO2'], 'ko', 
    markersize=3, alpha=0.6)
idx = np.isfinite(naps_2019['Mean']) & np.isfinite(naps_2019['AM NO2'])
m, b = np.polyfit(naps_2019['Mean'][idx], naps_2019['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax2.plot(xs, m*xs+b, ls='-', color='k')
ax2.text(0.03, 0.95, 'Bias$\,$=$\,$%.2f ppb'%np.mean(naps_2019['AM NO2'][idx]-
    naps_2019['Mean'][idx]), ha='left', va='center', 
    transform=ax2.transAxes, fontsize=8)
ax2.text(0.03, 0.89, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%m, ha='left',
    va='center', transform=ax2.transAxes, fontsize=8)
ax2.text(0.03, 0.83, 'b$\,$=$\,$%.2f ppb'%b, ha='left', va='center', 
    transform=ax2.transAxes, fontsize=8)
# All AQS
ax5.plot(aqs_2019['Arithmetic Mean'], aqs_2019['AM NO2'], 'ko', 
    markersize=3, alpha=0.6)
idx = np.isfinite(aqs_2019['Arithmetic Mean']) & np.isfinite(aqs_2019['AM NO2'])
m, b = np.polyfit(aqs_2019['Arithmetic Mean'][idx], 
    aqs_2019['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax5.plot(xs, m*xs+b, ls='-', color='k')
ax5.text(0.03, 0.95, 'Bias$\,$=$\,$%.2f ppb'%np.mean(aqs_2019['AM NO2'][idx]-
    aqs_2019['Arithmetic Mean'][idx]), ha='left', va='center', 
    transform=ax5.transAxes, fontsize=8)
ax5.text(0.03, 0.89, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%m, ha='left',
    va='center', transform=ax5.transAxes, fontsize=8)
ax5.text(0.03, 0.83, 'b$\,$=$\,$%.2f ppb'%b, ha='left', va='center', 
    transform=ax5.transAxes, fontsize=8)
# All EEA
ax8.plot(eea_2019['AQValue'], eea_2019['AM NO2'], 'ko', 
    markersize=3, alpha=0.6)
idx = np.isfinite(eea_2019['AQValue']) & np.isfinite(eea_2019['AM NO2'])
m, b = np.polyfit(eea_2019['AQValue'][idx], eea_2019['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax8.plot(xs, m*xs+b, ls='-', color='k')
ax8.text(0.03, 0.95, 'Bias$\,$=$\,$%.2f ppb'%np.mean(eea_2019['AM NO2'][idx]-
    eea_2019['AQValue'][idx]), ha='left', va='center', transform=
    ax8.transAxes, fontsize=8)
ax8.text(0.03, 0.89, 'm = %.2f ppb ppb$^{\mathregular{-1}}$'%m, ha='left',
    va='center', transform=ax8.transAxes, fontsize=8)
ax8.text(0.03, 0.83, 'b = %.2f ppb'%b, ha='left', va='center', 
    transform=ax8.transAxes, fontsize=8)
# Partition into rural and urban using the following
# — Class 30: “Urban Centre grid cell”, if the cell belongs to an Urban 
#      Centre spatial entity;
# — Class 23: “Dense Urban Cluster grid cell”, if the cell belongs to a 
#      Dense Urban Cluster spatial entity;
# — Class 22: “Semi-dense Urban Cluster grid cell”, if the cell belongs 
#      to a Semi-dense Urban Cluster spatial entity;
# — Class 21: “Suburban or per-urban grid cell”, if the cell belongs to 
#      an Urban Cluster cells at first hierarchical level but is not part 
#      of a Dense or Semi-dense Urban Cluster;
# — Class 13: “Rural cluster grid cell”, if the cell belongs to a Rural
#      Cluster spatial entity;
# — Class 12: “Low Density Rural grid cell”, if the cell is classified 
#      as Rural grid cells at first hierarchical level, has more than 
#      50 inhabitant and is not part of a Rural Cluster;
# — Class 11: “Very low density rural grid cell”, if the cell is 
#      classified as Rural grid cells at first hierarchical level, has 
#      less than 50 inhabitant and is not part of a Rural Cluster;
# — Class 10: “Water grid cell”, if the cell has 0.5 share covered by
#      permanent surface water and is not populated nor built.
# Discard values = 10, classify 11-13 as rural and 21-30 as urban
naps_2019_rural = naps_2019.loc[naps_2019['SMOD'].isin([11.,12.,13.])]
naps_2019_urban = naps_2019.loc[naps_2019['SMOD'].isin([21.,22.,23.,30.])]
aqs_2019_rural = aqs_2019.loc[aqs_2019['SMOD'].isin([11.,12.,13.])]
aqs_2019_urban = aqs_2019.loc[aqs_2019['SMOD'].isin([21.,22.,23.,30.])]
eea_2019_rural = eea_2019.loc[eea_2019['SMOD'].isin([11.,12.,13.])]
eea_2019_urban = eea_2019.loc[eea_2019['SMOD'].isin([21.,22.,23.,30.])]
# NAPS urban-rural
ax3.plot(naps_2019_urban['Mean'], naps_2019_urban['AM NO2'], 'o',
    color=col_urban, markersize=3, alpha=0.6)
ax3.plot(naps_2019_rural['Mean'], naps_2019_rural['AM NO2'], 'o',
    color=col_rural, markersize=3, alpha=0.6)
idx = np.isfinite(naps_2019_urban['Mean']) & \
    np.isfinite(naps_2019_urban['AM NO2'])
mu, bu = np.polyfit(naps_2019_urban['Mean'][idx], 
    naps_2019_urban['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax3.plot(xs, mu*xs+bu, ls='-', color=col_urban)
ax3.text(0.03, 0.95, 'Bias$\,$=$\,$%.2f ppb'%np.mean(
    naps_2019_urban['AM NO2'][idx]-naps_2019_urban['Mean'][idx]), 
    ha='left', va='center', transform=ax3.transAxes, color=col_urban, 
    fontsize=8)
ax3.text(0.03, 0.89, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%mu, 
    ha='left', va='center', transform=ax3.transAxes, color=col_urban, 
    fontsize=8)
ax3.text(0.03, 0.83, 'b$\,$=$\,$%.2f ppb'%bu, ha='left', va='center', 
    transform=ax3.transAxes, color=col_urban, fontsize=8)
idx = np.isfinite(naps_2019_rural['Mean']) & \
    np.isfinite(naps_2019_rural['AM NO2'])
mr, br = np.polyfit(naps_2019_rural['Mean'][idx], 
    naps_2019_rural['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax3.plot(xs, mr*xs+br, ls='-', color=col_rural)
ax3.text(0.03, 0.77, 'Bias$\,$=$\,$%.2f ppb'%np.mean(
    naps_2019_rural['AM NO2'][idx]-naps_2019_rural['Mean'][idx]), 
    ha='left', va='center', transform=ax3.transAxes, color=col_rural, 
    fontsize=8)
ax3.text(0.03, 0.71, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%mr, 
    ha='left', va='center', transform=ax3.transAxes, color=col_rural, 
    fontsize=8)
ax3.text(0.03, 0.65, 'b$\,$=$\,$%.2f ppb'%br, ha='left', va='center', 
    transform=ax3.transAxes, color=col_rural, fontsize=8)
# AQS urban-rural
ax6.plot(aqs_2019_urban['Arithmetic Mean'], aqs_2019_urban['AM NO2'], 'o',
    color=col_urban, markersize=3, alpha=0.6)
ax6.plot(aqs_2019_rural['Arithmetic Mean'], aqs_2019_rural['AM NO2'], 'o',
    color=col_rural, markersize=3, alpha=0.6)
idx = np.isfinite(aqs_2019_urban['Arithmetic Mean']) & \
    np.isfinite(aqs_2019_urban['AM NO2'])
ax6.text(0.03, 0.95, 'Bias$\,$=$\,$%.2f ppb'%np.mean(
    aqs_2019_urban['AM NO2'][idx]-aqs_2019_urban['Arithmetic Mean'][idx]), 
    ha='left', va='center', transform=ax6.transAxes, color=col_urban, 
    fontsize=8)
ax6.text(0.03, 0.89, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%mu, 
    ha='left', va='center', transform=ax6.transAxes, color=col_urban, 
    fontsize=8)
ax6.text(0.03, 0.83, 'b$\,$=$\,$%.2f ppb'%bu, ha='left', va='center', 
    transform=ax6.transAxes, color=col_urban, fontsize=8)
mu, bu = np.polyfit(aqs_2019_urban['Arithmetic Mean'][idx], 
    aqs_2019_urban['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax6.plot(xs, mu*xs+bu, ls='-', color=col_urban)
idx = np.isfinite(aqs_2019_rural['Arithmetic Mean']) & \
    np.isfinite(aqs_2019_rural['AM NO2'])
mr, br = np.polyfit(aqs_2019_rural['Arithmetic Mean'][idx], 
    aqs_2019_rural['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax6.plot(xs, mr*xs+br, ls='-', color=col_rural)
ax6.text(0.03, 0.77, 'Bias$\,$=$\,$%.2f ppb'%np.mean(
    aqs_2019_rural['AM NO2'][idx]-aqs_2019_rural['Arithmetic Mean'][idx]), 
    ha='left', va='center', transform=ax6.transAxes, color=col_rural, 
    fontsize=8)
ax6.text(0.03, 0.71, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%mr, 
    ha='left', va='center', transform=ax6.transAxes, color=col_rural,
    fontsize=8)
ax6.text(0.03, 0.65, 'b$\,$=$\,$%.2f ppb'%br, ha='left', va='center', 
    transform=ax6.transAxes, color=col_rural, fontsize=8)
# EEA urban-urban
ax9.plot(eea_2019_urban['AQValue'], eea_2019_urban['AM NO2'], 'o',
    color=col_urban, markersize=3, alpha=0.6)
ax9.plot(eea_2019_rural['AQValue'], eea_2019_rural['AM NO2'], 'o',
    color=col_rural, markersize=3, alpha=0.6)
idx = np.isfinite(eea_2019_urban['AQValue']) & \
    np.isfinite(eea_2019_urban['AM NO2'])
mu, bu = np.polyfit(eea_2019_urban['AQValue'][idx], 
    eea_2019_urban['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax9.plot(xs, mu*xs+bu, ls='-', color=col_urban)
ax9.text(0.03, 0.95, 'Bias$\,$=$\,$%.2f ppb'%np.mean(
    eea_2019_urban['AM NO2'][idx]-eea_2019_urban['AQValue'][idx]), 
    ha='left', va='center', transform=ax9.transAxes, color=col_urban, 
    fontsize=8)
ax9.text(0.03, 0.89, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%mu, ha='left',
    va='center', transform=ax9.transAxes, color=col_urban, fontsize=8)
ax9.text(0.03, 0.83, 'b$\,$=$\,$%.2f ppb'%bu, ha='left', va='center', 
    transform=ax9.transAxes, color=col_urban, fontsize=8)
idx = np.isfinite(eea_2019_rural['AQValue']) & \
    np.isfinite(eea_2019_rural['AM NO2'])
mr, br = np.polyfit(eea_2019_rural['AQValue'][idx], 
    eea_2019_rural['AM NO2'][idx], 1)
xs = np.linspace(0,40,500)
ax9.plot(xs, mr*xs+br, ls='-', color=col_rural)
ax9.text(0.03, 0.77, 'Bias$\,$=$\,$%.2f ppb'%np.mean(
    eea_2019_rural['AM NO2'][idx]-eea_2019_rural['AQValue'][idx]), 
    ha='left', va='center', transform=ax9.transAxes, color=col_rural, 
    fontsize=8)
ax9.text(0.03, 0.71, 'm$\,$=$\,$%.2f ppb ppb$^{\mathregular{-1}}$'%mr, ha='left',
    va='center', transform=ax9.transAxes, color=col_rural, fontsize=8)
ax9.text(0.03, 0.65, 'b$\,$=$\,$%.2f ppb'%br, ha='left', va='center', 
    transform=ax9.transAxes, color=col_rural, fontsize=8)
custom_lines = [Line2D([0], [0], marker='o', color=col_urban, lw=0),
    Line2D([0], [0], marker='o', color=col_rural, lw=0)]
ax8.legend(custom_lines, ['Urban', 'Rural'], 
    bbox_to_anchor=(1.05, -0.48), loc=8, ncol=2, fontsize=12, 
    frameon=False)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, wspace=0.3)
# Aesthetics
ax1.set_title('(a) Canada/NAPS', loc='left')
ax2.set_title('(b)', loc='left')
ax2.set_ylabel('Modeled NO$_{\mathregular{2}}$ [ppb]')
ax3.set_title('(c)', loc='left')
ax4.set_title('(d) United States/AQS', loc='left')
ax5.set_title('(e)', loc='left')
ax5.set_ylabel('Modeled NO$_{\mathregular{2}}$ [ppb]')
ax6.set_title('(f)', loc='left')
ax7.set_title('(g) European Union/EEA', loc='left')
ax8.set_title('(h)', loc='left')
ax8.set_ylabel('Modeled NO$_{\mathregular{2}}$ [ppb]')
ax8.set_xlabel('Observed NO$_{\mathregular{2}}$ [ppb]')
ax9.set_title('(i)', loc='left')
ax9.set_xlabel('Observed NO$_{\mathregular{2}}$ [ppb]')
for ax in [ax2, ax3, ax5, ax6, ax8, ax9]:
    ax.set_xlim([0,36])
    ax.set_xticks(np.linspace(0,36,7))
    ax.set_xticklabels([])
    ax.set_ylim([0,36])
    ax.set_yticks(np.linspace(0,36,7))    
    ax.set_yticklabels([])    
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),] 
    ax.plot(lims, lims, '--', color='darkgrey', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
# Axis tick labels
for ax in [ax8, ax9]:
    ax.set_xticklabels(['0', '', '12', '', '24', '', '36'])
for ax in [ax2, ax5, ax8]:
    ax.set_yticklabels(['0', '', '12', '', '24', '', '36'])
# Add colorbar
cbar_ax = fig.add_axes([ax7.get_position().x0, 0.08, 
    (ax7.get_position().x1-ax7.get_position().x0), 0.02])
cbar = fig.colorbar(mb, extend='max', cax=cbar_ax, orientation='horizontal')
cbar.set_label(label='NO$_{2}$ [ppbv]', fontsize=12)
plt.savefig('/Users/ghkerr/Desktop/susan_arash_paper.png', dpi=500)