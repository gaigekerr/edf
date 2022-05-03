#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:43:39 2021

@author: ghkerr
"""
DIR = '/Users/ghkerr/GW/edf/'
# DIR_HARM = DIR+'anenberg_mohegh_no2/harmonizedtables/'
#DIR_CENSUS = DIR_ROOT+'acs/'
DIR_CROSS = '/Users/ghkerr/GW/data/geography/'
DIR_GEO = DIR_CROSS+'tigerline/'
DIR_AQ = '/Users/ghkerr/GW/data/aq/aqs/'
#DIR_OUT = DIR_ROOT+'harmonizedtables/'
DIR_TYPEFACE = '/Users/ghkerr/Library/Fonts/'
DIR_GBD = '/Users/ghkerr/GW/data/gbd/'
DIR_FIG = '/Users/ghkerr/GW/edf/figs/'
import sys

if 'mpl' not in sys.modules:
    import matplotlib as mpl
    import matplotlib.font_manager as fm
    fe = fm.FontEntry(
        fname=DIR_TYPEFACE+'gulliver.ttf',
        name='gulliver')
    fm.fontManager.ttflist.insert(0, fe)
    mpl.rcParams['font.family'] = fe.name
    fe = fm.FontEntry(
        fname=DIR_TYPEFACE+'gulliverbold.ttf',
        name='gulliverbold')
    fm.fontManager.ttflist.insert(0, fe)
    mpl.rcParams['axes.unicode_minus'] = False

color1 = '#26C6DA'
color2 = '#112E51'
color3 = '#FF7043'
color4 = '#78909C'
color5 = '#2E78D2'
color6 = '#006C7A'
color7 = '#FFBEA9'
cscat = 'dodgerblue' 

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, 
    dh=.01, barh=.02, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values. Adapted from https://stackoverflow.com/
    questions/11517986/indicating-the-statistically-significant-
    difference-in-bar-graph

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """
    import matplotlib.pyplot as plt
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data <= p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'ns'
    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]
    ax_y0, ax_y1 = 0, 1
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
    # y = min(ly, ry) - dh
    y = min(height[num1], height[num2]) - dh
    # print(min(height)-y)
    barx = [lx+0.1, lx+0.1, rx-0.1, rx-0.1]
    bary = [y, y-barh, y-barh, y]
    mid = ((lx+rx)/2, y-barh-0.03)
    if text=='ns':
        mid = ((lx+rx)/2, (y)-barh-0.015)
    plt.plot(barx, bary, c='black', lw=0.75)
    kwargs = dict(ha='center', va='center')
    if fs is not None:
        kwargs['fontsize'] = fs
    if text=='ns':
        kwargs['fontsize'] = 8        
        plt.text(*mid, text, **kwargs)
    else:
        plt.text(*mid, text, **kwargs) 

def harmonize_aqstract_yearpoll(year, harmty, pollutant_aqs, pollutant): 
    """Fuction handles a given yera of AQS NO2 or PM25 concentrations (average
    concentrations at monitors) and identifies the census tract containing 
    these monitors and thereafter finds the corresponding tract-averaged NO2 or 
    PM2.5 concentrations from the Anenberg, Mohegh et al. (2022) or van 
    Donkelaar et al. (2021) datasets.

    Parameters
    ----------
    year : int
        Year of interest.
    harmty : TYPE
        DESCRIPTION.
    pollutant_aqs : TYPE
        Annual average AQS PM2.5 or NO2 concentrations by monitor
    pollutant : str
        Pollutant of interest; must match column names in argument 
        pollutant_aqs (i.e., 'NO2' or 'PM25).

    Returns
    -------
    None.

    """
    import numpy as np
    import requests
    url = "https://geo.fcc.gov/api/census/block/find"
    # Add empty columns for tract FIPS codes and dataset values
    pollutant_aqs['GEOID'] = np.nan
    pollutant_aqs['DATASET_VAL'] = np.nan
    # Find the tract code corresponding to each AQS monitor measurement; code
    # adapted from https://stackoverflow.com/questions/68665273/
    # mapping-gps-coordinates-with-census-tract-python
    for index, row in pollutant_aqs.iterrows():
        print(index, row['Parameter Name'], row['Year'])
        # Avoid monitors across the border
        if (row['State Name']!='Country Of Mexico') and \
            (row['State Name']!='Virgin Islands') :
            try:     
                block = requests.get(
                    url, params={"latitude": row["Latitude"], 
                        "longitude": row["Longitude"], "format": "json",
                        "censusYear":2010}).json(
                        )["Block"]["FIPS"]
                pollutant_aqs.loc[index,'GEOID'] = block[:11]                            
                # try:
                pollutant_aqs.loc[index,'DATASET_VAL'] = \
                    harmty.loc[block[:11]][pollutant]
                # except KeyError:
                # pollutant_aqs.loc[index,'DATASET_VAL'] = np.nan
            # Look up tract-average 
            except ValueError:
                print('No code for %.2f, %.2f'%(row['Latitude'], 
                    row['Longitude']))
                pass
    # Save output file 
    pollutant_aqs.to_csv(DIR_AQ+'%s_%d_bymonitor_harmonized.csv'%(
        pollutant, year), sep=',', encoding='utf-8')
    return 

def harmonize_aqstract(harmts):
    """Function opens annual average AQS data, extracts pollutants of interest
    (NO2 and PM2.5), and runs "harmonize_aqstract_yearpoll" for years of 
    interest (2010, 2015, and 2019).

    Parameters
    ----------
    harmts : pandas.core.frame.DataFrame
        Annual harmonized tract-level pollution concentrations, demographics, 
        and attributable fractions for measuring period. 

    Returns
    -------
    None.

    """
    import pandas as pd
    # Open AQS NO2 and PM25 for years of interest
    aqs2010 = pd.read_csv(DIR_AQ+'annual_conc_by_monitor_2010.csv', sep=',', 
        engine='python') 
    aqs2015 = pd.read_csv(DIR_AQ+'annual_conc_by_monitor_2015.csv', sep=',', 
        engine='python')
    aqs2019 = pd.read_csv(DIR_AQ+'annual_conc_by_monitor_2019.csv', sep=',', 
        engine='python') 
    # Sample NO2 and PM25
    # Note that observations have multiple entries for a single site; 
    # these multiple entries appear to stem from the fact that there are 
    # different entries for various Pollutant Standards and Metric Used.  In 
    # this case, use "Daily Mean" for metric and "PM25 24-hour 2012" for 
    # standard for PM2.5. Use "Observed values" for metric for NO2. 
    no2_aqs2010 = aqs2010.loc[(aqs2010['Parameter Code']==42602) & 
        (aqs2010['Metric Used']=='Observed values')].copy(deep=True)
    no2_aqs2015 = aqs2015.loc[(aqs2015['Parameter Code']==42602) & 
        (aqs2015['Metric Used']=='Observed values')].copy(deep=True)
    no2_aqs2019 = aqs2019.loc[(aqs2019['Parameter Code']==42602) & 
        (aqs2019['Metric Used']=='Observed values')].copy(deep=True)
    pm25_aqs2010 = aqs2010.loc[(aqs2010['Parameter Code']==88101) & 
        (aqs2010['Metric Used']=='Daily Mean') & 
        (aqs2010['Pollutant Standard']=='PM25 24-hour 2012')].copy(deep=True)
    pm25_aqs2015 = aqs2015.loc[(aqs2015['Parameter Code']==88101) & 
        (aqs2015['Metric Used']=='Daily Mean') & 
        (aqs2015['Pollutant Standard']=='PM25 24-hour 2012')].copy(deep=True)
    pm25_aqs2019 = aqs2019.loc[(aqs2019['Parameter Code']==88101) & 
        (aqs2019['Metric Used']=='Daily Mean') & 
        (aqs2019['Pollutant Standard']=='PM25 24-hour 2012')].copy(deep=True)
    # Harmonize
    harm2010 = harmts.loc[harmts.YEAR=='2006-2010']
    harm2015 = harmts.loc[harmts.YEAR=='2011-2015']
    harm2019 = harmts.loc[harmts.YEAR=='2015-2019']
    harmonize_aqstract_yearpoll(2010, harm2010, no2_aqs2010, 'NO2')
    harmonize_aqstract_yearpoll(2015, harm2015, no2_aqs2015, 'NO2')
    harmonize_aqstract_yearpoll(2019, harm2019, no2_aqs2019, 'NO2')
    harmonize_aqstract_yearpoll(2010, harm2010, pm25_aqs2010, 'PM25')
    harmonize_aqstract_yearpoll(2015, harm2015, pm25_aqs2015, 'PM25')
    harmonize_aqstract_yearpoll(2019, harm2019, pm25_aqs2019, 'PM25')
    return

def add_insetmap(axes_extent, map_extent, state_name, geometry, lng, lat, 
    quant, proj, fc='#f2f2f2', fips=None, harmonized=None, vara=None, 
    cmap=None, norm=None, sc=None):
    """Draws inset map

    Parameters
    ----------
    axes_extent : TYPE
        DESCRIPTION.
    map_extent : TYPE
        DESCRIPTION.
    state_name : TYPE
        DESCRIPTION.
    geometry : TYPE
        DESCRIPTION.
    lng : TYPE
        DESCRIPTION.
    lat : TYPE
        DESCRIPTION.
    quant : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    # Create new axes, set its projection; Mercator perserves shape well
    use_projection = ccrs.Mercator(central_longitude=0.0) 
    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
    sub_ax = plt.axes(axes_extent, projection=use_projection)  # normal units
    sub_ax.set_extent(map_extent, geodetic)  # map extents
    sub_ax.add_geometries(geometry.geoms, ccrs.PlateCarree(), lw=0.25, 
        facecolor=fc, edgecolor='k', zorder=5)
    # sub_ax.add_geometries(geometry.geoms, ccrs.PlateCarree(), lw=0.25, 
        # facecolor='None', edgecolor='k', zorder=15)   
    if cmap is not None: 
        sub_ax.scatter(lng, lat, s=quant, c=sc, zorder=15, 
            ec='None',
            clip_on=True, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    else: 
        sub_ax.scatter(lng, lat, s=quant, alpha=0.4, c=cscat, ec='None', 
            zorder=15, clip_on=True, transform=ccrs.PlateCarree())    
        sub_ax.scatter(lng, lat, s=quant, c='None', linewidth=.15, ec=cscat, 
            transform=proj, zorder=15, clip_on=True)    
    sub_ax.axis('off')
    if fips is not None: 
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+
            'tract_2010/tl_2019_%s_tract/tl_2019_%s_tract.shp'%(fips, fips))
        records = shp.records()
        tracts = shp.geometries()
        for record, tract in zip(records, tracts):
            # Find GEOID of tract
            gi = record.attributes['GEOID']
            # Look up harmonized NO2-census data for tract
            harmonized_tract = harmonized.loc[harmonized.index.isin([gi])]
            if harmonized_tract.empty==True: 
                sub_ax.add_geometries([tract], proj, facecolor='none', 
                    edgecolor="none", alpha=1., linewidth=0., rasterized=True, 
                    zorder=10)                
            else: 
                var_tract = harmonized_tract['%s'%vara].values[0]
                if np.isnan(var_tract)==True:
                    sub_ax.add_geometries([tract], proj, facecolor='none', 
                        edgecolor='none', alpha=1., linewidth=0.0, 
                        rasterized=True, zorder=10)
                else:
                    sub_ax.add_geometries([tract], proj, facecolor=cmap(
                        norm(var_tract)), edgecolor="none",
                        alpha=1., linewidth=0., rasterized=True, zorder=10)
    sub_ax.add_feature(cfeature.NaturalEarthFeature('physical', 
        'ocean', '10m', edgecolor='None', facecolor='w', alpha=1.), 
        zorder=14)
    return

def rect_from_bound(xmin, xmax, ymin, ymax):
    """Returns list of (x,y)'s for a rectangle"""
    xs = [xmax, xmin, xmin, xmax, xmax]
    ys = [ymax, ymax, ymin, ymin, ymax]
    return [(x, y) for x, y in zip(xs, ys)]

def drawPieMarker(xs, ys, ratios, sizes, colors, ax):
    """Adapated from www.fatalerrors.org/a/scatter-plot-with-matplotlib-and-
    mark-with-pie-chart.html. Here, xs, ys are the x and y values we pass in,
    ratio refers to our share (for example, 70% of men and 30% of women), sizes
    refer to the size of this point, and ax is the drawing function"""
    markers = []
    previous = 0
    # Calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x  = [0] + np.cos(np.linspace(previous, this, 30)).tolist() + [0]
        y  = [0] + np.sin(np.linspace(previous, this, 30)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes), 'facecolor':color})
    # Scatter each of the pie pieces to create piesggg
    for marker in markers:
        ax.scatter(xs, ys, **marker, alpha=1., zorder=30, clip_on=True)

def reject_outliers(data):
    data = np.array(data)
    mean = np.nanmean(data)
    maskMin = np.percentile(data, 0)
    maskMax = np.percentile(data, 100)
    mask = np.ma.masked_outside(data, maskMin, maskMax)
    # print('Masking values outside of {} and {}'.format(maskMin, maskMax))
    return mask

def pollhealthdisparities(burden):
    """Calculate population-weighted pollutant (NO2 and PM2.5) concentrations
    and mortality or incidence rates for pollution-attributable health 
    endpoints for a given DataFrame (e.g., all, urban, MSA) of census tracts. 
    
    Population-weighting is accomplished by calculating the product of the 
    demographic group (j) population (p_j) and a particular variable (e.g., 
    NO2 concentrations; NO2_j) in the ith tract, summed over all census tracts 
    with NO2 data (n) and divided by the summation of the group population 
    (p_j) 
                              n                   n
    Population-weighted NO2 = Σ NO2_i • p_{i,j} / Σ p_{i,j}
                              i                   i
    
    This operation is done for Blacks, whites, Hispanics, and non-Hispanics. 

    Parameters
    ----------
    burden : pandas.core.frame.DataFrame
        Tract-level disease incidence and demographic information for various 
        health endpoints for a given area (e.g., all tracts, MSA-specific)

    Returns
    -------
    pm25black : numpy.float64
        Population-weighted PM2.5 for Black residents.
    no2black : numpy.float64
        Population-weighted NO2 for Black residents.
    copdblack : numpy.float64
        Population-weighted COPD mortality rates for Black residents. 
    ihdblack : numpy.float64
        Population-weighted ischemic heart disease mortality rates for Black 
        residents. 
    lriblack : numpy.float64
        Population-weighted lower respiratory infection mortality rates for 
        Black residents. 
    dmblack : numpy.float64
        Population-weighted type 2 diabetes mortality rates for Black 
        residents. 
    lcblack : numpy.float64
        Population-weighted lung cancer mortality rates for Black residents. 
    stblack : numpy.float64
        Population-weighted stroke mortality rates for Black residents. 
    asthmablack : numpy.float64
        Population-weighted pediatric asthma incidence rates for Black 
        residents. 
    allmortblack : numpy.float64
        Population-weighted total PM2.5-attributable mortality rates for 
        Black residents. 
    pm25white : numpy.float64
        Population-weighted PM2.5 for white residents.
    no2white : numpy.float64
        Population-weighted NO2 for white residents.
    copdwhite : numpy.float64
        Population-weighted COPD mortality rates for white residents. 
    ihdwhite : numpy.float64
        Population-weighted ischemic heart disease mortality rates for white 
        residents. 
    lriwhite : numpy.float64
        Population-weighted lower respiratory infection mortality rates for 
        white residents. 
    dmwhite : numpy.float64
        Population-weighted type 2 diabetes mortality rates for white 
        residents. 
    lcwhite : numpy.float64
        Population-weighted lung cancer mortality rates for white residents. 
    stwhite : numpy.float64
        Population-weighted stroke mortality rates for white residents. 
    asthmawhite : numpy.float64
        Population-weighted pediatric asthma incidence rates for white 
        residents. 
    allmortwhite : numpy.float64
        Population-weighted total PM2.5-attributable mortality rates for 
        white residents.         
    pm25nh : numpy.float64
        Population-weighted PM2.5 for non-Hispanic residents.
    no2nh : numpy.float64
        Population-weighted NO2 for non-Hispanic residents.
    copdnh : numpy.float64
        Population-weighted COPD mortality rates for non-Hispanic residents. 
    ihdnh : numpy.float64
        Population-weighted ischemic heart disease mortality rates for 
        non-Hispanic residents. 
    lrinh : numpy.float64
        Population-weighted lower respiratory infection mortality rates for 
        non-Hispanic residents. 
    dmnh : numpy.float64
        Population-weighted type 2 diabetes mortality rates for non-Hispanic
        residents. 
    lcnh : numpy.float64
        Population-weighted lung cancer mortality rates for non-Hispanic
        residents. 
    stnh : numpy.float64
        Population-weighted stroke mortality rates for non-Hispanic residents. 
    asthmanh : numpy.float64
        Population-weighted pediatric asthma incidence rates for non-Hispanic
        residents. 
    allmortnh : numpy.float64
        Population-weighted total PM2.5-attributable mortality rates for 
        non-Hispanic residents.     
    pm25h : numpy.float64
        Population-weighted PM2.5 for Hispanic residents.
    no2h : numpy.float64
        Population-weighted NO2 for Hispanic residents.
    copdh : numpy.float64
        Population-weighted COPD mortality rates for Hispanic residents. 
    ihdh : numpy.float64
        Population-weighted ischemic heart disease mortality rates for 
        Hispanic residents. 
    lrih : numpy.float64
        Population-weighted lower respiratory infection mortality rates for 
        Hispanic residents. 
    dmh : numpy.float64
        Population-weighted type 2 diabetes mortality rates for Hispanic
        residents. 
    lch : numpy.float64
        Population-weighted lung cancer mortality rates for Hispanic residents. 
    sth : numpy.float64
        Population-weighted stroke mortality rates for Hispanic residents. 
    asthmah : numpy.float64
        Population-weighted pediatric asthma incidence rates for Hispanic
        residents. 
    allmorth : numpy.float64
        Population-weighted total PM2.5-attributable mortality rates for 
        Hispanic residents.     
    """
    # Population-weighted pollution and disease rates for Blacks
    pm25black = ((burden['PM25']*burden[['race_nh_black', 'race_h_black']
        ].sum(axis=1)).sum()/burden[['race_nh_black','race_h_black']
        ].sum(axis=1).sum())
    no2black = ((burden['NO2']*burden[['race_nh_black', 'race_h_black']
        ].sum(axis=1)).sum()/burden[['race_nh_black','race_h_black']
        ].sum(axis=1).sum())
    copdblack = ((burden['BURDENCOPDRATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    ihdblack = ((burden['BURDENIHDRATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    lriblack = ((burden['BURDENLRIRATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    dmblack = ((burden['BURDENDMRATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    lcblack = ((burden['BURDENLCRATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    stblack = ((burden['BURDENSTRATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    afasthmablack = ((burden['AFPA']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())  
    asthmablack = ((burden['BURDENASTHMARATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    allmortblack = ((burden['BURDENPMALLRATE']*burden[[
        'race_nh_black','race_h_black']].sum(axis=1)).sum()/burden[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())        
    # For whites                         
    pm25white = ((burden['PM25']*burden[['race_nh_white', 'race_h_white']
        ].sum(axis=1)).sum()/burden[['race_nh_white','race_h_white']
        ].sum(axis=1).sum())
    no2white = ((burden['NO2']*burden[['race_nh_white', 'race_h_white']
        ].sum(axis=1)).sum()/burden[['race_nh_white','race_h_white']
        ].sum(axis=1).sum())
    copdwhite = ((burden['BURDENCOPDRATE']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())
    ihdwhite = ((burden['BURDENIHDRATE']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())
    lriwhite = ((burden['BURDENLRIRATE']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())
    dmwhite = ((burden['BURDENDMRATE']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())
    lcwhite = ((burden['BURDENLCRATE']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())
    stwhite = ((burden['BURDENSTRATE']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())
    afasthmawhite = ((burden['AFPA']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())    
    asthmawhite = ((burden['BURDENASTHMARATE']*burden[['race_nh_white',
        'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
        'race_h_white']].sum(axis=1).sum())
    allmortwhite = ((burden['BURDENPMALLRATE']*burden[
        ['race_nh_white','race_h_white']].sum(axis=1)).sum()/burden[
        ['race_nh_white','race_h_white']].sum(axis=1).sum())            
    # For non-Hispanic
    no2nh = ((burden['NO2']*burden['race_nh']).sum()/burden['race_nh'].sum())
    pm25nh = ((burden['PM25']*burden['race_nh']).sum()/burden['race_nh'].sum())
    copdnh = ((burden['BURDENCOPDRATE']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    ihdnh = ((burden['BURDENIHDRATE']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    lrinh = ((burden['BURDENLRIRATE']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    dmnh = ((burden['BURDENDMRATE']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    lcnh = ((burden['BURDENLCRATE']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    stnh = ((burden['BURDENSTRATE']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    afasthmanh = ((burden['AFPA']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    asthmanh = ((burden['BURDENASTHMARATE']*burden['race_nh']).sum()/
        burden['race_nh'].sum())
    allmortnh = ((burden['BURDENPMALLRATE']*burden[
        'race_nh']).sum()/burden['race_nh'].sum())    
    # For Hispanic 
    no2h = ((burden['NO2']*burden['race_h']).sum()/burden['race_h'].sum())
    pm25h = ((burden['PM25']*burden['race_h']).sum()/burden['race_h'].sum())
    copdh = ((burden['BURDENCOPDRATE']*burden['race_h']).sum()/
        burden['race_h'].sum())
    ihdh = ((burden['BURDENIHDRATE']*burden['race_h']).sum()/
        burden['race_h'].sum())
    lrih = ((burden['BURDENLRIRATE']*burden['race_h']).sum()/
        burden['race_h'].sum())
    dmh = ((burden['BURDENDMRATE']*burden['race_h']).sum()/
        burden['race_h'].sum())
    lch = ((burden['BURDENLCRATE']*burden['race_h']).sum()/
        burden['race_h'].sum())
    sth = ((burden['BURDENSTRATE']*burden['race_h']).sum()/
        burden['race_h'].sum())
    afasthmah = ((burden['AFPA']*burden['race_h']).sum()/
        burden['race_h'].sum())
    asthmah = ((burden['BURDENASTHMARATE']*burden['race_h']).sum()/
        burden['race_h'].sum())
    allmorth = ((burden['BURDENPMALLRATE']*burden['race_h']).sum()/burden[
        'race_h'].sum())    
    return (pm25black, no2black, copdblack, ihdblack, lriblack, dmblack,
        lcblack, stblack, afasthmablack, asthmablack, allmortblack, pm25white, 
        no2white, copdwhite, ihdwhite, lriwhite, dmwhite, lcwhite, stwhite, 
        afasthmawhite, asthmawhite, allmortwhite, pm25nh, no2nh, copdnh, ihdnh, 
        lrinh, dmnh, lcnh, stnh, afasthmanh, asthmanh, allmortnh, pm25h, no2h, 
        copdh, ihdh, lrih, dmh, lch, sth, afasthmah,  asthmah, allmorth)

def fig1(burdents, pmburdenrate_allmsa, asthmaburdenrate_allmsa, lng_allmsa, 
    lat_allmsa): 
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.io import shapereader
    import matplotlib.patches as mpatches
    # # # # Constants
    proj = ccrs.PlateCarree(central_longitude=0.0)
    cscat = 'dodgerblue'  
    years = np.arange(2010,2020,1)
    # Calculate some pithy statistics
    from scipy import stats
    burdents_byyear = burdents.groupby(['YEAR'])[['PM25','NO2']].mean()
    trend_pm25 = stats.linregress(years, burdents_byyear['PM25'])
    trend_no2 = stats.linregress(years, burdents_byyear['NO2'])
    print('# # # # Nationwide PM2.5 trends')
    print('Slope/p-value = ', trend_pm25.slope, '/', trend_pm25.pvalue)
    pc_pm25 = ((burdents_byyear.PM25[-1]-burdents_byyear.PM25[0])/
        burdents_byyear.PM25[0])*100
    print('Percent change', pc_pm25,'%')
    print('# # # # Nationwide NO2 trends')
    print('Slope/p-value = ', trend_no2.slope, '/', trend_no2.pvalue)
    pc_no2 = ((burdents_byyear.NO2[-1]-burdents_byyear.NO2[0])/
        burdents_byyear.NO2[0])*100
    print('Percent change', pc_no2,'%')
    for state in np.unique(burdents.STATE):
        burdents_state = burdents.loc[burdents.STATE.isin([state])]
        burdents_state_byyear = burdents_state.groupby(['YEAR'])[['PM25','NO2']].mean()
        trend_pm25 = stats.linregress(years, burdents_state_byyear['PM25'])
        trend_no2 = stats.linregress(years, burdents_state_byyear['NO2'])
        print('# # # # %s PM2.5 trends'%state)
        print('Slope/p-value = ', trend_pm25.slope, '/', trend_pm25.pvalue)
        pc_pm25 = ((burdents_state_byyear.PM25[-1]-burdents_state_byyear.PM25[0])/
            burdents_state_byyear.PM25[0])*100
        print('Percent change', pc_pm25,'%')
        print('# # # # %s NO2 trends'%state)    
        print('Slope/p-value = ', trend_no2.slope, '/', trend_no2.pvalue)
        pc_no2 = ((burdents_state_byyear.NO2[-1]-burdents_state_byyear.NO2[0])/
            burdents_state_byyear.NO2[0])*100
        print('Percent change', pc_no2,'%')
        print('\n')
    # # # # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()   
    usaidx = [x.attributes['ADM0_A3'] for x in countries]
    usaidx = np.where(np.in1d(np.array(usaidx), ['PRI','USA'])==True)
    usa = list(reader.geometries())
    usa = np.array(usa, dtype=object)[usaidx[0]]
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shpreader.Reader(lakes)
    lakes = lakes_reader.records()   
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names)=='Lake Superior') |
        (np.array(lake_names)=='Lake Michigan') | 
        (np.array(lake_names)=='Lake Huron') |
        (np.array(lake_names)=='Lake Erie') |
        (np.array(lake_names)=='Lake Ontario'))[0]
    great_lakes = np.array(list(lakes_reader.geometries()), 
        dtype=object)[great_lakes]
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='10m', category='cultural', 
        name=shapename)
    states_shp = shpreader.Reader(states_shp)
    # # # # Discretize/bin attributable fractions, burdens, and rates
    # NO2-attributable asthma rates
    asthmaburdenrated_allmsa = np.empty(shape=asthmaburdenrate_allmsa.shape[0])
    asthmaburdenrated_allmsa[:] = np.nan
    no2p30 = np.nanpercentile(asthmaburdenrate_allmsa, 30)
    no2p60 = np.nanpercentile(asthmaburdenrate_allmsa, 60)
    no2p90 = np.nanpercentile(asthmaburdenrate_allmsa, 90)
    no2p95 = np.nanpercentile(asthmaburdenrate_allmsa, 97)
    burdenrateb1 = np.where(asthmaburdenrate_allmsa<no2p30)[0]
    burdenrateb2 = np.where((asthmaburdenrate_allmsa>=no2p30) & 
        (asthmaburdenrate_allmsa<no2p60))[0]
    burdenrateb3 = np.where((asthmaburdenrate_allmsa>=no2p60) & 
        (asthmaburdenrate_allmsa<no2p90))[0]
    burdenrateb4 = np.where((asthmaburdenrate_allmsa>=no2p90) & 
        (asthmaburdenrate_allmsa<no2p95))[0]
    burdenrateb5 = np.where((asthmaburdenrate_allmsa>=no2p95))[0]
    asthmaburdenrated_allmsa[burdenrateb1] = 6
    asthmaburdenrated_allmsa[burdenrateb2] = 20
    asthmaburdenrated_allmsa[burdenrateb3] = 40
    asthmaburdenrated_allmsa[burdenrateb4] = 70
    asthmaburdenrated_allmsa[burdenrateb5] = 200
    # PM2.5-attributable mortality rates
    pmburdenrated_allmsa = np.empty(shape=pmburdenrate_allmsa.shape[0])
    pmburdenrated_allmsa[:] = np.nan
    pmp30 = np.nanpercentile(pmburdenrate_allmsa, 30)
    pmp60 = np.nanpercentile(pmburdenrate_allmsa, 60)
    pmp90 = np.nanpercentile(pmburdenrate_allmsa, 90)
    pmp95 = np.nanpercentile(pmburdenrate_allmsa, 97)
    burdenrateb1 = np.where(pmburdenrate_allmsa<pmp30)[0]
    burdenrateb2 = np.where((pmburdenrate_allmsa>=pmp30) & 
        (pmburdenrate_allmsa<pmp60))[0]
    burdenrateb3 = np.where((pmburdenrate_allmsa>=pmp60) & 
        (pmburdenrate_allmsa<pmp90))[0]
    burdenrateb4 = np.where((pmburdenrate_allmsa>=pmp90) & 
        (pmburdenrate_allmsa<pmp95))[0]
    burdenrateb5 = np.where((pmburdenrate_allmsa>=pmp95))[0]
    pmburdenrated_allmsa[burdenrateb1] = 6
    pmburdenrated_allmsa[burdenrateb2] = 20
    pmburdenrated_allmsa[burdenrateb3] = 40
    pmburdenrated_allmsa[burdenrateb4] = 70
    pmburdenrated_allmsa[burdenrateb5] = 200
    # # # # Plotting
    fig = plt.figure(figsize=(12,8))
    axts1 = plt.subplot2grid((5,2),(0,0), rowspan=2)
    axts2 = plt.subplot2grid((5,2),(0,1), rowspan=2)
    ax2 = plt.subplot2grid((5,2),(2,0), rowspan=3, projection=proj)
    ax1 = plt.subplot2grid((5,2),(2,1), rowspan=3, projection=proj)
    axts1.set_title('(A) Premature deaths due to PM$_\mathregular{2.5}$', 
        loc='left')
    axts2.set_title('(B) New asthma cases due to NO$_\mathregular{2}$', 
        loc='left')
    ax2.set_title('(C) Premature deaths due to PM$_\mathregular{2.5}$ '+\
        'per 100000', loc='left')
    ax1.set_title('(D) New asthma cases due to NO$_\mathregular{2}$ '+\
        'per 100000', loc='left')
    ax1t = axts1.twinx()
    ax2t = axts2.twinx()
    pm25_mean, no2_mean = [], []
    copd, lri, lc, ihd, dm, st, asthma = [], [], [], [], [], [], []
    for year in np.arange(2010, 2020, 1):
        vintage = '%d-%d'%(year-4, year)
        burdenty = burdents.loc[burdents['YEAR']==vintage]
        # Pollutant concentrations
        pm25_mean.append(burdenty.PM25.mean())
        no2_mean.append(burdenty.NO2.mean())
        # Burdens for health endpoints    
        copd.append(burdenty.BURDENCOPD.sum())
        lri.append(burdenty.BURDENLRI.sum())
        lc.append(burdenty.BURDENLC.sum())
        ihd.append(burdenty.BURDENIHD.sum())
        dm.append(burdenty.BURDENDM.sum())
        st.append(burdenty.BURDENST.sum())
        asthma.append(burdenty.BURDENASTHMA.sum())
    # Convert lists to arrays
    pm25_mean = np.array(pm25_mean)
    no2_mean = np.array(no2_mean) 
    copd = np.array(copd)
    lri = np.array(lri) 
    lc = np.array(lc)
    ihd = np.array(ihd)
    dm = np.array(dm)
    st = np.array(st)
    asthma = np.array(asthma)
    # PM25 and PM25-attributable mortality
    for i in np.arange(0, len(np.arange(2010,2020,1)), 1): 
        copdty = copd[i]
        lrity = lri[i]
        lcty = lc[i]
        ihdty = ihd[i]
        dmty = dm[i]
        stty = st[i]
        axts1.bar(i+2010, ihdty, color=color1, bottom=0, zorder=10)
        axts1.bar(i+2010, stty, color=color2, bottom=ihdty, zorder=10)    
        axts1.bar(i+2010, lcty, color=color3, bottom=ihdty+stty, zorder=10)
        axts1.bar(i+2010, copdty, color=color4, bottom=ihdty+stty+lcty,
            zorder=10)
        axts1.bar(i+2010, dmty, color=color5, bottom=ihdty+stty+lcty+copdty,
            zorder=10)    
        axts1.bar(i+2010, lrity, color=color6, bottom=ihdty+stty+lcty+copdty+dmty,
            zorder=10)
    ax1t.plot(years, pm25_mean, ls='-', marker='o', color='k', lw=2)  
    axts1.set_ylim([0, 80000])
    axts1.set_yticks(np.linspace(0,80000,5))
    axts1.yaxis.set_label_coords(-0.15, 0.5)
    ax1t.set_ylim([0,10])  
    ax1t.set_yticks(np.linspace(0,10,6))
    ax1t.set_ylabel('PM$_{\mathregular{2.5}}$ [$\mathregular{\mu}$g m$'+\
        '^{\mathregular{-3}}$]', rotation=270)
    ax1t.yaxis.set_label_coords(1.1, 0.5)
    # NO2 and NO2-attributable new cases
    axts2.bar(years, asthma, color=color7, zorder=10)
    ax2t.errorbar(years, no2_mean, ls='-', marker='o', color='k')
    axts2.set_ylim([0, 200000])
    axts2.set_yticks(np.linspace(0,200000,5))
    axts2.yaxis.set_label_coords(-0.15, 0.5)
    ax2t.set_ylim([0, 12])
    ax2t.set_yticks(np.linspace(0,12,5))
    ax2t.set_ylabel('NO$_{\mathregular{2}}$ [ppbv]', rotation=270)
    ax2t.yaxis.set_label_coords(1.1, 0.5)
    for ax in [ax1t, ax2t]:
        ax.set_xlim([2009.25,2019.75])
        ax.set_xticks(np.arange(2010,2020,1))
    for ax in [ax1, ax2]:
        ax.grid(axis='y', which='major', zorder=0, color='grey', ls='--')
    # Add legend denoting PM2.5 timeseries
    ax1t.annotate('PM$_{\mathregular{2.5}}$', xy=(2019, pm25_mean[-3]), 
        xycoords='data', xytext=(2018.2, pm25_mean[-3]+1.4), 
        textcoords='data', arrowprops=dict(arrowstyle='->', color='k'), 
        fontsize=12)
    ax2t.annotate('NO$_{\mathregular{2}}$', xy=(2019, no2_mean[-3]), 
        xycoords='data', xytext=(2018.7, no2_mean[-3]+2.1), 
        textcoords='data', arrowprops=dict(arrowstyle='->', color='k'), 
        fontsize=12)
    # Add borders, set map extent, etc. 
    for ax in [ax1, ax2]:
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)
        ax.add_geometries(usa, crs=proj, lw=0.25, facecolor='None', 
            edgecolor='k', zorder=15)
        ax.add_geometries(great_lakes, crs=ccrs.PlateCarree(), 
            facecolor='w', lw=0.25, edgecolor='k', alpha=1., zorder=17)
        ax.axis('off')
    ax1.scatter(lng_allmsa, lat_allmsa, s=asthmaburdenrated_allmsa, alpha=0.4, 
        c=cscat, ec='None', transform=proj, zorder=30, clip_on=True)
    ax1.scatter(lng_allmsa, lat_allmsa, s=asthmaburdenrated_allmsa, fc='None', 
        linewidth=.15, ec=cscat, transform=proj, zorder=30, clip_on=True)
    ax2.scatter(lng_allmsa, lat_allmsa, s=pmburdenrated_allmsa, alpha=0.4, 
        c=cscat, ec='None', transform=proj, zorder=30, clip_on=True)
    ax2.scatter(lng_allmsa, lat_allmsa, s=pmburdenrated_allmsa, fc='None', 
        linewidth=.15, ec=cscat, transform=proj, zorder=30, clip_on=True)
    # # # # Legends (first create dummy)
    # Create legend for PM2.5 endpoints
    pihd = mpatches.Patch(color=color1, label='Ischemic heart disease')
    pst = mpatches.Patch(color=color2, label='Stroke')
    plc = mpatches.Patch(color=color3, label='Lung cancer')
    pcopd = mpatches.Patch(color=color4, label='COPD')
    pdm = mpatches.Patch(color=color5, label='Type 2 diabetes')
    plri = mpatches.Patch(color=color6, label='Lower respiratory infection')
    axts1.legend(handles=[pihd, pst, plc, pcopd, pdm, plri], 
        bbox_to_anchor=(1.3, -0.12), ncol=3, frameon=False)
    b1 = ax1.scatter([],[], s=2, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b2 = ax1.scatter([],[], s=4, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b3 = ax1.scatter([],[], s=20, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b4 = ax1.scatter([],[], s=50, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b5 = ax1.scatter([],[], s=300, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    ax1.legend((b1, b2, b3, b4, b5),
        ('< %d'%(no2p30), '%d-%d'%(no2p30,no2p60), '%d-%d'%(no2p60,no2p90), 
        '%d-%d'%(no2p90,no2p95), '> %d'%(no2p90)), scatterpoints=1, 
        labelspacing = 0.8, loc='center right', bbox_to_anchor=(1.04, 0.32),
        ncol=1, frameon=False, fontsize=8)
    b1 = ax2.scatter([],[], s=2, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b2 = ax2.scatter([],[], s=4, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b3 = ax2.scatter([],[], s=20, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b4 = ax2.scatter([],[], s=50, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    b5 = ax2.scatter([],[], s=300, marker='o', ec=cscat, fc=cscat, linewidth=.15, 
        alpha=0.4)
    ax2.legend((b1, b2, b3, b4, b5),
        ('< %d'%(pmp30), '%d-%d'%(pmp30, pmp60), '%d-%d'%(pmp60, pmp90), 
          '%d-%d'%(pmp90, pmp95), '> %d'%(pmp95)), scatterpoints=1, 
        labelspacing = 0.8, loc='center right', bbox_to_anchor=(1.04, 0.32),
            ncol=1, frameon=False, fontsize=8)
    # Adjust plots and make maps a little bigger 
    plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.)
    box1 = ax1.get_position()
    box2 = ax2.get_position()
    ax1.set_position([box1.x0-0.02, box1.y0-0.02, 
      (box1.x1-box1.x0)*1.2, (box1.y1-box1.y0)*1.2])
    ax2.set_position([box2.x0-0.02, box2.y0-0.02, 
      (box2.x1-box2.x0)*1.2, (box2.y1-box2.y0)*1.2])
    # Add inset maps
    for astate in states_shp.records():
        if astate.attributes['name']=='Alaska':
            # Alaska asthma
            map_extent = (-179.99, -130, 49, 73) # lonmin, lonmax, latmin, latmax
            axes_extent = (ax1.get_position().x0-0.05, ax1.get_position().y0-0.02, 
                0.11, 0.11) # LLx, LLy, width, height
            geometry = astate.geometry
            add_insetmap(axes_extent, map_extent, '', astate.geometry, 
                lng_allmsa, lat_allmsa, asthmaburdenrated_allmsa, proj)
            # Alaska PM2.5
            axes_extent = (ax2.get_position().x0-0.05, ax2.get_position().y0-0.02, 
                0.11, 0.11) # LLx, LLy, width, height
            add_insetmap(axes_extent, map_extent, '', astate.geometry, 
                lng_allmsa, lat_allmsa, pmburdenrated_allmsa, proj)
        elif astate.attributes['name']=='Hawaii':
            map_extent = (-162, -154, 18.75, 23)
            geometry = astate.geometry
            axes_extent = (ax1.get_position().x0+0.02, ax1.get_position().y0-0.01, 
                0.07, 0.07)
            add_insetmap(axes_extent, map_extent, '', astate.geometry, 
                lng_allmsa, lat_allmsa, asthmaburdenrated_allmsa, proj)
            axes_extent = (ax2.get_position().x0+0.02, ax2.get_position().y0-0.01, 
                0.07, 0.07)
            add_insetmap(axes_extent, map_extent, '', astate.geometry, 
                lng_allmsa, lat_allmsa, pmburdenrated_allmsa, proj)
        elif astate.attributes['name']=='PRI-00 (Puerto Rico aggregation)':
            map_extent = (-68., -65., 17.5, 18.8)
            geometry = astate.geometry
            axes_extent = (ax1.get_position().x0+0.08, ax1.get_position().y0-0.015, 
                0.07, 0.07)
            add_insetmap(axes_extent, map_extent, '', astate.geometry, 
                lng_allmsa, lat_allmsa, asthmaburdenrated_allmsa, proj)
            axes_extent = (ax2.get_position().x0+0.08, ax2.get_position().y0-0.015, 
                0.07, 0.07)
            add_insetmap(axes_extent, map_extent, '', astate.geometry, 
                lng_allmsa, lat_allmsa, pmburdenrated_allmsa, proj)
        elif astate.attributes['sr_adm0_a3']=='USA':
            geometry = astate.geometry
            # # Save off information for text on plots
            # statename.append(astate.attributes['name'])
            # statecentlat.append(geometry.centroid.y)
            # statecentlng.append(geometry.centroid.x)
            for ax in [ax1, ax2]:
                ax.add_geometries([geometry], crs=ccrs.PlateCarree(), 
                    facecolor='#f2f2f2', lw=0.5, edgecolor='w', alpha=1., 
                    zorder=0)
    plt.savefig(DIR_FIG+'fig1.pdf', dpi=1000)
    return 

def fig2(burdents):
    """
    
    Returns
    -------
    None.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
    jitter=0.04
    # Pollution concentrations and demographics for years of interest
    burden19 = burdents.loc[burdents['YEAR']=='2015-2019']
    (pm25black19, no2black19, copdblack19, ihdblack19, lriblack19, dmblack19,
     lcblack19, stblack19, afasthmablack19, asthmablack19, allmortblack19, 
     pm25white19, no2white19, copdwhite19, ihdwhite19, lriwhite19, dmwhite19, 
     lcwhite19, stwhite19, afasthmawhite19, asthmawhite19, allmortwhite19, 
     pm25nh19, no2nh19, copdnh19, ihdnh19, lrinh19, dmnh19, lcnh19, stnh19, 
     afasthmanh19, asthmanh19, allmortnh19, pm25h19, no2h19, copdh19, ihdh19, 
     lrih19, dmh19, lch19, sth19, afasthmah19, asthmah19, allmorth19) = \
     pollhealthdisparities(burden19)
    # Calculate most/least white pollutants and burdens as in Kerr et al. (2021)
    frac_white19 = ((burden19[['race_nh_white','race_h_white']].sum(axis=1))/
        burden19['race_tot'])
    frac_hisp19 = (burden19['race_h']/burden19['race_tot'])
    # Pollutants and rates averaged over all MSAs
    mostwhitepm25all = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].PM25.mean()
    leastwhitepm25all = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].PM25.mean()
    mostwhiteno2all = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].NO2.mean()
    leastwhiteno2all = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].NO2.mean()
    mostwhiteafasthmaall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].AFPA.mean()
    leastwhiteafasthmaall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].AFPA.mean()
    mostwhiteasthmaall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENASTHMARATE.mean()
    leastwhiteasthmaall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENASTHMARATE.mean()
    mostwhiteihdall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENIHDRATE.mean()
    leastwhiteihdall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENIHDRATE.mean()
    mostwhitestall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENSTRATE.mean()
    leastwhitestall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENSTRATE.mean()
    mostwhitelcall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENLCRATE.mean()
    leastwhitelcall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENLCRATE.mean()
    mostwhitedmall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENDMRATE.mean()
    leastwhitedmall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENDMRATE.mean()
    mostwhitecopdall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENCOPDRATE.mean()
    leastwhitecopdall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENCOPDRATE.mean()
    mostwhitelriall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENLRIRATE.mean()
    leastwhitelriall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENLRIRATE.mean()
    mostwhitemortall = burden19.iloc[np.where(frac_white19 >= 
        np.nanpercentile(frac_white19, 90))].BURDENPMALLRATE.mean()
    leastwhitemortall = burden19.iloc[np.where(frac_white19 <= 
        np.nanpercentile(frac_white19, 10))].BURDENPMALLRATE.mean()
    mosthisppm25all = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].PM25.mean()
    leasthisppm25all = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].PM25.mean()
    mosthispno2all = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].NO2.mean()
    leasthispno2all = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].NO2.mean()
    mosthispafasthmaall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].AFPA.mean()
    leasthispafasthmaall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].AFPA.mean()
    mosthispasthmaall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENASTHMARATE.mean()
    leasthispasthmaall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENASTHMARATE.mean()
    mosthispihdall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENIHDRATE.mean()
    leasthispihdall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENIHDRATE.mean()
    mosthispstall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENSTRATE.mean()
    leasthispstall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENSTRATE.mean()
    mosthisplcall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENLCRATE.mean()
    leasthisplcall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENLCRATE.mean()
    mosthispdmall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENDMRATE.mean()
    leasthispdmall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENDMRATE.mean()
    mosthispcopdall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENCOPDRATE.mean()
    leasthispcopdall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENCOPDRATE.mean()
    mosthisplriall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENLRIRATE.mean()
    leasthisplriall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENLRIRATE.mean()
    mosthispmortall = burden19.iloc[np.where(frac_hisp19 >= 
        np.nanpercentile(frac_hisp19, 90))].BURDENPMALLRATE.mean()
    leasthispmortall = burden19.iloc[np.where(frac_hisp19 <= 
        np.nanpercentile(frac_hisp19, 10))].BURDENPMALLRATE.mean()
    
    # Lists for MSA-specific rates or concentrations for population subgroup
    # extremes
    mostwhitepm25msa, leastwhitepm25msa = [], []
    mostwhiteno2msa, leastwhiteno2msa = [], []
    mostwhiteihdmsa, leastwhiteihdmsa = [], []
    mostwhitestmsa, leastwhitestmsa = [], []
    mostwhitelcmsa, leastwhitelcmsa = [], []
    mostwhitedmmsa, leastwhitedmmsa = [], []
    mostwhitecopdmsa, leastwhitecopdmsa = [], []
    mostwhitelrimsa, leastwhitelrimsa = [], []
    mostwhiteafasthmamsa, leastwhiteafasthmamsa = [], []
    mostwhiteasthmamsa, leastwhiteasthmamsa = [], []
    mostwhitemortmsa, leastwhitemortmsa = [], []
    mosthisppm25msa, leasthisppm25msa = [], []
    mosthispno2msa, leasthispno2msa = [], []
    mosthispihdmsa, leasthispihdmsa = [], []
    mosthispstmsa, leasthispstmsa = [], []
    mosthisplcmsa, leasthisplcmsa = [], []
    mosthispdmmsa, leasthispdmmsa = [], []
    mosthispcopdmsa, leasthispcopdmsa = [], []
    mosthisplrimsa, leasthisplrimsa = [], []
    mosthispafasthmamsa, leasthispafasthmamsa = [], []
    mosthispasthmamsa, leasthispasthmamsa = [], []
    mosthispmortmsa, leasthispmortmsa = [], []

    # Lists for MSA-specific rates or concentrations for population-weighted
    # values
    whitepwpm25msa, blackpwpm25msa = [], []
    whitepwno2msa, blackpwno2msa = [], []
    whitepwihdmsa, blackpwihdmsa = [], []
    whitepwstmsa, blackpwstmsa = [], []
    whitepwlcmsa, blackpwlcmsa = [], []
    whitepwdmmsa, blackpwdmmsa = [], []
    whitepwcopdmsa, blackpwcopdmsa = [], []
    whitepwlrimsa, blackpwlrimsa = [], []
    whitepwafasthmamsa, blackpwafasthmamsa = [], []
    whitepwasthmamsa, blackpwasthmamsa = [], []
    whitepwmortmsa, blackpwmortmsa = [], []
    hpwpm25msa, nhpwpm25msa = [], []
    hpwno2msa, nhpwno2msa = [], []
    hpwihdmsa, nhpwihdmsa = [], []
    hpwstmsa, nhpwstmsa = [], []
    hpwlcmsa, nhpwlcmsa = [], []
    hpwdmmsa, nhpwdmmsa = [], []
    hpwcopdmsa, nhpwcopdmsa = [], []
    hpwlrimsa, nhpwlrimsa = [], []
    hpwafasthmamsa, nhpwafasthmamsa = [], []
    hpwasthmamsa, nhpwasthmamsa = [], []
    hpwmortmsa, nhpwmortmsa = [], []
    msas = []
    geoids = burden19.index.values
    for msa in pm25no2_constants.majors:
        # if msa in ['Urban Honolulu, HI', 'Anchorage, AK',
        #     'Kahului-Wailuku-Lahaina, HI', 'Fairbanks, AK',
        #     'San Juan-Carolina-Caguas, PR', 'Aguadilla-Isabela, PR',
        #     'Ponce, PR', 'Arecibo, PR', 'San Germán, PR', 'Mayagüez, PR',
        #     'Guayama, PR']:
        #     pass
        # else:
        msas.append(msa)
        crosswalk_msa = crosswalk.loc[crosswalk['MSA Title']==msa]
        geoids_msa = []
        for prefix in crosswalk_msa['County Code'].values: 
            prefix = str(prefix).zfill(5)
            incounty = [x for x in geoids if x.startswith(prefix)]
            geoids_msa.append(incounty)
        geoids_msa = sum(geoids_msa, [])
        # MSA-specific tracts and demographics
        harm_imsa19 = burden19.loc[burden19.index.isin(geoids_msa)]
        frac_white19 = ((harm_imsa19[['race_nh_white','race_h_white']].sum(axis=1))/
            harm_imsa19['race_tot'])
        frac_hisp19 = (harm_imsa19['race_h_white']/harm_imsa19['race_tot'])
        # Concentrations and burdens for population subgroup extremes
        mostwhitepm25msa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].PM25.mean())
        leastwhitepm25msa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].PM25.mean())
        mostwhiteno2msa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].NO2.mean())
        leastwhiteno2msa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].NO2.mean())
        mostwhiteihdmsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].BURDENIHDRATE.mean())
        leastwhiteihdmsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENIHDRATE.mean())
        mostwhitestmsa.append(harm_imsa19.iloc[np.where(frac_white19 >=
            np.nanpercentile(frac_white19, 90))].BURDENSTRATE.mean())
        leastwhitestmsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENSTRATE.mean())
        mostwhitelcmsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].BURDENLCRATE.mean())
        leastwhitelcmsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENLCRATE.mean())
        mostwhitedmmsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].BURDENDMRATE.mean())
        leastwhitedmmsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENDMRATE.mean())
        mostwhitecopdmsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].BURDENCOPDRATE.mean())
        leastwhitecopdmsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENCOPDRATE.mean())
        mostwhitelrimsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].BURDENLRIRATE.mean())
        leastwhitelrimsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENLRIRATE.mean())    
        mostwhiteafasthmamsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].AFPA.mean())
        leastwhiteafasthmamsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].AFPA.mean())
        mostwhiteasthmamsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].BURDENASTHMARATE.mean())
        leastwhiteasthmamsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENASTHMARATE.mean())
        mostwhitemortmsa.append(harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))].BURDENPMALLRATE.mean())
        leastwhitemortmsa.append(harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))].BURDENPMALLRATE.mean())    
        # Concentrations and burdens for ethnic groups
        mosthisppm25msa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].PM25.mean())
        leasthisppm25msa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].PM25.mean())
        mosthispno2msa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].NO2.mean())
        leasthispno2msa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].NO2.mean())
        mosthispihdmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENIHDRATE.mean())
        leasthispihdmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENIHDRATE.mean())
        mosthispstmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENSTRATE.mean())
        leasthispstmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENSTRATE.mean())
        mosthisplcmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENLCRATE.mean())
        leasthisplcmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENLCRATE.mean())
        mosthispdmmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENDMRATE.mean())
        leasthispdmmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENDMRATE.mean())
        mosthispcopdmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENCOPDRATE.mean())
        leasthispcopdmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENCOPDRATE.mean())
        mosthisplrimsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENLRIRATE.mean())
        leasthisplrimsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENLRIRATE.mean())    
        mosthispafasthmamsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].AFPA.mean())
        leasthispafasthmamsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].AFPA.mean())
        mosthispasthmamsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENASTHMARATE.mean())
        leasthispasthmamsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENASTHMARATE.mean())
        mosthispmortmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 >= 
            np.nanpercentile(frac_hisp19, 90))].BURDENPMALLRATE.mean())
        leasthispmortmsa.append(harm_imsa19.iloc[np.where(frac_hisp19 <= 
            np.nanpercentile(frac_hisp19, 10))].BURDENPMALLRATE.mean())
        # Population-weighted concentrations and burdens 
        (pm25black19i, no2black19i, copdblack19i, ihdblack19i, lriblack19i, 
         dmblack19i, lcblack19i, stblack19i, afasthmablack19i, asthmablack19i, 
         allmortblack19i, pm25white19i, no2white19i, copdwhite19i, ihdwhite19i, 
         lriwhite19i, dmwhite19i, lcwhite19i, stwhite19i, afasthmawhite19i, 
         asthmawhite19i, allmortwhite19i, pm25nh19i, no2nh19i, copdnh19i, 
         ihdnh19i, lrinh19i, dmnh19i, lcnh19i, stnh19i, afasthmanh19i, 
         asthmanh19i, allmortnh19i, pm25h19i, no2h19i, copdh19i, ihdh19i, 
         lrih19i, dmh19i, lch19i, sth19i, afasthmah19i, asthmah19i, 
         allmorth19i) = pollhealthdisparities(harm_imsa19)
        whitepwpm25msa.append(pm25white19i)
        blackpwpm25msa.append(pm25black19i)
        whitepwno2msa.append(no2white19i)
        blackpwno2msa.append(no2black19i)
        whitepwihdmsa.append(ihdwhite19i)
        blackpwihdmsa.append(ihdblack19i)
        whitepwstmsa.append(stwhite19i)
        blackpwstmsa.append(stblack19i)
        whitepwlcmsa.append(lcwhite19i)
        blackpwlcmsa.append(lcblack19i)
        whitepwdmmsa.append(dmwhite19i)
        blackpwdmmsa.append(dmblack19i)
        whitepwcopdmsa.append(copdwhite19i)
        blackpwcopdmsa.append(copdblack19i)
        whitepwlrimsa.append(lriwhite19i)
        blackpwlrimsa.append(lriblack19i)
        whitepwafasthmamsa.append(afasthmawhite19i)
        blackpwafasthmamsa.append(afasthmablack19i)
        whitepwasthmamsa.append(asthmawhite19i)
        blackpwasthmamsa.append(asthmablack19i)
        whitepwmortmsa.append(allmortwhite19i)
        blackpwmortmsa.append(allmortblack19i)
        hpwpm25msa.append(pm25h19i)
        nhpwpm25msa.append(pm25nh19i)
        hpwno2msa.append(no2h19i)
        nhpwno2msa.append(no2nh19i)
        hpwihdmsa.append(ihdh19i)
        nhpwihdmsa.append(ihdnh19i)
        hpwstmsa.append(sth19i)
        nhpwstmsa.append(stnh19i)
        hpwlcmsa.append(lch19i)
        nhpwlcmsa.append(lcnh19i)
        hpwdmmsa.append(dmh19i)
        nhpwdmmsa.append(dmnh19i)
        hpwcopdmsa.append(copdh19i)
        nhpwcopdmsa.append(copdnh19i)
        hpwlrimsa.append(lrih19i)
        nhpwlrimsa.append(lrinh19i)
        hpwafasthmamsa.append(afasthmah19i)
        nhpwafasthmamsa.append(afasthmanh19i)
        hpwasthmamsa.append(asthmah19i)
        nhpwasthmamsa.append(asthmanh19i)
        hpwmortmsa.append(allmorth19i)
        nhpwmortmsa.append(allmortnh19i)
            
    # # # # # Save disparities file for EDF
    # edf = pd.DataFrame(list(zip(msas, 
    #     whitepwno2msa, mostwhiteno2msa, 
    #     blackpwno2msa, leastwhiteno2msa, 
    #     nhpwno2msa, leasthispno2msa, 
    #     hpwno2msa, mosthispno2msa,
    #     whitepwafasthmamsa, mostwhiteafasthmamsa, 
    #     blackpwafasthmamsa, leastwhiteafasthmamsa, 
    #     nhpwafasthmamsa, leasthispafasthmamsa, 
    #     hpwafasthmamsa, mosthispafasthmamsa,        
    #     whitepwasthmamsa, mostwhiteasthmamsa, 
    #     blackpwasthmamsa, leastwhiteasthmamsa, 
    #     nhpwasthmamsa, leasthispasthmamsa, 
    #     hpwasthmamsa, mosthispasthmamsa)),
    #     columns = ['MSA', 'no2_white_popwtg', 'no2_white_most', 
    #     'no2_black_popwtg', 'no2_white_least',
    #     'no2_nonhispanic_popwtg', 'no2_hispanic_least',
    #     'no2_hispanic_popwtg', 'no2_hispanic_most',
    #     'paf_white_popwtg', 'paf_white_most', 
    #     'paf_black_popwtg', 'paf_white_least',
    #     'paf_nonhispanic_popwtg', 'paf_hispanic_least',
    #     'paf_hispanic_popwtg', 'paf_hispanic_most',    
    #     'asthmaper100k_white_popwtg', 'asthmaper100k_white_most', 
    #     'asthmaper100k_black_popwtg', 'asthmaper100k_white_least',
    #     'asthmaper100k_nonhispanic_popwtg', 'asthmaper100k_hispanic_least',
    #     'asthmaper100k_hispanic_popwtg', 'asthmaper100k_hispanic_most'])    
    # # Add row for all MSA average 
    # edf_allmsa = {'MSA':"All MSAs", 
    #     'no2_white_popwtg':no2white19, 
    #     'no2_white_most':mostwhiteno2all, 
    #     'no2_black_popwtg':no2black19, 
    #     'no2_white_least':leastwhiteno2all,
    #     'no2_nonhispanic_popwtg':no2nh19, 
    #     'no2_hispanic_least':leasthispno2all,
    #     'no2_hispanic_popwtg':no2h19, 
    #     'no2_hispanic_most':mosthispno2all,
    #     'paf_white_popwtg':afasthmawhite19, 
    #     'paf_white_most':mostwhiteafasthmaall, 
    #     'paf_black_popwtg':afasthmablack19, 
    #     'paf_white_least':leastwhiteafasthmaall,
    #     'paf_nonhispanic_popwtg':afasthmanh19, 
    #     'paf_hispanic_least':leasthispafasthmaall,
    #     'paf_hispanic_popwtg':afasthmah19, 
    #     'paf_hispanic_most':mosthispafasthmaall,  
    #     'asthmaper100k_white_popwtg':asthmawhite19, 
    #     'asthmaper100k_white_most':mostwhiteasthmaall, 
    #     'asthmaper100k_black_popwtg':asthmablack19, 
    #     'asthmaper100k_white_least':leastwhiteasthmaall,
    #     'asthmaper100k_nonhispanic_popwtg':asthmanh19, 
    #     'asthmaper100k_hispanic_least':leasthispasthmaall,
    #     'asthmaper100k_hispanic_popwtg':asthmah19, 
    #     'asthmaper100k_hispanic_most':mosthispasthmaall
    #     }
    # edf = edf.append(edf_allmsa, ignore_index = True)
    # edf.set_index('MSA', inplace=True)
    # edf.to_csv(DIR+'docs/'+'disparities_mostleast_popweighted.csv', sep=',', 
    #     encoding='utf-8')
    
    # # # # For population-weighted
    fig = plt.figure(figsize=(12,7))
    ax1 = plt.subplot2grid((10,2),(0,0), rowspan=2)
    ax2 = plt.subplot2grid((10,2),(2,0), rowspan=6)
    ax3 = plt.subplot2grid((10,2),(8,0), rowspan=2)
    ax4 = plt.subplot2grid((10,2),(0,1), rowspan=2)
    ax5 = plt.subplot2grid((10,2),(2,1), rowspan=6)
    ax6 = plt.subplot2grid((10,2),(8,1), rowspan=2)
    # Titles
    ax1.set_title('(A) Racial disparities', fontsize=14, loc='left')
    ax4.set_title('(B) Ethnic disparities', fontsize=14, loc='left')
    ax1ticks, ax2ticks, ax3ticks = [], [], []
    ypos = 10 
    ax1ticks.append(ypos)
    # Plotting racial disparities
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwpm25msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwpm25msa))
    # MSA-specific values
    ax1.plot(reject_outliers(whitepwpm25msa), y1, '.', color=color3, 
        alpha=0.05, zorder=9)
    ax1.plot(reject_outliers(blackpwpm25msa), y2, '.', color=color2, 
        alpha=0.05, zorder=9)
    # Significance using K-S test
    ks = ks_2samp(np.array(whitepwpm25msa)[~np.isnan(whitepwpm25msa)], 
        np.array(blackpwpm25msa)[~np.isnan(blackpwpm25msa)])
    if ks.pvalue > 0.05:
        ax1.axhspan(ypos-0.75, ypos+0.75, alpha=0.3, color='lightgrey', zorder=0)
    # Mean values
    ax1.plot(pm25white19, ypos+0.2, 'o', color=color3, zorder=10)
    ax1.plot(pm25black19, ypos-0.2, 'o', color=color2, zorder=10)
    # # Quasi legend
    ax1.text(11, ypos-0.2, 'Black', va='center', color=color2, zorder=11)
    ax1.text(2, ypos+0.2, 'White', va='center', color=color3, zorder=11)
    ypos = ypos - 1.5
    ax1ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwno2msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwno2msa))
    ax1.plot(reject_outliers(whitepwno2msa), y1, '.', color=color3, 
        alpha=0.05, zorder=10)
    ax1.plot(reject_outliers(blackpwno2msa), y2, '.', color=color2, 
        alpha=0.05, zorder=10)
    ks = ks_2samp(np.array(whitepwno2msa)[~np.isnan(whitepwno2msa)], 
        np.array(blackpwno2msa)[~np.isnan(blackpwno2msa)])
    if ks.pvalue > 0.05:
        ax1.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax1.plot(no2white19, ypos+0.2, 'o', color=color3, zorder=10)
    ax1.plot(no2black19, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial lower respiratory infection 
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwlrimsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwlrimsa))
    ax2.plot(reject_outliers(whitepwlrimsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(blackpwlrimsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(whitepwlrimsa)[~np.isnan(whitepwlrimsa)], 
        np.array(blackpwlrimsa)[~np.isnan(blackpwlrimsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(lriwhite19, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(lriblack19, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial type 2 diabetes
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwdmmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwdmmsa))
    ax2.plot(reject_outliers(whitepwdmmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(blackpwdmmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(whitepwdmmsa)[~np.isnan(whitepwdmmsa)], 
        np.array(blackpwdmmsa)[~np.isnan(blackpwdmmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(dmwhite19, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(dmblack19, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial COPD
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwcopdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwcopdmsa))
    ax2.plot(reject_outliers(whitepwcopdmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(blackpwcopdmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(whitepwcopdmsa)[~np.isnan(whitepwcopdmsa)], 
        np.array(blackpwcopdmsa)[~np.isnan(blackpwcopdmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(copdwhite19, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(copdblack19, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial lung cancer
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwlcmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwlcmsa))
    ax2.plot(reject_outliers(whitepwlcmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(blackpwlcmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(whitepwlcmsa)[~np.isnan(whitepwlcmsa)], 
        np.array(blackpwlcmsa)[~np.isnan(blackpwlcmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(lcwhite19, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(lcblack19, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial stroke
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwstmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwstmsa))
    ax2.plot(reject_outliers(whitepwstmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(blackpwstmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(whitepwstmsa)[~np.isnan(whitepwstmsa)], 
        np.array(blackpwstmsa)[~np.isnan(blackpwstmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(stwhite19, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(stblack19, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial ischemic heart disease
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(whitepwihdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(blackpwihdmsa))
    ax2.plot(reject_outliers(whitepwihdmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=True, zorder=10)
    ax2.plot(reject_outliers(blackpwihdmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=True, zorder=10)
    ks = ks_2samp(np.array(whitepwihdmsa)[~np.isnan(whitepwihdmsa)], 
        np.array(blackpwihdmsa)[~np.isnan(blackpwihdmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(ihdwhite19, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(ihdblack19, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial asthma
    ypos = ypos - 1.5
    ax3ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, 0.05, size=len(whitepwasthmamsa))
    y2 = np.random.normal(ypos-0.2, 0.05, size=len(blackpwasthmamsa))
    ax3.plot(reject_outliers(whitepwasthmamsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=True, zorder=10)
    ax3.plot(reject_outliers(blackpwasthmamsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=True, zorder=10)
    ks = ks_2samp(np.array(whitepwasthmamsa)[~np.isnan(whitepwasthmamsa)], 
        np.array(blackpwasthmamsa)[~np.isnan(blackpwasthmamsa)])
    if ks.pvalue > 0.05:
        ax3.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax3.plot(asthmawhite19, ypos+0.2, 'o', color=color3, zorder=10)
    ax3.plot(asthmablack19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic disparities
    ax4ticks, ax5ticks, ax6ticks = [], [], []
    ypos = 10 
    ax4ticks.append(ypos)
    # Ethnic PM25
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwpm25msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwpm25msa))
    ax4.plot(reject_outliers(nhpwpm25msa), y1, '.', color=color3, 
        alpha=0.05, clip_on=True, zorder=9)
    ax4.plot(reject_outliers(hpwpm25msa), y2, '.', color=color2, 
        alpha=0.05, clip_on=True, zorder=9)
    ks = ks_2samp(np.array(nhpwpm25msa)[~np.isnan(nhpwpm25msa)], 
        np.array(hpwpm25msa)[~np.isnan(hpwpm25msa)])
    if ks.pvalue > 0.05:
        ax4.axhspan(ypos-0.7, ypos+1.1, alpha=0.3, color='lightgrey', zorder=0)
    ax4.plot(pm25nh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax4.plot(pm25h19, ypos-0.2, 'o', color=color2, zorder=10)
    ax4.text(11, ypos-0.2, 'Hispanic', va='center', color=color2, zorder=11)
    ax4.text(0.5, ypos+0.2, 'Non-Hispanic', va='center', color=color3, zorder=11)
    ypos = ypos - 1.5
    ax4ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwno2msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwno2msa))
    ax4.plot(reject_outliers(nhpwno2msa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax4.plot(reject_outliers(hpwno2msa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(nhpwno2msa)[~np.isnan(nhpwno2msa)], 
        np.array(hpwno2msa)[~np.isnan(hpwno2msa)])
    if ks.pvalue > 0.05:
        ax4.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax4.plot(no2nh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax4.plot(no2h19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic lower respiratory infection 
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwlrimsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwlrimsa))
    ax5.plot(reject_outliers(nhpwlrimsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(hpwlrimsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(nhpwlrimsa)[~np.isnan(nhpwlrimsa)], 
        np.array(hpwlrimsa)[~np.isnan(hpwlrimsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(lrinh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(lrih19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic type 2 diabetes
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwdmmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwdmmsa))
    ax5.plot(reject_outliers(nhpwdmmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(hpwdmmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(nhpwdmmsa)[~np.isnan(nhpwdmmsa)], 
        np.array(hpwdmmsa)[~np.isnan(hpwdmmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(dmnh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(dmh19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic COPD
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwcopdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwcopdmsa))
    ax5.plot(reject_outliers(nhpwcopdmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(hpwcopdmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(nhpwcopdmsa)[~np.isnan(nhpwcopdmsa)], 
        np.array(hpwcopdmsa)[~np.isnan(hpwcopdmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(copdnh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(copdh19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic lung cancer
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwlcmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwlcmsa))
    ax5.plot(reject_outliers(nhpwlcmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(hpwlcmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(nhpwlcmsa)[~np.isnan(nhpwlcmsa)], 
        np.array(hpwlcmsa)[~np.isnan(hpwlcmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(lcnh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(lch19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic stroke
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwstmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwstmsa))
    ax5.plot(reject_outliers(nhpwstmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(hpwstmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(nhpwstmsa)[~np.isnan(nhpwstmsa)], 
        np.array(hpwstmsa)[~np.isnan(hpwstmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(stnh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(sth19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic ischemic heart disease
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(nhpwihdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(hpwihdmsa))
    ax5.plot(reject_outliers(nhpwihdmsa), y1, '.', color=color3, 
        alpha=0.05, zorder=10)
    ax5.plot(reject_outliers(hpwihdmsa), y2, '.', color=color2, 
        alpha=0.05, zorder=10)
    ks = ks_2samp(np.array(nhpwihdmsa)[~np.isnan(nhpwihdmsa)], 
        np.array(hpwihdmsa)[~np.isnan(hpwihdmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(ihdnh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(ihdh19, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic asthma
    ypos = ypos - 1.5
    ax6ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, 0.05, size=len(nhpwasthmamsa))
    y2 = np.random.normal(ypos-0.2, 0.05, size=len(hpwasthmamsa))
    ax6.plot(reject_outliers(nhpwasthmamsa), y1, '.', color=color3, 
        alpha=0.05, zorder=10)
    ax6.plot(reject_outliers(hpwasthmamsa), y2, '.', color=color2, 
        alpha=0.05, zorder=10)
    ks = ks_2samp(np.array(nhpwasthmamsa)[~np.isnan(nhpwasthmamsa)], 
        np.array(hpwasthmamsa)[~np.isnan(hpwasthmamsa)])
    if ks.pvalue > 0.05:
        ax6.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax6.plot(asthmanh19, ypos+0.2, 'o', color=color3, zorder=10)
    ax6.plot(asthmah19, ypos-0.2, 'o', color=color2, zorder=10)
    # Set axis ticks
    for ax in [ax1, ax4]:
        ax.set_xlim([0,14])
        ax.set_xticks([0,3.5,7,10.5,14])
        ax.set_xticklabels(['0','','7','','14'])
        ax.set_xlabel('Concentration [$\mathregular{\mu}$g m$^'+\
            '{\mathregular{-3}}$  |  ppbv]', loc='left')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0]*0.95, ylim[1]*1.05]) 
    for ax in [ax2, ax5]:
        ax.set_xlim([0,16])
        ax.set_xticks([0,4,8,12,16])
        ax.set_xticklabels(['0','','8','','16'])
        ax.set_xlabel('PM$_{\mathregular{2.5}}$-attributable mortality rate'+\
            ' [per 100,000 population]', loc='left')    
    for ax in [ax3, ax6]:
        ax.set_xlim([0,400])
        ax.set_xticks([0,100,200,300,400])
        ax.set_xticklabels(['0','','200','','400'])    
        ax.set_xlabel('NO$_{\mathregular{2}}$-attributable incidence rate'+\
            ' [per 100,000 pediatric population]', loc='left')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0]*1.2, ylim[1]*0.6])
    ax1ticks.reverse()
    ax2ticks.reverse()
    ax3ticks.reverse()
    ax1.set_yticks(ax1ticks)
    ax1.set_yticklabels(['NO$_{\mathregular{2}}$', 'PM$_{\mathregular{2.5}}$'])
    ax2.set_yticks(ax2ticks)
    ax2.set_yticklabels(['Ischemic heart\ndisease', 'Stroke', 'Lung cancer', 
        'COPD', 'Type 2\ndiabetes', 'Lower respiratory\ninfection'])
    ax3.set_yticks(ax3ticks)
    ax3.set_yticklabels(['Pediatric\nasthma'])
    for ax in [ax4, ax5, ax6]:
        ax.set_yticklabels([])
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)    
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis=u'y', which=u'both', length=0)
    for ax in [ax4, ax5, ax6]:
        # Shift a tiny bit rightward
        pos = ax.get_position()
        pos.x0 = pos.x0+0.05
        pos.x1 = pos.x1+0.05    
        ax.set_position(pos)
    plt.subplots_adjust(right=0.96, hspace=2.7, top=0.95)
    plt.savefig(DIR_FIG+'fig2_popwtg.pdf', dpi=600)

    # # # # For subgroup extremes
    fig = plt.figure(figsize=(12,7))
    ax1 = plt.subplot2grid((10,2),(0,0), rowspan=2)
    ax2 = plt.subplot2grid((10,2),(2,0), rowspan=6)
    ax3 = plt.subplot2grid((10,2),(8,0), rowspan=2)
    ax4 = plt.subplot2grid((10,2),(0,1), rowspan=2)
    ax5 = plt.subplot2grid((10,2),(2,1), rowspan=6)
    ax6 = plt.subplot2grid((10,2),(8,1), rowspan=2)
    # Titles
    ax1.set_title('(A) Racial disparities', fontsize=14, loc='left')
    ax4.set_title('(B) Ethnic disparities', fontsize=14, loc='left')
    ax1ticks, ax2ticks, ax3ticks = [], [], []
    ypos = 10 
    ax1ticks.append(ypos)
    # Plotting racial disparities
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhitepm25msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhitepm25msa))
    # MSA-specific values
    ax1.plot(reject_outliers(mostwhitepm25msa), y1, '.', color=color3, 
        alpha=0.05, zorder=9)
    ax1.plot(reject_outliers(leastwhitepm25msa), y2, '.', color=color2, 
        alpha=0.05, zorder=9)
    # Significance using K-S test
    ks = ks_2samp(np.array(mostwhitepm25msa)[~np.isnan(mostwhitepm25msa)], 
        np.array(leastwhitepm25msa)[~np.isnan(leastwhitepm25msa)])
    if ks.pvalue > 0.05:
        ax1.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    # Mean values
    ax1.plot(mostwhitepm25all, ypos+0.2, 'o', color=color3, zorder=10)
    ax1.plot(leastwhitepm25all, ypos-0.2, 'o', color=color2, zorder=10)
    # Quasi legend
    ax1.text(11, ypos-0.2, 'Least\nwhite', va='center', color=color2, zorder=11)
    ax1.text(1, ypos+0.2, 'Most\nwhite', va='center', color=color3, zorder=11)
    ypos = ypos - 1.5
    ax1ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhiteno2msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhiteno2msa))
    ax1.plot(reject_outliers(mostwhiteno2msa), y1, '.', color=color3, 
        alpha=0.05, zorder=10)
    ax1.plot(reject_outliers(leastwhiteno2msa), y2, '.', color=color2, 
        alpha=0.05, zorder=10)
    ks = ks_2samp(np.array(mostwhiteno2msa)[~np.isnan(mostwhiteno2msa)], 
        np.array(leastwhiteno2msa)[~np.isnan(leastwhiteno2msa)])
    if ks.pvalue > 0.05:
        ax1.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax1.plot(mostwhiteno2all, ypos+0.2, 'o', color=color3, zorder=10)
    ax1.plot(leastwhiteno2all, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial lower respiratory infection 
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhitelrimsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhitelrimsa))
    ax2.plot(reject_outliers(mostwhitelrimsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(leastwhitelrimsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(mostwhitelrimsa)[~np.isnan(mostwhitelrimsa)], 
        np.array(leastwhitelrimsa)[~np.isnan(leastwhitelrimsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(mostwhitelriall, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(leastwhitelriall, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial type 2 diabetes
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhitedmmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhitedmmsa))
    ax2.plot(reject_outliers(mostwhitedmmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(leastwhitedmmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(mostwhitedmmsa)[~np.isnan(mostwhitedmmsa)], 
        np.array(leastwhitedmmsa)[~np.isnan(leastwhitedmmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(mostwhitedmall, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(leastwhitedmall, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial COPD
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhitecopdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhitecopdmsa))
    ax2.plot(reject_outliers(mostwhitecopdmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(leastwhitecopdmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(mostwhitecopdmsa)[~np.isnan(mostwhitecopdmsa)], 
        np.array(leastwhitecopdmsa)[~np.isnan(leastwhitecopdmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(mostwhitecopdall, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(leastwhitecopdall, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial lung cancer
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhitelcmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhitelcmsa))
    ax2.plot(reject_outliers(mostwhitelcmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(leastwhitelcmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(mostwhitelcmsa)[~np.isnan(mostwhitelcmsa)], 
        np.array(leastwhitelcmsa)[~np.isnan(leastwhitelcmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(mostwhitelcall, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(leastwhitelcall, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial stroke
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhitestmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhitestmsa))
    ax2.plot(reject_outliers(mostwhitestmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax2.plot(reject_outliers(leastwhitestmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(mostwhitestmsa)[~np.isnan(mostwhitestmsa)], 
        np.array(leastwhitestmsa)[~np.isnan(leastwhitestmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(mostwhitestall, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(leastwhitestall, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial ischemic heart disease
    ypos = ypos - 1.5
    ax2ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(mostwhiteihdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(leastwhiteihdmsa))
    ax2.plot(reject_outliers(mostwhiteihdmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=True, zorder=10)
    ax2.plot(reject_outliers(leastwhiteihdmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=True, zorder=10)
    ks = ks_2samp(np.array(mostwhiteihdmsa)[~np.isnan(mostwhiteihdmsa)], 
        np.array(leastwhiteihdmsa)[~np.isnan(leastwhiteihdmsa)])
    if ks.pvalue > 0.05:
        ax2.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax2.plot(mostwhiteihdall, ypos+0.2, 'o', color=color3, zorder=10)
    ax2.plot(leastwhiteihdall, ypos-0.2, 'o', color=color2, zorder=10)
    # Racial asthma
    ypos = ypos - 1.5
    ax3ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, 0.05, size=len(mostwhiteasthmamsa))
    y2 = np.random.normal(ypos-0.2, 0.05, size=len(leastwhiteasthmamsa))
    ax3.plot(reject_outliers(mostwhiteasthmamsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=True, zorder=10)
    ax3.plot(reject_outliers(leastwhiteasthmamsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=True, zorder=10)
    ks = ks_2samp(np.array(mostwhiteasthmamsa)[~np.isnan(mostwhiteasthmamsa)], 
        np.array(leastwhiteasthmamsa)[~np.isnan(leastwhiteasthmamsa)])
    if ks.pvalue > 0.05:
        ax3.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax3.plot(mostwhiteasthmaall, ypos+0.2, 'o', color=color3, zorder=10)
    ax3.plot(leastwhiteasthmaall, ypos-0.2, 'o', color=color2, zorder=10)
    
    # Ethnic disparities
    ax4ticks, ax5ticks, ax6ticks = [], [], []
    ypos = 10 
    ax4ticks.append(ypos)
    # Ethnic PM25
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthisppm25msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthisppm25msa))
    ax4.plot(reject_outliers(leasthisppm25msa), y1, '.', color=color3, 
        alpha=0.05, clip_on=True, zorder=9)
    ax4.plot(reject_outliers(mosthisppm25msa), y2, '.', color=color2, 
        alpha=0.05, clip_on=True, zorder=9)
    ks = ks_2samp(np.array(leasthisppm25msa)[~np.isnan(leasthisppm25msa)], 
        np.array(mosthisppm25msa)[~np.isnan(mosthisppm25msa)])
    if ks.pvalue > 0.05:
        ax4.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax4.plot(leasthisppm25all, ypos+0.2, 'o', color=color3, zorder=10)
    ax4.plot(mosthisppm25all, ypos-0.2, 'o', color=color2, zorder=10)
    ax4.text(11, ypos-0.2, 'Most\nHispanic', va='center', color=color2, zorder=11)
    ax4.text(1, ypos+0.2, 'Least\nHispanic', va='center', color=color3, zorder=11)
    ypos = ypos - 1.5
    ax4ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthispno2msa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthispno2msa))
    ax4.plot(reject_outliers(leasthispno2msa), y1, '.', color=color3, 
        alpha=0.05, clip_on=True, zorder=10)
    ax4.plot(reject_outliers(mosthispno2msa), y2, '.', color=color2, 
        alpha=0.05, clip_on=True, zorder=10)
    ks = ks_2samp(np.array(leasthispno2msa)[~np.isnan(leasthispno2msa)], 
        np.array(mosthispno2msa)[~np.isnan(mosthispno2msa)])
    if ks.pvalue > 0.05:
        ax4.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax4.plot(leasthispno2all, ypos+0.2, 'o', color=color3, zorder=10)
    ax4.plot(mosthispno2all, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic lower respiratory infection 
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthisplrimsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthisplrimsa))
    ax5.plot(reject_outliers(leasthisplrimsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(mosthisplrimsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(leasthisplrimsa)[~np.isnan(leasthisplrimsa)], 
        np.array(mosthisplrimsa)[~np.isnan(mosthisplrimsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(leasthisplriall, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(mosthisplriall, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic type 2 diabetes
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthispdmmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthispdmmsa))
    ax5.plot(reject_outliers(leasthispdmmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(mosthispdmmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(leasthispdmmsa)[~np.isnan(leasthispdmmsa)], 
        np.array(mosthispdmmsa)[~np.isnan(mosthispdmmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(leasthispdmall, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(mosthispdmall, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic COPD
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthispcopdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthispcopdmsa))
    ax5.plot(reject_outliers(leasthispcopdmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(mosthispcopdmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(leasthispcopdmsa)[~np.isnan(leasthispcopdmsa)], 
        np.array(mosthispcopdmsa)[~np.isnan(mosthispcopdmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(leasthispcopdall, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(mosthispcopdall, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic lung cancer
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthisplcmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthisplcmsa))
    ax5.plot(reject_outliers(leasthisplcmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(mosthisplcmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(leasthisplcmsa)[~np.isnan(leasthisplcmsa)], 
        np.array(mosthisplcmsa)[~np.isnan(mosthisplcmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(leasthisplcall, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(mosthisplcall, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic stroke
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthispstmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthispstmsa))
    ax5.plot(reject_outliers(leasthispstmsa), y1, '.', color=color3, 
        alpha=0.05, clip_on=False, zorder=10)
    ax5.plot(reject_outliers(mosthispstmsa), y2, '.', color=color2, 
        alpha=0.05, clip_on=False, zorder=10)
    ks = ks_2samp(np.array(leasthispstmsa)[~np.isnan(leasthispstmsa)], 
        np.array(mosthispstmsa)[~np.isnan(mosthispstmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(leasthispstall, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(mosthispstall, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic ischemic heart disease
    ypos = ypos - 1.5
    ax5ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, jitter, size=len(leasthispihdmsa))
    y2 = np.random.normal(ypos-0.2, jitter, size=len(mosthispihdmsa))
    ax5.plot(reject_outliers(leasthispihdmsa), y1, '.', color=color3, 
        alpha=0.05, zorder=10)
    ax5.plot(reject_outliers(mosthispihdmsa), y2, '.', color=color2, 
        alpha=0.05, zorder=10)
    ks = ks_2samp(np.array(leasthispihdmsa)[~np.isnan(leasthispihdmsa)], 
        np.array(mosthispihdmsa)[~np.isnan(mosthispihdmsa)])
    if ks.pvalue > 0.05:
        ax5.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax5.plot(leasthispihdall, ypos+0.2, 'o', color=color3, zorder=10)
    ax5.plot(mosthispihdall, ypos-0.2, 'o', color=color2, zorder=10)
    # Ethnic asthma
    ypos = ypos - 1.5
    ax6ticks.append(ypos)
    y1 = np.random.normal(ypos+0.2, 0.05, size=len(leasthispasthmamsa))
    y2 = np.random.normal(ypos-0.2, 0.05, size=len(mosthispasthmamsa))
    ax6.plot(reject_outliers(leasthispasthmamsa), y1, '.', color=color3, 
        alpha=0.05, zorder=10)
    ax6.plot(reject_outliers(mosthispasthmamsa), y2, '.', color=color2, 
        alpha=0.05, zorder=10)
    ks = ks_2samp(np.array(leasthispasthmamsa)[~np.isnan(leasthispasthmamsa)], 
        np.array(mosthispasthmamsa)[~np.isnan(mosthispasthmamsa)])
    if ks.pvalue > 0.05:
        ax6.axhspan(ypos-0.5, ypos+0.5, alpha=0.3, color='lightgrey', zorder=0)
    ax6.plot(leasthispasthmaall, ypos+0.2, 'o', color=color3, zorder=10)
    ax6.plot(mosthispasthmaall, ypos-0.2, 'o', color=color2, zorder=10)
    # Set axis ticks
    for ax in [ax1, ax4]:
        ax.set_xlim([0,14])
        ax.set_xticks([0,3.5,7,10.5,14])
        ax.set_xticklabels(['0','','7','','14'])
        ax.set_xlabel('Concentration [$\mathregular{\mu}$g m$^'+\
            '{\mathregular{-3}}$  |  ppbv]', loc='left')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0]*0.95, ylim[1]*1.05]) 
    for ax in [ax2, ax5]:
        ax.set_xlim([0,16])
        ax.set_xticks([0,4,8,12,16])
        ax.set_xticklabels(['0','','8','','16'])
        ax.set_xlabel('PM$_{\mathregular{2.5}}$-attributable mortality rate'+\
            ' [per 100,000 population]', loc='left')    
    for ax in [ax3, ax6]:
        ax.set_xlim([0,400])
        ax.set_xticks([0,100,200,300,400])
        ax.set_xticklabels(['0','','200','','400'])    
        ax.set_xlabel('NO$_{\mathregular{2}}$-attributable incidence rate'+\
            ' [per 100,000 pediatric population]', loc='left')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0]*1.2, ylim[1]*0.6])
    ax1ticks.reverse()
    ax2ticks.reverse()
    ax3ticks.reverse()
    ax1.set_yticks(ax1ticks)
    ax1.set_yticklabels(['NO$_{\mathregular{2}}$', 'PM$_{\mathregular{2.5}}$'])
    ax2.set_yticks(ax2ticks)
    ax2.set_yticklabels(['Ischemic heart\ndisease', 'Stroke', 'Lung cancer', 
        'COPD', 'Type 2\ndiabetes', 'Lower respiratory\ninfection'])
    ax3.set_yticks(ax3ticks)
    ax3.set_yticklabels(['Pediatric\nasthma'])
    for ax in [ax4, ax5, ax6]:
        ax.set_yticklabels([])
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)    
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis=u'y', which=u'both', length=0)
    for ax in [ax4, ax5, ax6]:
        # Shift a tiny bit rightward
        pos = ax.get_position()
        pos.x0 = pos.x0+0.05
        pos.x1 = pos.x1+0.05    
        ax.set_position(pos)    
    plt.subplots_adjust(right=0.96, hspace=2.7, top=0.95)
    plt.savefig(DIR_FIG+'fig2.pdf', dpi=600)
    return    

def fig3(burdents):
    """

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp    
    # Select burdens from beginning, midpoint, and final years of analysis
    # (2010, 2015, 2019) 
    burden10 = burdents.loc[burdents['YEAR']=='2006-2010'].copy(deep=True)
    burden15 = burdents.loc[burdents['YEAR']=='2011-2015'].copy(deep=True)
    burden19 = burdents.loc[burdents['YEAR']=='2015-2019'].copy(deep=True)
    # Calculate racial characteristics
    fracwhite10 = ((burden10[['race_nh_white','race_h_white']].sum(axis=1))/
        burden10['race_tot'])
    burden10['fracwhite10'] = fracwhite10
    fracwhite15 = ((burden15[['race_nh_white','race_h_white']].sum(axis=1))/
        burden15['race_tot'])
    burden15['fracwhite15'] = fracwhite15
    fracwhite19 = ((burden19[['race_nh_white','race_h_white']].sum(axis=1))/
        burden19['race_tot'])
    burden19['fracwhite19'] = fracwhite19
    # Find tracts with largest NO2-, PM25-, or PM25- and NO2-attributable 
    # health burdens
    # For 2010
    whereleast10 = burden10.loc[
        (burden10.BURDENASTHMARATE<=np.nanpercentile(burden10.BURDENASTHMARATE,10)) & 
        (burden10.BURDENPMALLRATE<=np.nanpercentile(burden10.BURDENPMALLRATE,10))]
    wheremost10 = burden10.loc[
        (burden10.BURDENASTHMARATE>=np.nanpercentile(burden10.BURDENASTHMARATE,90)) & 
        (burden10.BURDENPMALLRATE>=np.nanpercentile(burden10.BURDENPMALLRATE,90))]
    wheremostpm10 = burden10.loc[burden10.BURDENPMALLRATE>=
        np.nanpercentile(burden10.BURDENPMALLRATE,90)]
    wheremostasthma10 = burden10.loc[burden10.BURDENASTHMARATE>=
        np.nanpercentile(burden10.BURDENASTHMARATE,90)]
    whereleastpm10 = burden10.loc[burden10.BURDENPMALLRATE<=
        np.nanpercentile(burden10.BURDENPMALLRATE,10)]
    whereleastasthma10 = burden10.loc[burden10.BURDENASTHMARATE<=
        np.nanpercentile(burden10.BURDENASTHMARATE,10)]
    # For 2015
    whereleast15 = burden15.loc[
        (burden15.BURDENASTHMARATE<=np.nanpercentile(burden15.BURDENASTHMARATE,10)) & 
        (burden15.BURDENPMALLRATE<=np.nanpercentile(burden15.BURDENPMALLRATE,10))]
    wheremost15 = burden15.loc[
        (burden15.BURDENASTHMARATE>=np.nanpercentile(burden15.BURDENASTHMARATE,90)) & 
        (burden15.BURDENPMALLRATE>=np.nanpercentile(burden15.BURDENPMALLRATE,90))]
    wheremostpm15 = burden15.loc[burden15.BURDENPMALLRATE>=
        np.nanpercentile(burden15.BURDENPMALLRATE,90)]
    wheremostasthma15 = burden15.loc[burden15.BURDENASTHMARATE>=
        np.nanpercentile(burden15.BURDENASTHMARATE,90)]
    whereleastpm15 = burden15.loc[burden15.BURDENPMALLRATE<=
        np.nanpercentile(burden15.BURDENPMALLRATE,10)]
    whereleastasthma15 = burden15.loc[burden15.BURDENASTHMARATE<=
        np.nanpercentile(burden15.BURDENASTHMARATE,10)]
    # For 2019 
    whereleast19 = burden19.loc[
        (burden19.BURDENASTHMARATE<=np.nanpercentile(burden19.BURDENASTHMARATE,10)) & 
        (burden19.BURDENPMALLRATE<=np.nanpercentile(burden19.BURDENPMALLRATE,10))]
    wheremost19 = burden19.loc[
        (burden19.BURDENASTHMARATE>=np.nanpercentile(burden19.BURDENASTHMARATE,90)) & 
        (burden19.BURDENPMALLRATE>=np.nanpercentile(burden19.BURDENPMALLRATE,90))]
    wheremostpm19 = burden19.loc[burden19.BURDENPMALLRATE>=
        np.nanpercentile(burden19.BURDENPMALLRATE,90)]
    wheremostasthma19 = burden19.loc[burden19.BURDENASTHMARATE>=
        np.nanpercentile(burden19.BURDENASTHMARATE,90)]
    whereleastpm19 = burden19.loc[burden19.BURDENPMALLRATE<=
        np.nanpercentile(burden19.BURDENPMALLRATE,10)]
    whereleastasthma19 = burden19.loc[burden19.BURDENASTHMARATE<=
        np.nanpercentile(burden19.BURDENASTHMARATE,10)]
    colorsyear = [color1, color2, color3]
    # Plotting
    fig = plt.figure(figsize=(8,5))
    ax1 = plt.subplot2grid((1,1), (0,0))
    # Denote median percent white in years 
    ax1.axhline(burden10.fracwhite10.median(), ls='-', color=colorsyear[0], 
        zorder=0, lw=0.75)
    ax1.axhline(burden15.fracwhite15.median(), ls='-', color=colorsyear[1], 
        zorder=0, lw=0.75) 
    ax1.axhline(burden19.fracwhite19.median(), ls='-', color=colorsyear[2], 
        zorder=0, lw=0.75)
    # Racial composition of tracts with the smallest PM2.5-
    # attributable burdens
    heights = [whereleastpm10.fracwhite10, whereleastpm15.fracwhite15, 
        whereleastpm19.fracwhite19]
    pos = [0,1,2]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance 
    ks_1015 = ks_2samp(whereleastpm10.fracwhite10, whereleastpm15.fracwhite15)
    ks_1519 = ks_2samp(whereleastpm15.fracwhite15, whereleastpm19.fracwhite19)
    ks_1019 = ks_2samp(whereleastpm10.fracwhite10, whereleastpm19.fracwhite19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the smallest NO2-attributable 
    # burdens
    heights = [whereleastasthma10.fracwhite10, 
        whereleastasthma15.fracwhite15, whereleastasthma19.fracwhite19]
    pos = [4,5,6]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(whereleastasthma10.fracwhite10, 
        whereleastasthma15.fracwhite15)
    ks_1519 = ks_2samp(whereleastasthma15.fracwhite15, 
        whereleastasthma19.fracwhite19)
    ks_1019 = ks_2samp(whereleastasthma10.fracwhite10, 
        whereleastasthma19.fracwhite19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the smallest NO2- and PM2.5-
    # attributable burdens
    heights = [whereleast10.fracwhite10, whereleast15.fracwhite15, 
        whereleast19.fracwhite19]
    pos = [8,9,10]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(whereleast10.fracwhite10, whereleast15.fracwhite15)
    ks_1519 = ks_2samp(whereleast15.fracwhite15, whereleast19.fracwhite19)
    ks_1019 = ks_2samp(whereleast10.fracwhite10, whereleast19.fracwhite19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the largest PM2.5-attributable burdens    
    heights = [wheremostpm10.fracwhite10, wheremostpm15.fracwhite15, 
        wheremostpm19.fracwhite19]
    pos = [13,14,15]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(wheremostpm10.fracwhite10, wheremostpm15.fracwhite15)
    ks_1519 = ks_2samp(wheremostpm15.fracwhite15, wheremostpm19.fracwhite19)
    ks_1019 = ks_2samp(wheremostpm10.fracwhite10, wheremostpm19.fracwhite19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the largest NO2-attributable burdens    
    heights = [wheremostasthma10.fracwhite10, 
        wheremostasthma15.fracwhite15, wheremostasthma19.fracwhite19]
    pos = [17,18,19]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(wheremostasthma10.fracwhite10, 
        wheremostasthma15.fracwhite15)
    ks_1519 = ks_2samp(wheremostasthma15.fracwhite15, 
        wheremostasthma19.fracwhite19)
    ks_1019 = ks_2samp(wheremostasthma10.fracwhite10, 
        wheremostasthma19.fracwhite19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.08, maxasterix=(3))
    # Racial composition of tracts with the largest NO2- and PM2.5-attributable burdens    
    heights = [wheremost10.fracwhite10, wheremost15.fracwhite15, 
        wheremost19.fracwhite19]
    pos = [21,22,23]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(wheremost10.fracwhite10, wheremost15.fracwhite15)
    ks_1519 = ks_2samp(wheremost15.fracwhite15, wheremost19.fracwhite19)
    ks_1019 = ks_2samp(wheremost10.fracwhite10, wheremost19.fracwhite19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, np.round(ks_1019.pvalue,2), pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Aesthetics
    ax1.set_xlim([-0.75, 23.75])
    ax1.set_xticks([1,5,9,14,18,22])
    ax1.set_xticklabels([
        'PM$_\mathregular{2.5}$-\nattributable',
        'NO$_\mathregular{2}$-\nattributable',
        'NO$_\mathregular{2}$- and PM$_{\mathregular{2.5}}$-\nattributable',
        'PM$_\mathregular{2.5}$-\nattributable',
        'NO$_\mathregular{2}$-\nattributable',    
        'NO$_\mathregular{2}$- and PM$_{\mathregular{2.5}}$-\nattributable'])
    ax1.tick_params(axis=u'x', which=u'both',length=0)
    ax1.set_ylim([0,1])
    ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.])
    ax1.set_yticklabels(['0','20','40','60','80','100'])
    ax1.set_ylabel('Proportion of white population [%]', fontsize=14)
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Create legend
    ax1.text(5, 1.05,'L e a s t    b u r d e n e d', fontsize=18, ha='center')
    ax1.text(18, 1.05,'M o s t    b u r d e n e d', fontsize=18, ha='center')
    patch1 = mpatches.Patch(color=colorsyear[0], label='2010')
    patch2 = mpatches.Patch(color=colorsyear[1], label='2015')
    patch3 = mpatches.Patch(color=colorsyear[2], label='2019')
    all_handles = (patch1, patch2, patch3)
    leg = ax1.legend(handles=all_handles, frameon=False, ncol=1,
        bbox_to_anchor=(0.2,0.25))
    plt.savefig(DIR_FIG+'fig3.pdf', dpi=500)
    # Calculate some pithy statistics
    print('Median white population [%]')
    print('2010 = %.2f'%(burden10.fracwhite10.median()*100.))
    print('2015 = %.2f'%(burden15.fracwhite15.median()*100.))
    print('2019 = %.2f'%(burden19.fracwhite19.median()*100.))
    print('\n')
    print('White population in tracts with smallest '+\
        'PM2.5-attributable burdens [%]')
    print('2010 = %.2f'%(whereleastpm10.fracwhite10.median()*100.))
    print('2015 = %.2f'%(whereleastpm15.fracwhite15.median()*100.))
    print('2019 = %.2f'%(whereleastpm19.fracwhite19.median()*100.)) 
    print('White population in tracts with largest '+\
        'PM2.5-attributable burdens [%]')
    print('2010 = %.2f'%(wheremostpm10.fracwhite10.median()*100.))
    print('2015 = %.2f'%(wheremostpm15.fracwhite15.median()*100.))
    print('2019 = %.2f'%(wheremostpm19.fracwhite19.median()*100.))    
    print('\n')
    print('White population in tracts with smallest '+\
        'NO2-attributable burdens [%]')
    print('2010 = %.2f'%(whereleastasthma10.fracwhite10.median()*100.))
    print('2015 = %.2f'%(whereleastasthma15.fracwhite15.median()*100.))
    print('2019 = %.2f'%(whereleastasthma19.fracwhite19.median()*100.))
    print('White population in tracts with largest '+\
        'NO2-attributable burdens [%]')
    print('2010 = %.2f'%(wheremostasthma10.fracwhite10.median()*100.))
    print('2015 = %.2f'%(wheremostasthma15.fracwhite15.median()*100.))
    print('2019 = %.2f'%(wheremostasthma19.fracwhite19.median()*100.))
    print('\n')
    print('White population in tracts with smallest '+\
        'PM2.5- AND NO2-attributable burdens [%]')
    print('2010 = %.2f'%(whereleast10.fracwhite10.median()*100.), 
        '(%d tracts)'%whereleast10.shape[0])
    print('2015 = %.2f'%(whereleast15.fracwhite15.median()*100.), 
        '(%d tracts)'%whereleast15.shape[0])
    print('2019 = %.2f'%(whereleast19.fracwhite19.median()*100.), 
        '(%d tracts)'%whereleast19.shape[0]) 
    print('White population in tracts with largest '+\
        'PM2.5- AND NO2-attributable burdens [%]')
    print('2010 = %.2f'%(wheremost10.fracwhite10.median()*100.), 
        '(%d tracts)'%wheremost10.shape[0])
    print('2015 = %.2f'%(wheremost15.fracwhite15.median()*100.), 
        '(%d tracts)'%wheremost15.shape[0])
    print('2019 = %.2f'%(wheremost19.fracwhite19.median()*100.), 
        '(%d tracts)'%wheremost19.shape[0])
    return

def table1(burdents):
    """

    Parameters
    ----------
    burdents : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import numpy as np
    from prettytable import PrettyTable
    
    # # # # For NO2
    burden19 = burdents.loc[burdents['YEAR']=='2015-2019'].copy(deep=True)
    
    # For mean NO2 and total pediatric asthma
    x1 = PrettyTable()
    x1.field_names = ["Scenario", "Mean NO2", "Total pediatric asthma", 
        "Percentage change pediatric asthma"]
    x1.add_rows(
        [
        ["Control", 
         np.round(burden19.NO2.mean(), 2), 
         np.round(burden19.BURDENASTHMA.sum(), 2), 
         np.round(((burden19.BURDENASTHMA.sum()-burden19.BURDENASTHMA.sum())/
          burden19.BURDENASTHMA.sum())*100., 2)],
        ["WHO IT-1", 
         np.round(burden19.NO2WHO40.mean(), 2), 
         np.round(burden19.BURDENASTHMAWHO40.sum(), 2),
         np.round(((burden19.BURDENASTHMAWHO40.sum()-
          burden19.BURDENASTHMA.sum())/burden19.BURDENASTHMA.sum())*100., 2)],     
        ["WHO IT-2", 
         np.round(burden19.NO2WHO30.mean(), 2), 
         np.round(burden19.BURDENASTHMAWHO30.sum(), 2),
         np.round(((burden19.BURDENASTHMAWHO30.sum()-
         burden19.BURDENASTHMA.sum())/burden19.BURDENASTHMA.sum())*100., 2)],
        ["WHO IT-3", 
         np.round(burden19.NO2WHO20.mean(), 2), 
         np.round(burden19.BURDENASTHMAWHO20.sum(), 2),
         np.round(((burden19.BURDENASTHMAWHO20.sum()-
         burden19.BURDENASTHMA.sum())/burden19.BURDENASTHMA.sum())*100., 2)],
        ["WHO AQG", 
         np.round(burden19.NO2WHO10.mean(), 2), 
         np.round(burden19.BURDENASTHMAWHO10.sum(), 2),
         np.round(((burden19.BURDENASTHMAWHO10.sum()-
         burden19.BURDENASTHMA.sum())/burden19.BURDENASTHMA.sum())*100., 2)] 
        ]
    )
    print(x1.get_string(title='Mean NO2 and total pediatric asthma incidence'))
    
    # NO2 disparities for most/least white population subgroups
    fracwhite = ((burden19[['race_nh_white','race_h_white']].sum(axis=1))/
        burden19['race_tot'])
    burden19['fracwhite'] = fracwhite
    mostwhite = burden19.iloc[np.where(burden19.fracwhite >= 
        np.nanpercentile(burden19.fracwhite, 90))]
    leastwhite = burden19.iloc[np.where(burden19.fracwhite <= 
        np.nanpercentile(burden19.fracwhite, 10))]
    x2 = PrettyTable()
    x2.field_names = ["Scenario", "Most white", "Percentage change most white",
        "Least white", "Percentage change least white"]
    x2.add_rows(
        [
        ["Control", 
         np.round(mostwhite.BURDENASTHMARATE.mean(), 2), 
         np.round(((mostwhite.BURDENASTHMARATE.mean()-
         mostwhite.BURDENASTHMARATE.mean())/
         mostwhite.BURDENASTHMARATE.mean())*100., 2),
         np.round(leastwhite.BURDENASTHMARATE.mean(), 2), 
         np.round(((leastwhite.BURDENASTHMARATE.mean()-
         leastwhite.BURDENASTHMARATE.mean())/
         leastwhite.BURDENASTHMARATE.mean())*100., 2)],
        ["WHO IT-1", 
         np.round(mostwhite.BURDENASTHMARATEWHO40.mean(), 2), 
         np.round(((mostwhite.BURDENASTHMARATEWHO40.mean()-
         mostwhite.BURDENASTHMARATE.mean())/
          mostwhite.BURDENASTHMARATE.mean())*100., 2),
         np.round(leastwhite.BURDENASTHMARATEWHO40.mean(), 2), 
         np.round(((leastwhite.BURDENASTHMARATEWHO40.mean()-
         leastwhite.BURDENASTHMARATE.mean())/
         leastwhite.BURDENASTHMARATE.mean())*100., 2)],
        ["WHO IT-2", 
         np.round(mostwhite.BURDENASTHMARATEWHO30.mean(), 2), 
         np.round(((mostwhite.BURDENASTHMARATEWHO30.mean()-
         mostwhite.BURDENASTHMARATE.mean())/
         mostwhite.BURDENASTHMARATE.mean())*100., 2),
         np.round(leastwhite.BURDENASTHMARATEWHO30.mean(), 2), 
         np.round(((leastwhite.BURDENASTHMARATEWHO30.mean()-
         leastwhite.BURDENASTHMARATE.mean())/
         leastwhite.BURDENASTHMARATE.mean())*100., 2)],
        ["WHO IT-3", 
         np.round(mostwhite.BURDENASTHMARATEWHO20.mean(), 2), 
         np.round(((mostwhite.BURDENASTHMARATEWHO20.mean()-
         mostwhite.BURDENASTHMARATE.mean())/
         mostwhite.BURDENASTHMARATE.mean())*100., 2),
         np.round(leastwhite.BURDENASTHMARATEWHO20.mean(), 2), 
         np.round(((leastwhite.BURDENASTHMARATEWHO20.mean()-
         leastwhite.BURDENASTHMARATE.mean())/
         leastwhite.BURDENASTHMARATE.mean())*100., 2)],
        ["WHO AQG", 
         np.round(mostwhite.BURDENASTHMARATEWHO10.mean(), 2), 
         np.round(((mostwhite.BURDENASTHMARATEWHO10.mean()-
         mostwhite.BURDENASTHMARATE.mean())/
         mostwhite.BURDENASTHMARATE.mean())*100., 2),
         np.round(leastwhite.BURDENASTHMARATEWHO10.mean(), 2), 
         np.round(((leastwhite.BURDENASTHMARATEWHO10.mean()-
         leastwhite.BURDENASTHMARATE.mean())/
         leastwhite.BURDENASTHMARATE.mean())*100., 2)]     
        ]
    )
    print(x2.get_string(title='Disparities in NO2-attributable asthma'+\
        ' rates per 100K (subgroup extremes method)'))
        
    # NO2 disparities for white and black population-weighted subgroups
    asthmablack = ((burden19['BURDENASTHMARATE']*burden19[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden19[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    asthmablackwho40 = ((burden19['BURDENASTHMARATEWHO40']*burden19[
        ['race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())
    asthmablackwho30 = ((burden19['BURDENASTHMARATEWHO30']*burden19[
        ['race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())                                  
    asthmablackwho20 = ((burden19['BURDENASTHMARATEWHO20']*burden19[
        ['race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())                                                  
    asthmablackwho10 = ((burden19['BURDENASTHMARATEWHO10']*burden19[
        ['race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())                                                  
    asthmawhite = ((burden19['BURDENASTHMARATE']*
        burden19[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
        burden19[['race_nh_white','race_h_white']].sum(axis=1).sum())                
    asthmawhitewho40 = ((burden19['BURDENASTHMARATEWHO40']*
        burden19[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
        burden19[['race_nh_white','race_h_white']].sum(axis=1).sum())                
    asthmawhitewho30 = ((burden19['BURDENASTHMARATEWHO30']*
        burden19[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
        burden19[['race_nh_white','race_h_white']].sum(axis=1).sum())                
    asthmawhitewho20 = ((burden19['BURDENASTHMARATEWHO20']*
        burden19[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
        burden19[['race_nh_white','race_h_white']].sum(axis=1).sum())                
    asthmawhitewho10 = ((burden19['BURDENASTHMARATEWHO10']*
        burden19[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
        burden19[['race_nh_white','race_h_white']].sum(axis=1).sum())                
    x3 = PrettyTable()
    x3.field_names = ["Scenario", "White pop.-wtg.", 
        "Percentage change white pop.-wtg.",
        "Black pop.wtg.", "Percentage change black pop.-wtg."]
    x3.add_rows(
        [
        ["Control", 
         np.round(asthmawhite, 2), 
         np.round(((asthmawhite-asthmawhite)/asthmawhite)*100., 2),
         np.round(asthmablack, 2),
         np.round(((asthmablack-asthmablack)/asthmablack)*100., 2)],
        ["WHO IT-1", 
         np.round(asthmawhitewho40, 2), 
         np.round(((asthmawhitewho40-asthmawhite)/asthmawhite)*100., 2),
         np.round(asthmablackwho40, 2),
         np.round(((asthmablackwho40-asthmablack)/asthmablack)*100.,2)],
         ["WHO IT-2", 
         np.round(asthmawhitewho30, 2), 
         np.round(((asthmawhitewho30-asthmawhite)/asthmawhite)*100., 2),
         np.round(asthmablackwho30, 2),
         np.round(((asthmablackwho30-asthmablack)/asthmablack)*100., 2)],
         ["WHO IT-3", 
         np.round(asthmawhitewho20, 2), 
         np.round(((asthmawhitewho20-asthmawhite)/asthmawhite)*100., 2),
         np.round(asthmablackwho20, 2),
         np.round(((asthmablackwho20-asthmablack)/asthmablack)*100., 2)],
         ["WHO AQG", 
         np.round(asthmawhitewho10, 2), 
         np.round(((asthmawhitewho10-asthmawhite)/asthmawhite)*100., 2),
         np.round(asthmablackwho10, 2),
         np.round(((asthmablackwho10-asthmablack)/asthmablack)*100., 2)]
        ]                                                  
    )
    print(x3.get_string(title='Disparities in NO2-attributable asthma'+\
        ' rates per 100K (pop.-wtg. method)'))   
                                                      
    # # # # For PM2.5    
    # For mean PM2.5 and total premature mortality
    x4 = PrettyTable()
    x4.field_names = ["Scenario", "Mean PM25", "Total premature mortality", 
        "Percentage change premature mortality"]
    x4.add_rows(
        [
        ["Control", 
         np.round(burden19.PM25.mean(), 2), 
         np.round(burden19.BURDENPMALL.sum(), 2), 
         np.round(((burden19.BURDENPMALL.sum()-burden19.BURDENPMALL.sum())/
          burden19.BURDENPMALL.sum())*100., 2)],
        ["WHO IT-3", 
         np.round(burden19.PM25WHO15.mean(), 2), 
         np.round(burden19.BURDENPMALLWHO15.sum(), 2),
         np.round(((burden19.BURDENPMALLWHO15.sum()-
         burden19.BURDENPMALL.sum())/burden19.BURDENPMALL.sum())*100., 2)],     
        ["EPA NAAQS", 
         np.round(burden19.PM25NAAQS12.mean(), 2), 
         np.round(burden19.BURDENPMALLNAAQS12.sum(), 2),
         np.round(((burden19.BURDENPMALLNAAQS12.sum()-
         burden19.BURDENPMALL.sum())/burden19.BURDENPMALL.sum())*100., 2)],
        ["WHO IT-4", 
         np.round(burden19.PM25WHO10.mean(), 2), 
         np.round(burden19.BURDENPMALLWHO10.sum(), 2),
         np.round(((burden19.BURDENPMALLWHO10.sum()-
         burden19.BURDENPMALL.sum())/burden19.BURDENPMALL.sum())*100., 2)],
        ["WHO AQG", 
         np.round(burden19.PM25WHO5.mean(), 2), 
         np.round(burden19.BURDENPMALLWHO5.sum(), 2),
         np.round(((burden19.BURDENPMALLWHO5.sum()-
         burden19.BURDENPMALL.sum())/burden19.BURDENPMALL.sum())*100., 2)] 
        ]
    )
    print(x4.get_string(title='Mean PM2.5 and total premature mortality'))
    
    # PM2.5 disparities for most/least white population subgroups
    fracwhite = ((burden19[['race_nh_white','race_h_white']].sum(axis=1))/
        burden19['race_tot'])
    burden19['fracwhite'] = fracwhite
    mostwhite = burden19.iloc[np.where(burden19.fracwhite >= 
        np.nanpercentile(burden19.fracwhite, 90))]
    leastwhite = burden19.iloc[np.where(burden19.fracwhite <= 
        np.nanpercentile(burden19.fracwhite, 10))]
    x5 = PrettyTable()
    x5.field_names = ["Scenario", "Most white", "Percentage change most white",
        "Least white", "Percentage change least white"]
    x5.add_rows(
        [
        ["Control", 
         np.round(mostwhite.BURDENPMALLRATE.mean(), 2), 
         np.round(((mostwhite.BURDENPMALLRATE.mean()-
         mostwhite.BURDENPMALLRATE.mean())/
         mostwhite.BURDENPMALLRATE.mean())*100., 2),
         np.round(leastwhite.BURDENPMALLRATE.mean(), 2), 
         np.round(((leastwhite.BURDENPMALLRATE.mean()-
         leastwhite.BURDENPMALLRATE.mean())/
         leastwhite.BURDENPMALLRATE.mean())*100., 2)],
        ["WHO IT-3", 
         np.round(mostwhite.BURDENPMALLRATEWHO15.mean(), 2), 
         np.round(((mostwhite.BURDENPMALLRATEWHO15.mean()-
         mostwhite.BURDENPMALLRATE.mean())/
         mostwhite.BURDENPMALLRATE.mean())*100., 2),
         np.round(leastwhite.BURDENPMALLRATEWHO15.mean(), 2), 
         (np.round(((leastwhite.BURDENPMALLRATEWHO15.mean()-
         leastwhite.BURDENPMALLRATE.mean())/
          leastwhite.BURDENPMALLRATE.mean())*100., 2))],
        ["EPA NAAQS", 
         np.round(mostwhite.BURDENPMALLRATENAAQS12.mean(), 2), 
         np.round(((mostwhite.BURDENPMALLRATENAAQS12.mean()-
         mostwhite.BURDENPMALLRATE.mean())/
         mostwhite.BURDENPMALLRATE.mean())*100., 2),
         np.round(leastwhite.BURDENPMALLRATENAAQS12.mean(), 2), 
         np.round(((leastwhite.BURDENPMALLRATENAAQS12.mean()-
         leastwhite.BURDENPMALLRATE.mean())/
         leastwhite.BURDENPMALLRATE.mean())*100., 2)],
        ["WHO IT-4", 
         np.round(mostwhite.BURDENPMALLRATEWHO10.mean(), 2), 
         np.round(((mostwhite.BURDENPMALLRATEWHO10.mean()-
         mostwhite.BURDENPMALLRATE.mean())/
         mostwhite.BURDENPMALLRATE.mean())*100., 2),
         np.round(leastwhite.BURDENPMALLRATEWHO10.mean(), 2), 
         np.round(((leastwhite.BURDENPMALLRATEWHO10.mean()-
         leastwhite.BURDENPMALLRATE.mean())/
         leastwhite.BURDENPMALLRATE.mean())*100., 2)],
        ["WHO AQG", 
         np.round(mostwhite.BURDENPMALLRATEWHO5.mean(), 2), 
         np.round(((mostwhite.BURDENPMALLRATEWHO5.mean()-
         mostwhite.BURDENPMALLRATE.mean())/
         mostwhite.BURDENPMALLRATE.mean())*100., 2),
         np.round(leastwhite.BURDENPMALLRATEWHO5.mean(), 2), 
         np.round(((leastwhite.BURDENPMALLRATEWHO5.mean()-
         leastwhite.BURDENPMALLRATE.mean())/
         leastwhite.BURDENPMALLRATE.mean())*100., 2)]  
        ]
    )
    print(x5.get_string(title='Disparities in PM2.5-attributable mortality'+\
        ' rates per 100K (subgroup extremes method)'))
    
    # PM25 disparities for white and black population-weighted subgroups
    allmortblack = ((burden19['BURDENPMALLRATE']*burden19[[
        'race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum()) 
    allmortblackwho15 = ((burden19['BURDENPMALLRATEWHO15']*burden19[[
        'race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum()) 
    allmortblacknaaqs12 = ((burden19['BURDENPMALLRATENAAQS12']*burden19[[
        'race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())         
    allmortblackwho10 = ((burden19['BURDENPMALLRATEWHO10']*burden19[[
        'race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())         
    allmortblackwho5 = ((burden19['BURDENPMALLRATEWHO5']*burden19[[
        'race_nh_black','race_h_black']].sum(axis=1)).sum()/burden19[[
        'race_nh_black','race_h_black']].sum(axis=1).sum())                 
    allmortwhite = ((burden19['BURDENPMALLRATE']*burden19[
        ['race_nh_white','race_h_white']].sum(axis=1)).sum()/burden19[
        ['race_nh_white','race_h_white']].sum(axis=1).sum()) 
    allmortwhitewho15 = ((burden19['BURDENPMALLRATEWHO15']*burden19[
        ['race_nh_white','race_h_white']].sum(axis=1)).sum()/burden19[
        ['race_nh_white','race_h_white']].sum(axis=1).sum()) 
    allmortwhitenaaqs12 = ((burden19['BURDENPMALLRATENAAQS12']*burden19[
        ['race_nh_white','race_h_white']].sum(axis=1)).sum()/burden19[
        ['race_nh_white','race_h_white']].sum(axis=1).sum()) 
    allmortwhitewho10 = ((burden19['BURDENPMALLRATEWHO10']*burden19[
        ['race_nh_white','race_h_white']].sum(axis=1)).sum()/burden19[
        ['race_nh_white','race_h_white']].sum(axis=1).sum()) 
    allmortwhitewho5 = ((burden19['BURDENPMALLRATEWHO5']*burden19[
        ['race_nh_white','race_h_white']].sum(axis=1)).sum()/burden19[
        ['race_nh_white','race_h_white']].sum(axis=1).sum())    
    x6 = PrettyTable()
    x6.field_names = ["Scenario", "White pop.-wtg.", 
        "Percentage change white pop.-wtg.",
        "Black pop.wtg.", "Percentage change black pop.-wtg."]
    x6.add_rows(
        [
        ["Control", 
         np.round(allmortwhite, 2), 
         np.round(((allmortwhite-allmortwhite)/allmortwhite)*100., 2),
         np.round(allmortblack, 2),
         np.round(((allmortblack-allmortblack)/allmortblack)*100., 2)],
        ["WHO IT-3", 
         np.round(allmortwhitewho15, 2), 
         np.round(((allmortwhitewho15-allmortwhite)/allmortwhite)*100., 2),
         np.round(allmortblackwho15, 2),
         np.round(((allmortblackwho15-allmortblack)/allmortblack)*100.,2)],
         ["EPA NAAQS", 
         np.round(allmortwhitenaaqs12, 2), 
         np.round(((allmortwhitenaaqs12-allmortwhite)/allmortwhite)*100., 2),
         np.round(allmortblacknaaqs12, 2),
         np.round(((allmortblacknaaqs12-allmortblack)/allmortblack)*100., 2)],
         ["WHO IT-4", 
         np.round(allmortwhitewho10, 2), 
         np.round(((allmortwhitewho10-allmortwhite)/allmortwhite)*100., 2),
         np.round(allmortblackwho10, 2),
         np.round(((allmortblackwho10-allmortblack)/allmortblack)*100., 2)],
         ["WHO AQG", 
         np.round(allmortwhitewho5, 2), 
         np.round(((allmortwhitewho5-allmortwhite)/allmortwhite)*100., 2),
         np.round(allmortblackwho5, 2),
         np.round(((allmortblackwho5-allmortblack)/allmortblack)*100., 2)]
        ]                                                  
    )
    print(x6.get_string(title='Disparities in PM25-attributable mortality'+\
        ' rates per 100K (pop.-wtg. method)'))   
    with open(DIR+'docs/IT_AQG_benefits.txt','w') as file:
        file.write(x1.get_string())
        file.write('\n')
        file.write(x2.get_string())
        file.write('\n')    
        file.write(x3.get_string())    
        file.write('\n')    
        file.write(x4.get_string())    
        file.write('\n')    
        file.write(x5.get_string())    
        file.write('\n')    
        file.write(x6.get_string())
    return

def figS1(burdents):
    """Plot maps of tract-averaged NO2 and PM2.5 for the contiguous U.S., 
    Hawaii, Alaska, and Puerto Rico for the first year of the measuring period,
    a midpoint year, and the final year (2010, 2015, 2019). 

    Parameters
    ----------
    burdents : pandas.core.frame.DataFrame
        Tract-level NO2 and PM2.5 pollution concentrations and attributable
        health burdens for all years in measuring period. 

    Returns
    -------
    None.

    """
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely.geometry import Polygon
    import matplotlib.pyplot as plt
    from cartopy.io import shapereader
    import matplotlib
    from operator import itemgetter
    # Subset harmonized dataset for years of interest
    harm2010 = burdents.loc[(burdents.YEAR=='2006-2010')]
    harm2015 = burdents.loc[(burdents.YEAR=='2011-2015')]
    harm2019 = burdents.loc[(burdents.YEAR=='2015-2019')]
    # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shapereader.Reader(shpfilename)
    countries = reader.records()   
    usa = [x.attributes['ADM0_A3'] for x in countries]
    # usaidx = np.where(np.in1d(np.array(usaidx), ['PRI','USA'])==True)
    # usa = list(reader.geometries())
    # usa = np.array(usa, dtype=object)[usaidx[0]]
    usa = np.where(np.array(usa)=='USA')[0][0]
    usa = list(reader.geometries())[usa].geoms
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shapereader.Reader(lakes)
    lakes = lakes_reader.records()   
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names)=='Lake Superior') |
        (np.array(lake_names)=='Lake Michigan') | 
        (np.array(lake_names)=='Lake Huron') |
        (np.array(lake_names)=='Lake Erie') |
        (np.array(lake_names)=='Lake Ontario'))[0]
    great_lakes = itemgetter(*great_lakes)(list(lakes_reader.geometries()))
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shapereader.natural_earth(resolution='10m', category='cultural', 
        name=shapename)
    states_shp = shapereader.Reader(states_shp)
    # Constants
    proj = ccrs.PlateCarree(central_longitude=0.0)
    # Initialize figure, subplots
    fig = plt.figure(figsize=(12,4.25))
    ax1 = plt.subplot2grid((2,3),(0,0), projection=proj)
    ax2 = plt.subplot2grid((2,3),(0,1), projection=proj)
    ax3 = plt.subplot2grid((2,3),(0,2), projection=proj)
    ax4 = plt.subplot2grid((2,3),(1,0), projection=proj)
    ax5 = plt.subplot2grid((2,3),(1,1), projection=proj)
    ax6 = plt.subplot2grid((2,3),(1,2), projection=proj)
    # Subplot titles
    ax1.set_title('(A) 2010 NO$_{\mathregular{2}}$', loc='left')
    ax2.set_title('(B) 2015 NO$_{\mathregular{2}}$', loc='left')
    ax3.set_title('(C) 2019 NO$_{\mathregular{2}}$', loc='left')
    ax4.set_title('(D) 2010 PM$_{\mathregular{2.5}}$', loc='left')
    ax5.set_title('(E) 2015 PM$_{\mathregular{2.5}}$', loc='left')
    ax6.set_title('(F) 2019 PM$_{\mathregular{2.5}}$', loc='left')
    # Create discrete colormaps
    cmappm = plt.get_cmap('magma_r', 8)
    normpm = matplotlib.colors.Normalize(vmin=0, vmax=12)
    cmapno2 = plt.get_cmap('magma_r', 8)
    normno2 = matplotlib.colors.Normalize(vmin=0, vmax=24)
    # Adjust subplot position 
    plt.subplots_adjust(left=0.05)
    # Plotting 
    FIPS = ['01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', 
        '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
        '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', 
        '39', '40', '41', '42', '44', '45', '46', '47', '48', '49', '50',
        '51', '53', '54', '55', '56']
    # FIPS = ['04']
    for FIPS_i in FIPS: 
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+
            'tract_2010/tl_2019_%s_tract/tl_2019_%s_tract.shp'%(FIPS_i, FIPS_i))
        records = shp.records()
        tracts = shp.geometries()
        for record, tract in zip(records, tracts):
            # Find GEOID of tract
            gi = record.attributes['GEOID']
            # Look up harmonized NO2 and PM25 data for tract
            harm2010_tract = harm2010.loc[harm2010.index.isin([gi])]
            harm2015_tract = harm2015.loc[harm2015.index.isin([gi])]
            harm2019_tract = harm2019.loc[harm2019.index.isin([gi])]     
            # Plot NO2 and PM25
            if harm2010_tract.empty==True: 
                ax1.add_geometries([tract], proj, facecolor='#f2f2f2', 
                    edgecolor='none', alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
                ax4.add_geometries([tract], proj, facecolor='#f2f2f2', 
                    edgecolor='none', alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
            else: 
                no2_harm2010_tract = harm2010_tract.NO2.values[0]
                pm25_harm2010_tract = harm2010_tract.PM25.values[0]
                ax1.add_geometries([tract], proj, facecolor=cmapno2(
                    normno2(no2_harm2010_tract)), edgecolor=cmapno2(
                    normno2(no2_harm2010_tract)), alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
                ax4.add_geometries([tract], proj, facecolor=cmappm(
                    normpm(pm25_harm2010_tract)), edgecolor=cmappm(
                    normpm(pm25_harm2010_tract)), alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
            if harm2015_tract.empty==True:
                ax2.add_geometries([tract], proj, facecolor='#f2f2f2', 
                    edgecolor='none', alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
                ax5.add_geometries([tract], proj, facecolor='#f2f2f2', 
                    edgecolor='none', alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)                
            else:             
                no2_harm2015_tract = harm2015_tract.NO2.values[0]
                pm25_harm2015_tract = harm2015_tract.PM25.values[0]
                ax2.add_geometries([tract], proj, facecolor=cmapno2(
                    normno2(no2_harm2015_tract)), edgecolor=cmapno2(
                    normno2(no2_harm2015_tract)), alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
                ax5.add_geometries([tract], proj, facecolor=cmappm(
                    normpm(pm25_harm2015_tract)), edgecolor=cmappm(
                    normpm(pm25_harm2015_tract)), alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
            if harm2019_tract.empty==True:             
                ax3.add_geometries([tract], proj, facecolor='#f2f2f2', 
                    edgecolor='none', alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
                ax6.add_geometries([tract], proj, facecolor='#f2f2f2', 
                    edgecolor='none', alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)                
            else: 
                no2_harm2019_tract = harm2019_tract.NO2.values[0]  
                pm25_harm2019_tract = harm2019_tract.PM25.values[0]                  
                ax3.add_geometries([tract], proj, facecolor=cmapno2(
                    normno2(no2_harm2019_tract)), edgecolor=cmapno2(
                    normno2(no2_harm2019_tract)), alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)
                ax6.add_geometries([tract], proj, facecolor=cmappm(
                    normpm(pm25_harm2019_tract)), edgecolor=cmappm(
                    normpm(pm25_harm2019_tract)), alpha=1., linewidth=0.1, 
                    rasterized=True, zorder=10)                    
    # Add borders, set map extent, etc. 
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)
        ax.add_geometries(usa, crs=proj, lw=0.25, facecolor='None', 
            edgecolor='k', zorder=1)
        ax.add_geometries(great_lakes, crs=proj, facecolor='w',
            lw=0.25, edgecolor='k', alpha=1., zorder=12)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 
            'ocean', '10m', edgecolor='None', facecolor='w', alpha=1.), 
            zorder=11)
        ax.axis('off')
        # Add state borders
        for astate in states_shp.records():
            if astate.attributes['name'] in ['Alaska', 'Hawaii', 
                'PRI-00 (Puerto Rico aggregation)']:
                pass
            elif astate.attributes['sr_adm0_a3']=='USA':
                geometry = astate.geometry.geoms
                ax.add_geometries([geometry], crs=proj, facecolor='None',
                    lw=0.25, edgecolor='k', zorder=100)    
    # Find shapefiles of states of interest
    states_all = list(states_shp.records())
    states_all_name = []
    for s in states_all:
        states_all_name.append(s.attributes['name'])
    states_all = np.array(states_all)    
    alaska = states_all[np.where(np.array(states_all_name)=='Alaska')[0]][0]
    hawaii = states_all[np.where(np.array(states_all_name)=='Hawaii')[0]][0]
    puertorico = states_all[np.where(np.array(states_all_name)==
        'PRI-00 (Puerto Rico aggregation)')[0]][0]
    # Select harmonized dataset in states in inset maps
    alaska2010 = burdents.loc[(burdents.YEAR=='2006-2010') & 
        (burdents.STATE=='Alaska')]
    alaska2014 = burdents.loc[(burdents.YEAR=='2010-2014') & 
        (burdents.STATE=='Alaska')]
    alaska2015 = burdents.loc[(burdents.YEAR=='2011-2015') & 
        (burdents.STATE=='Alaska')]
    alaska2017 = burdents.loc[(burdents.YEAR=='2013-2017') & 
        (burdents.STATE=='Alaska')]
    alaska2019 = burdents.loc[(burdents.YEAR=='2015-2019') & 
        (burdents.STATE=='Alaska')]
    hawaii2010 = burdents.loc[(burdents.YEAR=='2006-2010') & 
        (burdents.STATE=='Hawaii')]
    hawaii2014 = burdents.loc[(burdents.YEAR=='2010-2014') & 
        (burdents.STATE=='Hawaii')]
    hawaii2015 = burdents.loc[(burdents.YEAR=='2011-2015') & 
        (burdents.STATE=='Hawaii')]
    hawaii2017 = burdents.loc[(burdents.YEAR=='2013-2017') & 
        (burdents.STATE=='Hawaii')]
    hawaii2019 = burdents.loc[(burdents.YEAR=='2015-2019') & 
        (burdents.STATE=='Hawaii')]
    puertorico2010 = burdents.loc[(burdents.YEAR=='2006-2010') & 
        (burdents.STATE=='Puerto Rico')]
    puertorico2014 = burdents.loc[(burdents.YEAR=='2010-2014') & 
        (burdents.STATE=='Puerto Rico')]
    puertorico2015 = burdents.loc[(burdents.YEAR=='2011-2015') & 
        (burdents.STATE=='Puerto Rico')]
    puertorico2017 = burdents.loc[(burdents.YEAR=='2013-2017') & 
        (burdents.STATE=='Puerto Rico')]
    puertorico2019 = burdents.loc[(burdents.YEAR=='2015-2019') & 
        (burdents.STATE=='Puerto Rico')]
    # Add inset maps 
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    harms = [[alaska2010, hawaii2010, puertorico2010],
        [alaska2015, hawaii2015, puertorico2015],
        [alaska2019, hawaii2019, puertorico2019],
        [alaska2010, hawaii2010, puertorico2010],
        [alaska2014, hawaii2014, puertorico2014],
        [alaska2017, hawaii2017, puertorico2017]]
    cmaps = [cmapno2, cmapno2, cmapno2, cmappm, cmappm, cmappm]
    norms = [normno2, normno2, normno2, normpm, normpm, normpm]
    varsa = ['NO2', 'NO2', 'NO2', 'PM25', 'PM25', 'PM25']
    for i in np.arange(0, len(axes), 1):
        ax = axes[i]
        harm = harms[i]
        cmap = cmaps[i]
        norm = norms[i]
        vara = varsa[i]
        # Hawaii 
        axes_extent = (ax.get_position().x0-0.01, ax.get_position().y0-0.01, 
            (ax.get_position().x1-ax.get_position().x0)*0.31,
            (ax.get_position().x1-ax.get_position().x0)*0.31)
        add_insetmap(axes_extent, (-162, -154, 18.75, 23), '', hawaii.geometry, 
            [0.], [0.], [0.], proj, fips='15', harmonized=harm[1], vara=vara, 
            cmap=cmap, norm=norm)
        # Alaska
        axes_extent = (ax.get_position().x0-0.085, ax.get_position().y0-0.04, 
            (ax.get_position().x1-ax.get_position().x0)*0.62,
            (ax.get_position().x1-ax.get_position().x0)*0.62)
        add_insetmap(axes_extent, (-179.99, -130, 49, 73), '', alaska.geometry, 
            [0.], [0.], [0.], proj, fips='02', harmonized=harm[0], vara=vara, 
            cmap=cmap, norm=norm)    
        # Puerto Rico 
        axes_extent = (ax.get_position().x0+0.045, ax.get_position().y0-0.02, 
            (ax.get_position().x1-ax.get_position().x0)*0.27,
            (ax.get_position().x1-ax.get_position().x0)*0.27)        
        add_insetmap(axes_extent, (-68., -65., 17.5, 19.), '', puertorico.geometry, 
            [0.], [0.], [0.], proj, fips='72', harmonized=harm[2], vara=vara, 
            cmap=cmap, norm=norm)
    # Add colorbars 
    caxno2 = fig.add_axes([ax3.get_position().x1+0.01, 
        ax3.get_position().y0, 0.01,
        (ax3.get_position().y1-ax3.get_position().y0)])
    mpl.colorbar.ColorbarBase(caxno2, cmap=cmapno2, norm=normno2, 
        spacing='proportional', orientation='vertical', extend='max', 
        label='[ppbv]', ticks=np.linspace(0,24,5))
    caxpm = fig.add_axes([ax6.get_position().x1+0.01, 
        ax6.get_position().y0, 0.01,
        (ax6.get_position().y1-ax6.get_position().y0)])
    mpl.colorbar.ColorbarBase(caxpm, cmap=cmappm, norm=normpm, 
        spacing='proportional', orientation='vertical', extend='max', 
        label='[$\mathregular{\mu}$g m$^{\mathregular{-3}}$]', 
        ticks=np.linspace(0,12,5))
    plt.savefig(DIR_FIG+'figS1.png', dpi=1000)
    return 
    
def figS2(): 
    """

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    """
    import numpy as np
    import pandas as pd
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.io import shapereader
    import cartopy.feature as cfeature
    from operator import itemgetter
    
    def find_eparegion(df):
        """Find EPA Region of each AQS monitor. 
    
        Parameters
        ----------
        df : pandas.core.frame.DataFrame
            Harmonized AQS PM2.5 or NO2 observations with tract-averaged 
            concentrations in tracts colocated with monitors.
    
        Returns
        -------
        df : TYPE
            Harmonized AQS PM2.5 or NO2 observations with tract-averaged 
            concentrations in tracts colocated with monitors and the corresponding
            EPA region
        """
        import numpy as np
        df['EPA_REGION'] = np.nan
        # EPA Region 1 - Boston (serving CT, ME, MA, NH, RI, and VT)
        df.loc[df['State Name'].isin(['Vermont', 'Rhode Island', 'New Hampshire', 
            'Massachusetts', 'Maine', 'Connecticut']), 'EPA_REGION'] = 1
        # Region 2 – New York City (serving NJ, NY, Puerto Rico, and the U.S. 
        # Virgin Islands)
        df.loc[df['State Name'].isin(['New Jersey', 'New York', 'Puerto Rico']), 
            'EPA_REGION'] = 2
        # Region 3 – Philadelphia (serving DE, DC, MD, PA, VA, WV and 7 federally
        # recognized tribes)
        df.loc[df['State Name'].isin(['Delaware', 'District Of Columbia', 
            'Maryland', 'Pennsylvania', 'Virginia', 'West Virginia']), 
            'EPA_REGION'] = 3
        # Region 4 – Atlanta (serving AL, FL, GA, KY, MS, NC, SC, and TN)
        df.loc[df['State Name'].isin(['Alabama', 'Florida', 'Georgia', 'Kentucky', 
            'Mississippi', 'North Carolina', 'South Carolina', 'Tennessee']), 
            'EPA_REGION'] = 4
        # Region 5 – Chicago (serving IL, IN, MI, MN, OH, and WI)
        df.loc[df['State Name'].isin(['Illinois', 'Indiana', 'Michigan', 
            'Minnesota', 'Ohio', 'Wisconsin']), 'EPA_REGION'] = 5
        # Region 6 – Dallas (serving AR, LA, NM, OK, and TX)
        df.loc[df['State Name'].isin(['Arkansas', 'Louisiana', 'New Mexico', 
            'Oklahoma', 'Texas']), 'EPA_REGION'] = 6
        # Region 7 - Kansas City (serving IA, KS, MO, and NE)
        df.loc[df['State Name'].isin(['Iowa', 'Kansas', 'Missouri', 'Nebraska']), 
            'EPA_REGION'] = 7
        # Region 8 – Denver (serving CO, MT, ND, SD, UT, and WY)
        df.loc[df['State Name'].isin(['Colorado','Montana', 'North Dakota', 
            'South Dakota', 'Utah', 'Wyoming']), 'EPA_REGION'] = 8
        # Region 9 - San Francisco (serving AZ, CA, HI, NV, American Samoa, 
        # Commonwealth of the Northern Mariana Islands, Federated States of 
        # Micronesia, Guam, Marshall Islands, and Republic of Palau)
        df.loc[df['State Name'].isin(['Arizona', 'California', 'Hawaii', 
            'Nevada']), 'EPA_REGION'] = 9
        # Region 10 – Seattle (serving AK, ID, OR, WA and 271 native tribes)
        df.loc[df['State Name'].isin(['Alaska', 'Idaho', 'Oregon', 'Washington']), 
            'EPA_REGION'] = 10
        return df
    no2_aqs2010 = pd.read_csv(DIR_AQ+'NO2_2010_bymonitor_harmonized.csv', 
        sep=',', engine='python')
    no2_aqs2015 = pd.read_csv(DIR_AQ+'NO2_2015_bymonitor_harmonized.csv', 
        sep=',', engine='python')
    no2_aqs2019 = pd.read_csv(DIR_AQ+'NO2_2019_bymonitor_harmonized.csv', 
        sep=',', engine='python')
    pm25_aqs2010 = pd.read_csv(DIR_AQ+'PM25_2010_bymonitor_harmonized.csv', 
        sep=',', engine='python')
    pm25_aqs2015 = pd.read_csv(DIR_AQ+'PM25_2015_bymonitor_harmonized.csv', 
        sep=',', engine='python')
    pm25_aqs2019 = pd.read_csv(DIR_AQ+'PM25_2019_bymonitor_harmonized.csv', 
        sep=',', engine='python')    
    # Find EPA Region 
    no2_aqs2010 =  find_eparegion(no2_aqs2010)
    no2_aqs2015 =  find_eparegion(no2_aqs2015)
    no2_aqs2019 =  find_eparegion(no2_aqs2019)
    pm25_aqs2010 =  find_eparegion(pm25_aqs2010)
    pm25_aqs2015 =  find_eparegion(pm25_aqs2015)
    pm25_aqs2019 = find_eparegion(pm25_aqs2019)
    # Make colormap for EPA Region 
    col_dict = {1:color1, 2:color2, 3:color3, 4:color4, 5:color5, 6:color6,
        7:color7, 8:'#97BCE9', 9:'#C8D7DF', 10:'#D4F4F8'}
    col_dict = {1:'k', 2:'k', 3:'k', 4:'k', 5:'k', 6:'k', 7:'k', 8:'k', 9:'k', 
        10:'k'}
    # Create a colormar from our list of colors
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    proj = ccrs.PlateCarree(central_longitude=0.0)
    fig = plt.figure(figsize=(9,9))
    axtl = plt.subplot2grid((3,6),(0,0), colspan=3, projection=proj)
    axtr = plt.subplot2grid((3,6),(0,3), colspan=3, projection=proj)
    ax1 = plt.subplot2grid((3,6),(1,0), colspan=2)
    ax2 = plt.subplot2grid((3,6),(1,2), colspan=2)
    ax3 = plt.subplot2grid((3,6),(1,4), colspan=2)
    ax4 = plt.subplot2grid((3,6),(2,0), colspan=2)
    ax5 = plt.subplot2grid((3,6),(2,2), colspan=2)
    ax6 = plt.subplot2grid((3,6),(2,4), colspan=2)
    # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()   
    usa = [x.attributes['ADM0_A3'] for x in countries]
    usa = np.where(np.array(usa)=='USA')[0][0]
    usa = list(reader.geometries())[usa].geoms
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shapereader.Reader(lakes)
    lakes = lakes_reader.records()   
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names)=='Lake Superior') |
        (np.array(lake_names)=='Lake Michigan') | 
        (np.array(lake_names)=='Lake Huron') |
        (np.array(lake_names)=='Lake Erie') |
        (np.array(lake_names)=='Lake Ontario'))[0]
    great_lakes = itemgetter(*great_lakes)(list(lakes_reader.geometries()))
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shpreader.natural_earth(resolution='10m', category='cultural', 
        name=shapename)
    states_shp = shpreader.Reader(states_shp)
    # Find shapefiles of states of interest for inset maps
    states_all = list(states_shp.records())
    states_all_name = []
    for s in states_all:
        states_all_name.append(s.attributes['name'])
    states_all = np.array(states_all)    
    alaska = states_all[np.where(np.array(states_all_name)=='Alaska')[0]][0]
    hawaii = states_all[np.where(np.array(states_all_name)=='Hawaii')[0]][0]
    puertorico = states_all[np.where(np.array(states_all_name)==
        'PRI-00 (Puerto Rico aggregation)')[0]][0]
    # Set titles
    axtl.set_title('(A) NO$_{\mathregular{2}}$ monitors', loc='left')
    axtr.set_title('(B) PM$_{\mathregular{2.5}}$ monitors', loc='left')
    ax1.set_title('(C) 2010 NO$_{\mathregular{2}}$', loc='left')
    ax2.set_title('(D) 2015 NO$_{\mathregular{2}}$', loc='left')
    ax3.set_title('(E) 2019 NO$_{\mathregular{2}}$', loc='left')
    ax4.set_title('(F) 2010 PM$_{\mathregular{2.5}}$', loc='left')
    ax5.set_title('(G) 2015 PM$_{\mathregular{2.5}}$', loc='left')
    ax6.set_title('(H) 2019 PM$_{\mathregular{2.5}}$', loc='left')
    # Colormaps
    cmap = plt.cm.get_cmap('bwr', 8)
    norm = mpl.colors.Normalize(vmin=-5, vmax=5)
    # Plot location of monitors 
    for ax, monitors in zip([axtl, axtr], [no2_aqs2015, pm25_aqs2015]): 
        ax.scatter(monitors['Longitude'], monitors['Latitude'], transform=proj, 
            zorder=30, clip_on=True, s=7, alpha=0.4, 
            c=(monitors['Arithmetic Mean']-monitors['DATASET_VAL']), ec='None',
            cmap=cmap, norm=norm) 
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)
        ax.add_geometries(usa, crs=proj, lw=0.25, facecolor='None', 
            edgecolor='k', zorder=1)
        ax.add_geometries(great_lakes, crs=proj, facecolor='w',
            lw=0.25, edgecolor='k', alpha=1., zorder=12)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 
            'ocean', '10m', edgecolor='None', facecolor='w', alpha=1.), 
            zorder=11)
        ax.axis('off')
        # Add state outlines
        for astate in states_shp.records():
            if astate.attributes['sr_adm0_a3']=='USA':
                geometry = astate.geometry
                ax.add_geometries([geometry], crs=ccrs.PlateCarree(), 
                    facecolor='#f2f2f2', lw=0.5, edgecolor='w', alpha=1., 
                    zorder=0)
        # Hawaii 
        axes_extent = (ax.get_position().x0+0.015, ax.get_position().y0-0.005, 
            (ax.get_position().x1-ax.get_position().x0)*0.2,
            (ax.get_position().x1-ax.get_position().x0)*0.2)
        add_insetmap(axes_extent, (-162, -154, 18.75, 23), '', 
            hawaii.geometry, monitors['Longitude'], monitors['Latitude'], 
            6, proj, sc=(monitors['Arithmetic Mean']-monitors['DATASET_VAL']), 
            cmap=cmap, norm=norm) 
        # Alaska
        axes_extent = (ax.get_position().x0-0.06, ax.get_position().y0-0.01, 
            (ax.get_position().x1-ax.get_position().x0)*0.25,
            (ax.get_position().x1-ax.get_position().x0)*0.25)
        add_insetmap(axes_extent, (-179.99, -130, 49, 73), '', 
            alaska.geometry, monitors['Longitude'], monitors['Latitude'], 
            6, proj, sc=(monitors['Arithmetic Mean']-monitors['DATASET_VAL']), 
            cmap=cmap, norm=norm) 
        # Puerto Rico 
        axes_extent = (ax.get_position().x0+0.085, ax.get_position().y0-0.01, 
            (ax.get_position().x1-ax.get_position().x0)*0.18,
            (ax.get_position().x1-ax.get_position().x0)*0.18)        
        add_insetmap(axes_extent, (-68., -65., 17.5, 19.), '', 
            puertorico.geometry, monitors['Longitude'], 
            monitors['Latitude'], 6, proj, sc=(monitors['Arithmetic Mean']-
            monitors['DATASET_VAL']), cmap=cmap, norm=norm) 
    # Plotting scatterplots 
    ax1.scatter(no2_aqs2010['Arithmetic Mean'], no2_aqs2010['DATASET_VAL'], 
       c=no2_aqs2010['EPA_REGION'], cmap=cm, zorder=10, s=2)
    ax2.scatter(no2_aqs2015['Arithmetic Mean'], no2_aqs2015['DATASET_VAL'], 
       c=no2_aqs2015['EPA_REGION'], cmap=cm, zorder=10, s=2)
    ax3.scatter(no2_aqs2019['Arithmetic Mean'], no2_aqs2019['DATASET_VAL'], 
       c=no2_aqs2019['EPA_REGION'], cmap=cm, zorder=10, s=2)
    ax4.scatter(pm25_aqs2010['Arithmetic Mean'], pm25_aqs2010['DATASET_VAL'],
       c=pm25_aqs2010['EPA_REGION'], cmap=cm, zorder=10, s=2)
    ax5.scatter(pm25_aqs2015['Arithmetic Mean'], pm25_aqs2015['DATASET_VAL'], 
       c=pm25_aqs2015['EPA_REGION'], cmap=cm, zorder=10, s=2)
    ax6.scatter(pm25_aqs2019['Arithmetic Mean'], pm25_aqs2019['DATASET_VAL'], 
       c='k', cmap=cm, zorder=10, s=2)    
    # Ticks and labels for NO2 plots
    for ax in [ax1, ax2, ax3]: 
        ax.set_xlim([0,30])
        ax.set_xticks(np.linspace(0,30,7))
        ax.set_ylim([0,30])
        ax.set_yticks(np.linspace(0,30,7))
        ax.set_yticklabels([])
        ax.set_xlabel('Observed NO$_{2}$ [ppbv]', loc='left')     
    # Add 1:1 and 2:1/1:2 lines, statistics, etc
    for ax, df in zip([ax1, ax2, ax3, ax4, ax5, ax6], [no2_aqs2010, no2_aqs2015, 
        no2_aqs2019, pm25_aqs2010, pm25_aqs2015, pm25_aqs2019]):
        # 1:1 and 2:1/1:2 lines
        ax.plot(np.linspace(0, 30 ,100), np.linspace(0, 30 ,100), '--',
            lw=1., color='grey', zorder=0, label='1:1')
        ax.plot(np.linspace(0, 30, 100), 2*np.linspace(0, 30, 100), '-', 
            lw=0.5, color='grey', zorder=0)    
        ax.plot(np.linspace(0, 30, 100), 0.5*np.linspace(0, 30, 100), '-', 
            lw=0.5, color='grey', zorder=0, label='1:2 and 2:1')        
        # Line of best fit
        idx = np.isfinite(df['Arithmetic Mean']) & np.isfinite(df['DATASET_VAL'])
        fit = np.polyfit(df['Arithmetic Mean'][idx], df['DATASET_VAL'][idx], 1)
        # Correlation coefficient
        r = np.corrcoef(df['Arithmetic Mean'][idx], df['DATASET_VAL'][idx])[0,1]
        # Normalized mean bias
        nmb = (np.nansum(df['DATASET_VAL'][idx]-df['Arithmetic Mean'][idx])/
            np.nansum(df['Arithmetic Mean'][idx]))
        ax.plot(np.linspace(0, 30, 100), np.poly1d(fit)(
            np.linspace(0, 30, 100)), '-r', zorder=12, label='Fit')
        ax.text(0.03, 0.94, 'm=%.2f, b=%.2f'%(fit[0],fit[1]), ha='left', 
            va='center', transform=ax.transAxes, fontsize=8)    
        ax.text(0.03, 0.86, 'N=%d'%np.where(idx==True)[0].shape[0], ha='left', 
            va='center', transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.78, 'NMB=%.2f'%nmb, ha='left', va='center', 
            transform=ax.transAxes, fontsize=8)    
        ax.text(0.03, 0.7, 'r=%.2f'%r, ha='left', va='center', 
            transform=ax.transAxes, fontsize=8)
    ax1.set_ylabel('Anenberg, Mohegh, et al.\n(2022) NO$_{2}$ [ppbv]', 
        loc='bottom') 
    ax1.set_yticklabels([int(x) for x in np.linspace(0,30,7)])
    # Ticks and labels for PM2.5 plots
    for ax in [ax4, ax5, ax6]: 
        ax.set_xlim([0,20])
        ax.set_xticks(np.linspace(0,20,6))
        ax.set_ylim([0,20])
        ax.set_yticks(np.linspace(0,20,6))
        ax.set_yticklabels([])
        ax.set_xlabel('Observed PM$_{2.5}$ [$\mathregular{\mu}$g '+\
            'm$^{\mathregular{-3}}$]', loc='left') 
        # ax.set_aspect('equal', 'box')
    ax4.set_ylabel('van Donkelaar et al. (2021)\nPM$_{2.5}$'+\
        ' [$\mathregular{\mu}$g m$^{\mathregular{-3}}$]', loc='bottom') 
    ax4.set_yticklabels([int(x) for x in np.linspace(0,20,6)])
    plt.subplots_adjust(hspace=0.6, wspace=0.3)#, right=0.85)

    # Add colorbar
    cax = fig.add_axes([axtr.get_position().x1+0.01, 
        axtr.get_position().y0, 0.01,
        (axtr.get_position().y1-axtr.get_position().y0)])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, 
        spacing='proportional', orientation='vertical', extend='both', 
        label='[$\mathregular{\mu}$g m$^{\mathregular{-3}}$ | ppbv]')#, ticks=np.linspace(0,24,5))    
    # Add legend
    ax5.legend(ncol=3, frameon=False, bbox_to_anchor=(0.4, -0.7), loc=8,
        fontsize=14)
    plt.savefig(DIR_FIG+'figS2.png', dpi=500)
    return

def figS3(): 
    """
    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    cfpm=4.15
    cfno2=5.3675
    # Load MRBRT RR estimates
    rrno2asthma = pd.read_csv(DIR_GBD+'no2_rr_draws_summary.csv', sep=',',
        engine='python')
    rrpmihd = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_ihd.csv')
    rrpmst = pd.read_csv(DIR_GBD+'mrbrt_summary/cvd_stroke.csv')
    rrpmcopd = pd.read_csv(DIR_GBD+'mrbrt_summary/resp_copd.csv')
    rrpmlc = pd.read_csv(DIR_GBD+'mrbrt_summary/neo_lung.csv')
    rrpmdm = pd.read_csv(DIR_GBD+'mrbrt_summary/t2_dm.csv')
    rrpmlri = pd.read_csv(DIR_GBD+'mrbrt_summary/lri.csv')
    # # # # Find closest exposure spline to counterfactual PM2.5 and NO2 
    # concentrations
    cipm = rrpmihd['exposure'].sub(cfpm).abs().idxmin()
    cino2 = rrno2asthma['exposure'].sub(cfno2).abs().idxmin()
    cirrno2asthma = rrno2asthma.iloc[cino2]['mean']
    cirrno2asthmalower = rrno2asthma.iloc[cino2]['lower']
    cirrno2asthmaupper = rrno2asthma.iloc[cino2]['upper']
    cirrpmihd = rrpmihd.iloc[cipm]['mean']
    cirrpmihdlower = rrpmihd.iloc[cipm]['lower']
    cirrpmihdupper = rrpmihd.iloc[cipm]['upper']
    cirrpmst = rrpmst.iloc[cipm]['mean']
    cirrpmstlower = rrpmst.iloc[cipm]['lower']
    cirrpmstupper = rrpmst.iloc[cipm]['upper']
    cirrpmcopd = rrpmcopd.iloc[cipm]['mean']
    cirrpmcopdlower = rrpmcopd.iloc[cipm]['lower']
    cirrpmcopdupper = rrpmcopd.iloc[cipm]['upper']
    cirrpmlc = rrpmlc.iloc[cipm]['mean']
    cirrpmlclower = rrpmlc.iloc[cipm]['lower']
    cirrpmlcupper = rrpmlc.iloc[cipm]['upper']
    cirrpmdm = rrpmdm.iloc[cipm]['mean']
    cirrpmdmlower = rrpmdm.iloc[cipm]['lower']
    cirrpmdmupper = rrpmdm.iloc[cipm]['upper']
    cirrpmlri = rrpmlri.iloc[cipm]['mean']
    cirrpmlrilower = rrpmlri.iloc[cipm]['lower']
    cirrpmlriupper = rrpmlri.iloc[cipm]['upper']    
    
    # Calculate the attributable fraction corresponding to the low 
    # concentration theshold/TMREL ("counterfactual") from the GBD
    # Study. For NO2-attributable cases of paediatric asthma incidence, 
    # the attributable fraction takes the form of a log-linear concentration-    
    # response function that was epidemiologically derived. For PM2.5-
    # attrituable premature mortality, 
    afno2asthma = (1-np.exp(-cirrno2asthma))
    afno2asthmalower = (1-np.exp(-cirrno2asthmalower))
    afno2asthmaupper = (1-np.exp(-cirrno2asthmaupper))
    afpmlri = (cirrpmlri-1.)/cirrpmlri
    afpmlrilower = (cirrpmlrilower-1.)/cirrpmlrilower
    afpmlriupper = (cirrpmlriupper-1.)/cirrpmlriupper
    afpmdm = (cirrpmdm-1.)/cirrpmdm
    afpmdmlower = (cirrpmdmlower-1.)/cirrpmdmlower
    afpmdmupper = (cirrpmdmupper-1.)/cirrpmdmupper
    afpmcopd = (cirrpmcopd-1.)/cirrpmcopd
    afpmcopdlower = (cirrpmcopdlower-1.)/cirrpmcopdlower
    afpmcopdupper = (cirrpmcopdupper-1.)/cirrpmcopdupper
    afpmlc = (cirrpmlc-1.)/cirrpmlc
    afpmlclower = (cirrpmlclower-1.)/cirrpmlclower
    afpmlcupper = (cirrpmlcupper-1.)/cirrpmlcupper
    afpmihd = (cirrpmihd-1.)/cirrpmihd
    afpmihdlower = (cirrpmihdlower-1.)/cirrpmihdlower
    afpmihdupper = (cirrpmihdupper-1.)/cirrpmihdupper
    afpmst = (cirrpmst-1.)/cirrpmst
    afpmstlower = (cirrpmstlower-1.)/cirrpmstlower
    afpmstupper = (cirrpmstupper-1.)/cirrpmstupper
    # Calculate NO2; subtract off counterfactual
    afno2asthma = (1-np.exp(-rrno2asthma['mean']))-afno2asthma 
    afno2asthmalower = (1-np.exp(-rrno2asthma['lower']))-afno2asthmalower
    afno2asthmaupper = (1-np.exp(-rrno2asthma['upper']))-afno2asthmaupper
    afpmlri = ((rrpmlri['mean']-1.)/rrpmlri['mean'])-afpmlri
    afpmlrilower = ((rrpmlri['lower']-1.)/rrpmlri['lower'])-afpmlrilower
    afpmlriupper = ((rrpmlri['upper']-1.)/rrpmlri['upper'])-afpmlriupper
    afpmdm = ((rrpmdm['mean']-1.)/rrpmdm['mean'])-afpmdm
    afpmdmlower = ((rrpmdm['lower']-1.)/rrpmdm['lower'])-afpmdmlower
    afpmdmupper = ((rrpmdm['upper']-1.)/rrpmdm['upper'])-afpmdmupper
    afpmcopd = ((rrpmcopd['mean']-1.)/rrpmcopd['mean'])-afpmcopd
    afpmcopdlower = ((rrpmcopd['lower']-1.)/rrpmcopd['lower'])-afpmcopdlower
    afpmcopdupper = ((rrpmcopd['upper']-1.)/rrpmcopd['upper'])-afpmcopdupper
    afpmlc = ((rrpmlc['mean']-1.)/rrpmlc['mean'])-afpmlc
    afpmlclower = ((rrpmlc['lower']-1.)/rrpmlc['lower'])-afpmlclower
    afpmlcupper = ((rrpmlc['upper']-1.)/rrpmlc['upper'])-afpmlcupper
    afpmst = ((rrpmst['mean']-1.)/rrpmst['mean'])-afpmst
    afpmstlower = ((rrpmst['lower']-1.)/rrpmst['lower'])-afpmstlower
    afpmstupper = ((rrpmst['upper']-1.)/rrpmst['upper'])-afpmstupper
    afpmihd = ((rrpmihd['mean']-1.)/rrpmihd['mean'])-afpmihd
    afpmihdlower = ((rrpmihd['lower']-1.)/rrpmihd['lower'])-afpmihdlower
    afpmihdupper = ((rrpmihd['upper']-1.)/rrpmihd['upper'])-afpmihdupper  
    # Force AF < 0 to 0 (protective effects?)
    afno2asthma.loc[afno2asthma<0]=0.004
    afno2asthmaupper.loc[afno2asthmaupper<0]=0.
    # afno2asthmalower.loc[afno2asthmalower<0]=0.
    afpmlri.loc[afpmlri<0]=0.004
    afpmdm.loc[afpmdm<0]=0.004
    afpmcopd.loc[afpmcopd<0]=0.004
    afpmlc.loc[afpmlc<0]=0.004
    afpmst.loc[afpmst<0]=0.004
    afpmihd.loc[afpmihd<0]=0.004
    # Plotting
    colorupper = '#FF7043'
    colorlower = '#0095A8'
    fig = plt.figure(figsize=(6,8))
    ax1 = plt.subplot2grid((4,2),(0,0))
    ax2 = plt.subplot2grid((4,2),(1,0))
    ax3 = plt.subplot2grid((4,2),(2,0))
    ax4 = plt.subplot2grid((4,2),(2,1))
    ax5 = plt.subplot2grid((4,2),(0,1))
    ax6 = plt.subplot2grid((4,2),(1,1))
    ax7 = plt.subplot2grid((4,2),(3,0))
    ax1.set_title('(A) Lower respiratory infection', loc='left')
    ax2.set_title('(C) Type 2 diabetes', loc='left')
    ax3.set_title('(E) COPD', loc='left')
    ax4.set_title('(F) Lung cancer', loc='left')
    ax5.set_title('(B) Stroke', loc='left')
    ax6.set_title('(D) Ischemic heart disease', loc='left')
    ax7.set_title('(G) Pediatric asthma', loc='left')
    ax1.plot(rrpmlri['exposure'], afpmlri.values*100., color='k', zorder=20)
    ax1.plot(rrpmlri['exposure'], afpmlrilower.values*100., color=colorlower, 
        zorder=20)
    ax1.plot(rrpmlri['exposure'], afpmlriupper.values*100., color=colorupper, 
        zorder=20)        
    ax2.plot(rrpmdm['exposure'], afpmdm.values*100., color='k', zorder=20)
    ax2.plot(rrpmdm['exposure'], afpmdmlower.values*100., color=colorlower, 
        zorder=20)
    ax2.plot(rrpmdm['exposure'], afpmdmupper.values*100., color=colorupper, 
        zorder=20)        
    ax3.plot(rrpmcopd['exposure'], afpmcopd.values*100., color='k', zorder=20)
    ax3.plot(rrpmcopd['exposure'], afpmcopdlower.values*100., color=colorlower, 
        zorder=20)
    ax3.plot(rrpmcopd['exposure'], afpmcopdupper.values*100., color=colorupper, 
        zorder=20)        
    ax4.plot(rrpmlc['exposure'], afpmlc.values*100., color='k', zorder=20)
    ax4.plot(rrpmlc['exposure'], afpmlclower.values*100., color=colorlower, 
        zorder=20)
    ax4.plot(rrpmlc['exposure'], afpmlcupper.values*100., color=colorupper, 
        zorder=20)        
    ax5.plot(rrpmst['exposure'], afpmst.values*100., color='k', zorder=20)
    ax5.plot(rrpmst['exposure'], afpmstlower.values*100., color=colorlower, 
        zorder=20)
    ax5.plot(rrpmst['exposure'], afpmstupper.values*100., color=colorupper, 
        zorder=20)        
    ax6.plot(rrpmihd['exposure'], afpmihd.values*100., color='k', zorder=20)
    ax6.plot(rrpmihd['exposure'], afpmihdlower.values*100., color=colorlower, 
        zorder=20)
    ax6.plot(rrpmihd['exposure'], afpmihdupper.values*100., color=colorupper, 
        zorder=20)    
    ax7.plot(rrno2asthma['exposure'], afno2asthma.values*100., 
        color='k', ls='-', zorder=30, label='Mean')
    ax7.plot(rrno2asthma.loc[rrno2asthma['exposure']>cfno2]['exposure'], 
        afno2asthmalower.values[54:]*100., 
        color=colorlower, ls='-', zorder=20, label='Lower')
    ax7.plot(rrno2asthma['exposure'], afno2asthmaupper.values*100., 
        color=colorupper, ls='-', zorder=20, label='Upper')    
    ax7.set_xlabel('NO$_{\mathregular{2}}$ [ppbv]')
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim([0,20])
        ax.set_xticks(np.linspace(0,20,5))
        ax.set_xticklabels([])
        ax.set_ylim([0,25.])
        ax.set_yticks(np.linspace(0,25,6))
        ax.set_yticklabels([])
        ax.grid(axis='both', which='major', zorder=0, color='grey', ls='-', 
            lw=0.25)
        for k, spine in ax.spines.items():
            spine.set_zorder(30)    
        # Denote TMREL
        ax.axvspan(0, cfpm, alpha=1., color='lightgrey', zorder=10)
    for ax in [ax4, ax3]:
        ax.set_xlabel('PM$_{\mathregular{2.5}}$ [${\mu}$g m$'+\
            '^{\mathregular{-3}}$]')
        ax.set_xticklabels(['0','5','10','15','20'])
    for ax in [ax1, ax2, ax3]:
        ax.set_yticklabels(['0','5','10','15','20','25'])
    fig.text(0.02, 0.5, 'Population attributable fraction [%]', va='center', 
         rotation='vertical', fontsize=14)
    ax7.set_xlim([0,40])
    ax7.set_xticks(np.linspace(0,40,5))
    ax7.set_ylim([-50,50])
    ax7.set_yticks(np.linspace(-50,50,5))
    ax7.legend(ncol=1, frameon=False, bbox_to_anchor=(1.4,1.1), fontsize=14)
    ax7.axvspan(0, cfno2, alpha=1., color='lightgrey', zorder=10)
    ax7.grid(axis='both', which='major', zorder=0, color='grey', ls='-', lw=0.25)
    for k, spine in ax7.spines.items(): 
        spine.set_zorder(30)
    plt.subplots_adjust(hspace=1., right=0.95)
    plt.savefig(DIR_FIG+'figS3.png', dpi=500)
    return

def figS4():     
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from cartopy.io import shapereader
    import matplotlib
    import pandas as pd
    # Open GBD 2019 incidence rates
    ihd = pd.read_csv(DIR_GBD+'incidence/'+'IHME-GBD_2019_DATA-cvd_ihd.csv', 
        engine='python', sep=',')
    stroke = pd.read_csv(DIR_GBD+'incidence/'+
        'IHME-GBD_2019_DATA-cvd_stroke.csv', engine='python', sep=',')
    lri = pd.read_csv(DIR_GBD+'incidence/'+'IHME-GBD_2019_DATA-lri.csv', 
        engine='python', sep=',')
    lung = pd.read_csv(DIR_GBD+'incidence/'+
        'IHME-GBD_2019_DATA-neo_lung.csv', engine='python', sep=',')
    copd = pd.read_csv(DIR_GBD+'incidence/'+
        'IHME-GBD_2019_DATA-resp_copd.csv', engine='python', sep=',')
    dm = pd.read_csv(DIR_GBD+'incidence/'+'IHME-GBD_2019_DATA-t2_dm.csv', 
        engine='python', sep=',') 
    asthma = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-9abad4e5-1.csv', 
        engine='python', sep=',')
    # Group by year and calculate some statistics regarding nationwide-averaged
    # trends
    from scipy import stats    
    ihdyr = ihd.groupby([ihd.year]).mean()
    strokeyr = ihd.groupby([stroke.year]).mean()
    lriyr = ihd.groupby([lri.year]).mean()
    lungyr = ihd.groupby([lung.year]).mean()
    copdyr = ihd.groupby([copd.year]).mean()
    dmyr = ihd.groupby([dm.year]).mean()
    asthmayr = ihd.groupby([asthma.year]).mean()
    trend_ihdyr = stats.linregress(ihdyr.index, ihdyr.val)
    trend_strokeyr = stats.linregress(strokeyr.index, strokeyr.val)
    trend_lriyr = stats.linregress(lriyr.index, lriyr.val)
    trend_lungyr = stats.linregress(lungyr.index, lungyr.val)
    trend_copdyr = stats.linregress(copdyr.index, copdyr.val)
    trend_dmyr = stats.linregress(dmyr.index, dmyr.val)
    trend_asthmayr = stats.linregress(asthmayr.index, asthmayr.val)    
    print('# # # # Nationwide IHD incidence trends')
    print('Slope/p-value = ', trend_ihdyr.slope, '/', trend_ihdyr.pvalue)
    pc_ihd = ((ihdyr.val.values[-1]-ihdyr.val.values[0])/
        ihdyr.val.values[0])*100
    print('Percent change', pc_ihd, '%')
    print('\n')
    print('# # # # Nationwide stroke incidence trends')
    print('Slope/p-value = ', trend_strokeyr.slope, '/', trend_strokeyr.pvalue)
    pc_stroke = ((strokeyr.val.values[-1]-strokeyr.val.values[0])/
        strokeyr.val.values[0])*100
    print('Percent change', pc_stroke, '%')
    print('\n')
    print('# # # # Nationwide lower respiratory infection incidence trends')
    print('Slope/p-value = ', trend_lriyr.slope, '/', trend_lriyr.pvalue)
    pc_lri = ((lriyr.val.values[-1]-lriyr.val.values[0])/
        lriyr.val.values[0])*100
    print('Percent change', pc_lri, '%')
    print('\n')
    print('# # # # Nationwide lung cancer incidence trends')
    print('Slope/p-value = ', trend_lungyr.slope, '/', trend_lungyr.pvalue)
    pc_lung = ((lungyr.val.values[-1]-lungyr.val.values[0])/
        lungyr.val.values[0])*100
    print('Percent change', pc_lung, '%')    
    print('\n')
    print('# # # # Nationwide COPD incidence trends')
    print('Slope/p-value = ', trend_copdyr.slope, '/', trend_copdyr.pvalue)
    pc_copd = ((copdyr.val.values[-1]-copdyr.val.values[0])/
        copdyr.val.values[0])*100
    print('Percent change', pc_copd, '%')    
    print('\n')
    print('# # # # Nationwide diabetes mellitus incidence trends')
    print('Slope/p-value = ', trend_dmyr.slope, '/', trend_dmyr.pvalue)
    pc_dm = ((dmyr.val.values[-1]-dmyr.val.values[0])/dmyr.val.values[0])*100
    print('Percent change', pc_dm, '%')    
    print('\n')
    print('# # # # Nationwide pediatric asthma incidence trends')
    print('Slope/p-value = ', trend_asthmayr.slope, '/', trend_asthmayr.pvalue)
    pc_asthma = ((asthmayr.val.values[-1]-asthmayr.val.values[0])/
        asthmayr.val.values[0])*100
    print('Percent change', pc_asthma, '%')    
    print('\n')
    # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shapereader.Reader(shpfilename)
    countries = reader.records()   
    usaidx = [x.attributes['ADM0_A3'] for x in countries]
    usaidx = np.where(np.in1d(np.array(usaidx), ['PRI','USA'])==True)
    usa = list(reader.geometries())
    usa = np.array(usa, dtype=object)[usaidx[0]]
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shapereader.Reader(lakes)
    lakes = lakes_reader.records()   
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names)=='Lake Superior') |
        (np.array(lake_names)=='Lake Michigan') | 
        (np.array(lake_names)=='Lake Huron') |
        (np.array(lake_names)=='Lake Erie') |
        (np.array(lake_names)=='Lake Ontario'))[0]
    great_lakes = np.array(list(lakes_reader.geometries()), 
        dtype=object)[great_lakes]
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shapereader.natural_earth(resolution='10m', category='cultural', 
        name=shapename)
    states_shp = shapereader.Reader(states_shp)
    # Find shapefiles of states of interest for inset maps
    states_all = list(states_shp.records())
    states_all_name = []
    for s in states_all:
        states_all_name.append(s.attributes['name'])
    states_all = np.array(states_all)    
    alaska = states_all[np.where(np.array(states_all_name)=='Alaska')[0]][0]
    hawaii = states_all[np.where(np.array(states_all_name)=='Hawaii')[0]][0]
    puertorico = states_all[np.where(np.array(states_all_name)==
        'PRI-00 (Puerto Rico aggregation)')[0]][0]
    # Define colorscheme
    cmap = plt.get_cmap('magma_r', 7)
    normlri = matplotlib.colors.Normalize(vmin=3500, vmax=5250)
    normstroke = matplotlib.colors.Normalize(vmin=250, vmax=450)
    normdm = matplotlib.colors.Normalize(vmin=350, vmax=600)
    normihd = matplotlib.colors.Normalize(vmin=400, vmax=1000)
    normcopd = matplotlib.colors.Normalize(vmin=250, vmax=500)
    normlung = matplotlib.colors.Normalize(vmin=20, vmax=120)
    normasthma = matplotlib.colors.Normalize(vmin=2000, vmax=4000)
    # Define endpoint order
    endpoints = [lri, stroke, dm, ihd, copd, lung, asthma]
    norms = [normlri, normstroke, normdm, normihd, normcopd, normlung, normasthma]
    endpoints_names = ['Lower respir-\natory infection', 'Stroke\n', 
        'Type 2\ndiabetes', 'Ischemic\nheart disease', 'COPD\n', 'Lung cancer\n', 
        'Pediatric\nasthma']
    # Initialize figure, subplots 
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(7,3,figsize=(8.5,11), 
        subplot_kw=dict(projection=proj))
    plt.subplots_adjust(left=0.12, top=0.95, bottom=0.05)    
    for i, ax in enumerate(axs.flatten()):
        # Select endpoint and year
        endpointloc = np.int(np.round((i/3)-(1./3)))
        endpoint = endpoints[endpointloc]
        endpoint_name = endpoints_names[endpointloc]
        if i in [0,3,6,9,12,15,18]:
            year = 2010
        elif i in [1,4,7,10,13,16,19]:
            year = 2015
        else: 
            year = 2019
        endpoint = endpoint.loc[endpoint.year==year]
        # Add endpoint name as y-label
        if i in [0,3,6,9,12,15,18]:
            ax.text(-0.2, -0.05, endpoint_name, va='bottom', ha='left',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=14)
        ax.set_extent([-125,-66.5, 24.5, 49.48], proj)
        ax.add_geometries(usa, crs=proj, lw=0.25, facecolor='None', 
            edgecolor='k', zorder=1)
        ax.add_geometries(great_lakes, crs=proj, facecolor='w',lw=0.25, 
            edgecolor='k', alpha=1., zorder=12)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m', 
            edgecolor='None', facecolor='w', alpha=1.), zorder=11)
        ax.axis('off')
        # Add colorbar
        if i in [2,5,8,11,14,17,20]:
            cax = fig.add_axes([axs.flatten()[i].get_position().x1+0.015, 
                axs.flatten()[i].get_position().y0, 0.012, 
                (axs.flatten()[i].get_position().y1-
                axs.flatten()[i].get_position().y0)])
            mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norms[endpointloc], 
                orientation='vertical', extend='both', 
                label='', ticks=np.linspace(norms[endpointloc].vmin, 
                norms[endpointloc].vmax, 3))
        # Color states by incidence rates and add state borders
        for astate in states_shp.records():
            if astate.attributes['name'] in ['Alaska', 'Hawaii', 
                'PRI-00 (Puerto Rico aggregation)']:
                # Parse information about rates in states
                if astate.attributes['name']=='Alaska':
                    fcak = endpoint.loc[endpoint['location_name']=='Alaska']
                if astate.attributes['name']=='Hawaii':
                    fchi = endpoint.loc[endpoint['location_name']=='Hawaii']
                if astate.attributes['name']=='PRI-00 (Puerto Rico aggregation)':
                    fcpr = endpoint.loc[endpoint['location_name']=='Puerto Rico']
                pass
            elif astate.attributes['sr_adm0_a3']=='USA':
                inrate = endpoint.loc[endpoint['location_name']==
                    astate.attributes['name']]
                geometry = astate.geometry
                ax.add_geometries([geometry], crs=proj, zorder=100, ec='k',
                    fc=cmap(norms[endpointloc](inrate.val.mean())), lw=0.25) 
        # Hawaii 
        axes_extent = (ax.get_position().x0-0.0, ax.get_position().y0-0.015, 
            (ax.get_position().x1-ax.get_position().x0)*0.22,
            (ax.get_position().x1-ax.get_position().x0)*0.22)
        add_insetmap(axes_extent, (-162, -154, 18.75, 23), '', hawaii.geometry, 
            [0.], [0.], [0.], proj, 
            fc=matplotlib.colors.to_hex(cmap(norms[endpointloc](fchi.val.mean()))))
        # Alaska
        axes_extent = (ax.get_position().x0-0.04, ax.get_position().y0-0.01, 
            (ax.get_position().x1-ax.get_position().x0)*0.22,
            (ax.get_position().x1-ax.get_position().x0)*0.22)
        add_insetmap(axes_extent, (-179.99, -130, 49, 73), '', alaska.geometry, 
            [0.], [0.], [0.], proj, 
            fc=matplotlib.colors.to_hex(cmap(norms[endpointloc](fcak.val.mean()))))
        # Puerto Rico 
        axes_extent = (ax.get_position().x0+0.048, ax.get_position().y0-0.015, 
            (ax.get_position().x1-ax.get_position().x0)*0.16,
            (ax.get_position().x1-ax.get_position().x0)*0.16)        
        add_insetmap(axes_extent, (-68., -65., 17.5, 19.), '', puertorico.geometry, 
            [0.], [0.], [0.], proj, 
            fc=matplotlib.colors.to_hex(cmap(norms[endpointloc](fcpr.val.mean()))))
    # Subplot titles
    axs.flatten()[0].set_title('(A) 2010', loc='left')
    axs.flatten()[1].set_title('(B) 2015', loc='left')
    axs.flatten()[2].set_title('(C) 2019', loc='left')
    axs.flatten()[3].set_title('(D) 2010', loc='left')
    axs.flatten()[4].set_title('(E) 2015', loc='left')
    axs.flatten()[5].set_title('(F) 2019', loc='left')
    axs.flatten()[6].set_title('(G) 2010', loc='left')
    axs.flatten()[7].set_title('(H) 2015', loc='left')
    axs.flatten()[8].set_title('(I) 2019', loc='left')
    axs.flatten()[9].set_title('(J) 2010', loc='left')
    axs.flatten()[10].set_title('(K) 2015', loc='left')
    axs.flatten()[11].set_title('(L) 2019', loc='left')
    axs.flatten()[12].set_title('(M) 2010', loc='left')
    axs.flatten()[13].set_title('(N) 2015', loc='left')
    axs.flatten()[14].set_title('(O) 2019', loc='left')
    axs.flatten()[15].set_title('(P) 2010', loc='left')
    axs.flatten()[16].set_title('(Q) 2015', loc='left')
    axs.flatten()[17].set_title('(R) 2019', loc='left')
    axs.flatten()[18].set_title('(S) 2010', loc='left')
    axs.flatten()[19].set_title('(T) 2015', loc='left')
    axs.flatten()[20].set_title('(U) 2019', loc='left')
    plt.savefig(DIR_FIG+'figS4.pdf', dpi=500)
    return 

def figS6(burdents):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches    
    # For absolute concentrations/rates for racial subgroups
    PM25_mostwhite, PM25_leastwhite = [], []
    NO2_mostwhite, NO2_leastwhite = [], []
    pd_mostwhite, pd_leastwhite = [], []
    asthma_mostwhite, asthma_leastwhite = [], []
    # For relative disparities
    PM25_race_relative, NO2_race_relative = [], []
    pd_race_relative, asthma_race_relative = [], []
    # Same as above but for ethnic subgroups
    PM25_mosthisp, PM25_leasthisp = [], []
    NO2_mosthisp, NO2_leasthisp = [], []
    pd_mosthisp, pd_leasthisp = [], []
    asthma_mosthisp, asthma_leasthisp = [], []
    PM25_ethnic_relative, NO2_ethnic_relative = [], []
    pd_ethnic_relative, asthma_ethnic_relative = [], []
    for year in np.arange(2010, 2020, 1):
        yearst = '%d-%d'%(year-4, year)
        burdenty = burdents.loc[burdents['YEAR']==yearst].copy(deep=True)
        # Define ethnoracial groups
        burdenty['fracwhite'] = ((burdenty[['race_nh_white','race_h_white'
            ]].sum(axis=1))/burdenty['race_tot'])
        burdenty['frachisp'] = (burdenty['race_h']/burdenty['race_tot'])    
        # Define extreme ethnoracial subgroups
        mostwhite = burdenty.iloc[np.where(burdenty.fracwhite >= 
            np.nanpercentile(burdenty.fracwhite, 90))]
        leastwhite = burdenty.iloc[np.where(burdenty.fracwhite <= 
            np.nanpercentile(burdenty.fracwhite, 10))]
        mosthisp = burdenty.iloc[np.where(burdenty.frachisp > 
            np.nanpercentile(burdenty.frachisp, 90))]
        leasthisp = burdenty.iloc[np.where(burdenty.frachisp < 
            np.nanpercentile(burdenty.frachisp, 10))]
        # Save off information for year 
        PM25_mostwhite.append(mostwhite.PM25.mean())
        PM25_leastwhite.append(leastwhite.PM25.mean())
        NO2_mostwhite.append(mostwhite.NO2.mean())
        NO2_leastwhite.append(leastwhite.NO2.mean())
        pd_mostwhite.append(mostwhite.BURDENPMALLRATE.mean())
        pd_leastwhite.append(leastwhite.BURDENPMALLRATE.mean())
        asthma_mostwhite.append(mostwhite.BURDENASTHMARATE.mean())
        asthma_leastwhite.append(leastwhite.BURDENASTHMARATE.mean())
        PM25_mosthisp.append(mosthisp.PM25.mean())
        PM25_leasthisp.append(leasthisp.PM25.mean())
        NO2_mosthisp.append(mosthisp.NO2.mean())
        NO2_leasthisp.append(leasthisp.NO2.mean())
        pd_mosthisp.append(mosthisp.BURDENPMALLRATE.mean())
        pd_leasthisp.append(leasthisp.BURDENPMALLRATE.mean())
        asthma_mosthisp.append(mosthisp.BURDENASTHMARATE.mean())
        asthma_leasthisp.append(leasthisp.BURDENASTHMARATE.mean())
        # Relative disparities
        PM25_race_relative.append(leastwhite.PM25.mean()/mostwhite.PM25.mean())
        NO2_race_relative.append(leastwhite.NO2.mean()/mostwhite.NO2.mean())
        pd_race_relative.append(leastwhite.BURDENPMALLRATE.mean()/
            mostwhite.BURDENPMALLRATE.mean())
        asthma_race_relative.append(leastwhite.BURDENASTHMARATE.mean()/
            mostwhite.BURDENASTHMARATE.mean())
        PM25_ethnic_relative.append(mosthisp.PM25.mean()/leasthisp.PM25.mean())
        NO2_ethnic_relative.append(mosthisp.NO2.mean()/leasthisp.NO2.mean())
        pd_ethnic_relative.append(mosthisp.BURDENPMALLRATE.mean()/
            leasthisp.BURDENPMALLRATE.mean())
        asthma_ethnic_relative.append(mosthisp.BURDENASTHMARATE.mean()/
            leasthisp.BURDENASTHMARATE.mean())    
    fig = plt.figure(figsize=(8.5,11))    
    ax1 = plt.subplot2grid((4,2),(0,0))
    ax1t = ax1.twinx()
    ax2 = plt.subplot2grid((4,2),(0,1))
    ax2t = ax2.twinx()
    ax3 = plt.subplot2grid((4,2),(1,0))
    ax3t = ax3.twinx()
    ax4 = plt.subplot2grid((4,2),(1,1))
    ax4t = ax4.twinx()
    ax5 = plt.subplot2grid((4,2),(2,0))
    ax5t = ax5.twinx()
    ax6 = plt.subplot2grid((4,2),(2,1))
    ax6t = ax6.twinx()
    ax7 = plt.subplot2grid((4,2),(3,0))
    ax7t = ax7.twinx()
    ax8 = plt.subplot2grid((4,2),(3,1))
    ax8t = ax8.twinx()
    years = np.arange(2010,2020,1)
    # Racial PM2.5 disparities
    ax1t.bar(years, PM25_race_relative, color='grey', zorder=0)
    ax1.plot(years, PM25_mostwhite, lw=2, color=color3, zorder=10)
    ax1.plot(years, PM25_leastwhite, lw=2, color=color2, zorder=10)
    # Ethnic PM2.5 disparities
    ax2t.bar(years, PM25_ethnic_relative, color='grey', zorder=0)
    ax2.plot(years, PM25_leasthisp, lw=2, color=color3, zorder=10)
    ax2.plot(years, PM25_mosthisp, lw=2, color=color2, zorder=10)
    # Racial NO2 disparities
    ax3t.bar(years, NO2_race_relative, color='grey', zorder=0)
    ax3.plot(years, NO2_mostwhite, lw=2, color=color3, zorder=10)
    ax3.plot(years, NO2_leastwhite, lw=2, color=color2, zorder=10)
    # Ethnic NO2 disparities
    ax4t.bar(years, NO2_ethnic_relative, color='grey', zorder=0)
    ax4.plot(years, NO2_leasthisp, lw=2, color=color3, zorder=10)
    ax4.plot(years, NO2_mosthisp, lw=2, color=color2, zorder=10)
    # Racial premature death disparities
    ax5t.bar(years, pd_race_relative, color='grey', zorder=0)
    ax5.plot(years, pd_mostwhite, lw=2, color=color3, zorder=10)
    ax5.plot(years, pd_leastwhite, lw=2, color=color2, zorder=10)
    # Ethnic premature death disparities
    ax6t.bar(years, pd_ethnic_relative, color='grey', zorder=0)
    ax6.plot(years, pd_leasthisp, lw=2, color=color3, zorder=10)
    ax6.plot(years, pd_mosthisp, lw=2, color=color2, zorder=10)
    # Racial asthma disparities
    ax7t.bar(years, asthma_race_relative, color='grey', zorder=0)
    ax7.plot(years, asthma_mostwhite, lw=2, color=color3, zorder=10)
    ax7.plot(years, asthma_leastwhite, lw=2, color=color2, zorder=10)
    # Ethnic asthma disparities
    ax8t.bar(years, asthma_ethnic_relative, color='grey', zorder=0)
    ax8.plot(years, asthma_leasthisp, lw=2, color=color3, zorder=10)
    ax8.plot(years, asthma_mosthisp, lw=2, color=color2, zorder=10)
    # Aesthetics    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:    
        ax.set_xlim([2010,2019])
        ax.set_xticks(np.arange(2010,2020,1))
        ax.set_xticklabels([])
    # PM25
    for ax in [ax1, ax2]:
        ax.set_ylim([0, 12])
        ax.set_yticks(np.linspace(0, 12, 5))
    for ax in [ax1t, ax2t]:
        ax.set_ylim([0.9,1.3])
    # NO2
    for ax in [ax3, ax4]:
        ax.set_ylim([0, 20])
        ax.set_yticks(np.linspace(0, 20, 5))
    for ax in [ax3t, ax4t]:
        ax.set_ylim([1.0, 2.5])
    # PM2.5-attributable mortality 
    for ax in [ax5, ax6]:
        ax.set_ylim([0, 28])
        ax.set_yticks(np.linspace(0, 28, 5))
    for ax in [ax5t, ax6t]:
        ax.set_ylim([0.6, 1.2])
        # ax.set_yticks(np.linspace(0.5, 1.1, 5))    
    # NO2-attributable asthma
    for ax in [ax7, ax8]:
        ax.set_xticklabels(['2010', '', '', '2013', '', '', '2016', '', '', 
            '2019'])
        ax.set_ylim([0, 450])
        ax.set_yticks(np.linspace(0, 500, 6))
    for ax in [ax7t, ax8t]:
        ax.set_ylim([1, 8])
    # Change default zorder     
    for ax, axt in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], 
        [ax1t, ax2t, ax3t, ax4t, ax5t, ax6t, ax7t, ax8t]):   
        ax.set_zorder(axt.get_zorder()+1)
        ax.patch.set_visible(False)
    # Y-axis labels for primary axes
    ax1.set_ylabel('PM$_{\mathregular{2.5}}$\n[$\mathregular{\mu}$g'+\
        ' m$^{\mathregular{-3}}$]', loc='bottom', fontsize=14)
    ax1.yaxis.set_label_coords(-0.18,0.0)
    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    ax3.set_ylabel('NO$_{\mathregular{2}}$\n[ppbv]', loc='bottom', fontsize=14)
    ax3.yaxis.set_label_coords(-0.18,0.0)
    ax4.set_ylabel('')
    ax4.set_yticklabels([])    
    ax5.set_ylabel('Premature deaths due \nto PM$_\mathregular{2.5}$ '+\
        '[per 100000]', loc='bottom', fontsize=14)
    ax5.yaxis.set_label_coords(-0.18,0.0)
    ax6.set_ylabel('')
    ax6.set_yticklabels([])    
    ax7.set_ylabel('New asthma cases due \nto NO$_\mathregular{2}$ '+\
        '[per 100000]', loc='bottom', fontsize=14)    
    ax7.yaxis.set_label_coords(-0.18,0.0)
    ax8.set_ylabel('')
    ax8.set_yticklabels([])    
    # Y-axis labels for secondary axes
    ax1t.set_ylabel('')
    ax1t.set_yticklabels([])
    ax2t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax2t.yaxis.set_label_coords(1.25,0.5)
    ax3t.set_ylabel('')
    ax3t.set_yticklabels([])
    ax4t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax4t.yaxis.set_label_coords(1.25,0.5)
    ax5t.set_ylabel('')
    ax5t.set_yticklabels([])
    ax6t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax6t.yaxis.set_label_coords(1.25,0.5)
    ax7t.set_ylabel('')
    ax7t.set_yticklabels([])
    ax8t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax8t.yaxis.set_label_coords(1.25,0.5)
    # Generate legend 
    patch2 = mpatches.Patch(color=color2, label='Least\nwhite')
    patch1 = mpatches.Patch(color=color3, label='Most\nwhite')
    all_handles = (patch2, patch1)
    ax7.legend(handles=all_handles, loc=8, frameon=False, ncol=2,
        bbox_to_anchor=(0.5, -0.5), fontsize=14)
    patch1 = mpatches.Patch(color=color2, label='Most\nHispanic')
    patch2 = mpatches.Patch(color=color3, label='Least\nHispanic')
    all_handles = (patch1, patch2)
    ax8.legend(handles=all_handles, loc=8, frameon=False, ncol=2,
        bbox_to_anchor=(0.5, -0.5), fontsize=14)
    # Subplot labels
    ax1.set_title('(A) Racial disparities', fontsize=14, loc='left')
    ax2.set_title('(B) Ethnic disparities', fontsize=14, loc='left')
    ax3.set_title('(C)', fontsize=14, loc='left')
    ax4.set_title('(D)', fontsize=14, loc='left')
    ax5.set_title('(E)', fontsize=14, loc='left')
    ax6.set_title('(F)', fontsize=14, loc='left')
    ax7.set_title('(G)', fontsize=14, loc='left')
    ax8.set_title('(H)', fontsize=14, loc='left')
    plt.subplots_adjust(wspace=0.3, top=0.95, bottom=0.1, hspace=0.3)
    plt.savefig(DIR_FIG+'figS6.png', dpi=600)
    return 

def figS7(burdents):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    # For absolute concentrations/rates for racial subgroups
    PM25_white, PM25_black = [], []
    NO2_white, NO2_black = [], []
    pd_white, pd_black = [], []
    asthma_white, asthma_black = [], []
    # For relative disparities
    PM25_race_relative, NO2_race_relative = [], []
    pd_race_relative, asthma_race_relative = [], []
    # Same as above but for ethnic subgroups
    PM25_hisp, PM25_nonhisp = [], []
    NO2_hisp, NO2_nonhisp = [], []
    pd_hisp, pd_nonhisp = [], []
    asthma_hisp, asthma_nonhisp = [], []
    PM25_ethnic_relative, NO2_ethnic_relative = [], []
    pd_ethnic_relative, asthma_ethnic_relative = [], []
    for year in np.arange(2010, 2020, 1):
        yearst = '%d-%d'%(year-4, year)
        burdenty = burdents.loc[burdents['YEAR']==yearst].copy(deep=True)
        # Define ethnoracial groups
        burdenty['fracwhite'] = ((burdenty[['race_nh_white','race_h_white'
            ]].sum(axis=1))/burdenty['race_tot'])
        burdenty['frachisp'] = (burdenty['race_h']/burdenty['race_tot'])    
        # Define population-weighted categories
        pm25black = ((burdenty['PM25']*burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1).sum())
        pm25white = ((burdenty['PM25']*
            burdenty[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
            burdenty[['race_nh_white','race_h_white']].sum(axis=1).sum())         
        no2black = ((burdenty['NO2']*burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1).sum())
        no2white = ((burdenty['NO2']*
            burdenty[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
            burdenty[['race_nh_white','race_h_white']].sum(axis=1).sum())       
        pdblack = ((burdenty['BURDENPMALLRATE']*burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1).sum())
        pdwhite = ((burdenty['BURDENPMALLRATE']*
            burdenty[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
            burdenty[['race_nh_white','race_h_white']].sum(axis=1).sum())       
        asthmablack = ((burdenty['BURDENASTHMARATE']*burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1).sum())
        asthmawhite = ((burdenty['BURDENASTHMARATE']*
            burdenty[['race_nh_white','race_h_white']].sum(axis=1)).sum()/
            burdenty[['race_nh_white','race_h_white']].sum(axis=1).sum())     
        pm25nh = ((burdenty['PM25']*burdenty['race_nh']).sum() /
            burdenty['race_nh'].sum())
        pm25h = ((burdenty['PM25']*burdenty['race_h']).sum() /
            burdenty['race_h'].sum())
        no2nh = ((burdenty['NO2']*burdenty['race_nh']).sum() /
            burdenty['race_nh'].sum())
        no2h = ((burdenty['NO2']*burdenty['race_h']).sum() /
            burdenty['race_h'].sum())
        pdnh = ((burdenty['BURDENPMALLRATE']*burdenty['race_nh']).sum() /
            burdenty['race_nh'].sum())
        pdh = ((burdenty['BURDENPMALLRATE']*burdenty['race_h']).sum() /
            burdenty['race_h'].sum())    
        asthmanh = ((burdenty['BURDENASTHMARATE']*burdenty['race_nh']).sum() /
            burdenty['race_nh'].sum())
        asthmah = ((burdenty['BURDENASTHMARATE']*burdenty['race_h']).sum() /
            burdenty['race_h'].sum())
        # For absolute concentrations/rates for racial subgroups
        PM25_white.append(pm25white)
        PM25_black.append(pm25black)
        NO2_white.append(no2white)
        NO2_black.append(no2black)
        pd_white.append(pdwhite)
        pd_black.append(pdblack)
        asthma_white.append(asthmawhite)
        asthma_black.append(asthmablack)
        PM25_race_relative.append(pm25black/pm25white)
        NO2_race_relative.append(no2black/no2white)
        pd_race_relative.append(pdblack/pdwhite)
        asthma_race_relative.append(asthmablack/asthmawhite)
        PM25_hisp.append(pm25h)
        PM25_nonhisp.append(pm25nh)
        NO2_hisp.append(no2h)
        NO2_nonhisp.append(no2nh)
        pd_hisp.append(pdh)
        pd_nonhisp.append(pdnh)
        asthma_hisp.append(asthmah)
        asthma_nonhisp.append(asthmanh)
        PM25_ethnic_relative.append(pm25h/pm25nh)
        NO2_ethnic_relative.append(no2h/no2nh)
        pd_ethnic_relative.append(pdh/pdnh)
        asthma_ethnic_relative.append(asthmah/asthmanh)
    fig = plt.figure(figsize=(8.5,11))    
    ax1 = plt.subplot2grid((4,2),(0,0))
    ax1t = ax1.twinx()
    ax2 = plt.subplot2grid((4,2),(0,1))
    ax2t = ax2.twinx()
    ax3 = plt.subplot2grid((4,2),(1,0))
    ax3t = ax3.twinx()
    ax4 = plt.subplot2grid((4,2),(1,1))
    ax4t = ax4.twinx()
    ax5 = plt.subplot2grid((4,2),(2,0))
    ax5t = ax5.twinx()
    ax6 = plt.subplot2grid((4,2),(2,1))
    ax6t = ax6.twinx()
    ax7 = plt.subplot2grid((4,2),(3,0))
    ax7t = ax7.twinx()
    ax8 = plt.subplot2grid((4,2),(3,1))
    ax8t = ax8.twinx()
    years = np.arange(2010,2020,1)
    # Racial PM2.5 disparities
    ax1t.bar(years, PM25_race_relative, color='grey', zorder=0)
    ax1.plot(years, PM25_white, lw=2, color=color3, zorder=10)
    ax1.plot(years, PM25_black, lw=2, color=color2, zorder=10)
    # Ethnic PM2.5 disparities
    ax2t.bar(years, PM25_ethnic_relative, color='grey', zorder=0)
    ax2.plot(years, PM25_nonhisp, lw=2, color=color3, zorder=10)
    ax2.plot(years, PM25_hisp, lw=2, color=color2, zorder=10)
    # Racial NO2 disparities
    ax3t.bar(years, NO2_race_relative, color='grey', zorder=0)
    ax3.plot(years, NO2_white, lw=2, color=color3, zorder=10)
    ax3.plot(years, NO2_black, lw=2, color=color2, zorder=10)
    # Ethnic NO2 disparities
    ax4t.bar(years, NO2_ethnic_relative, color='grey', zorder=0)
    ax4.plot(years, NO2_nonhisp, lw=2, color=color3, zorder=10)
    ax4.plot(years, NO2_hisp, lw=2, color=color2, zorder=10)
    # Racial premature death disparities
    ax5t.bar(years, pd_race_relative, color='grey', zorder=0)
    ax5.plot(years, pd_white, lw=2, color=color3, zorder=10)
    ax5.plot(years, pd_black, lw=2, color=color2, zorder=10)
    # Ethnic premature death disparities
    ax6t.bar(years, pd_ethnic_relative, color='grey', zorder=0)
    ax6.plot(years, pd_nonhisp, lw=2, color=color3, zorder=10)
    ax6.plot(years, pd_hisp, lw=2, color=color2, zorder=10)
    # Racial asthma disparities
    ax7t.bar(years, asthma_race_relative, color='grey', zorder=0)
    ax7.plot(years, asthma_white, lw=2, color=color3, zorder=10)
    ax7.plot(years, asthma_black, lw=2, color=color2, zorder=10)
    # Ethnic asthma disparities
    ax8t.bar(years, asthma_ethnic_relative, color='grey', zorder=0)
    ax8.plot(years, asthma_nonhisp, lw=2, color=color3, zorder=10)
    ax8.plot(years, asthma_hisp, lw=2, color=color2, zorder=10)
    # Aesthetics    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:    
        ax.set_xlim([2010,2019])
        ax.set_xticks(np.arange(2010,2020,1))
        ax.set_xticklabels([])
    # PM25
    for ax in [ax1, ax2]:
        ax.set_ylim([0, 12])
        ax.set_yticks(np.linspace(0, 12, 5))
    for ax in [ax1t, ax2t]:
        ax.set_ylim([0.9,1.3])
    # NO2
    for ax in [ax3, ax4]:
        ax.set_ylim([0, 20])
        ax.set_yticks(np.linspace(0, 20, 5))
    for ax in [ax3t, ax4t]:
        ax.set_ylim([1.0, 2.5])
    # PM2.5-attributable mortality 
    for ax in [ax5, ax6]:
        ax.set_ylim([0, 28])
        ax.set_yticks(np.linspace(0, 28, 5))
    for ax in [ax5t, ax6t]:
        ax.set_ylim([0.6, 1.2])
        # ax.set_yticks(np.linspace(0.5, 1.1, 5))    
    # NO2-attributable asthma
    for ax in [ax7, ax8]:
        ax.set_xticklabels(['2010', '', '', '2013', '', '', '2016', '', '', 
            '2019'])
        ax.set_ylim([0, 450])
        ax.set_yticks(np.linspace(0, 500, 6))
    for ax in [ax7t, ax8t]:
        ax.set_ylim([1, 8])
    # Change default zorder     
    for ax, axt in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8], 
        [ax1t, ax2t, ax3t, ax4t, ax5t, ax6t, ax7t, ax8t]):   
        ax.set_zorder(axt.get_zorder()+1)
        ax.patch.set_visible(False)
    # Y-axis labels for primary axes
    ax1.set_ylabel('PM$_{\mathregular{2.5}}$\n[$\mathregular{\mu}$g'+\
        ' m$^{\mathregular{-3}}$]', loc='bottom', fontsize=14)
    ax1.yaxis.set_label_coords(-0.18,0.0)
    ax2.set_ylabel('')
    ax2.set_yticklabels([])
    ax3.set_ylabel('NO$_{\mathregular{2}}$\n[ppbv]', loc='bottom', fontsize=14)
    ax3.yaxis.set_label_coords(-0.18,0.0)
    ax4.set_ylabel('')
    ax4.set_yticklabels([])    
    ax5.set_ylabel('Premature deaths due \nto PM$_\mathregular{2.5}$ '+\
        '[per 100000]', loc='bottom', fontsize=14)
    ax5.yaxis.set_label_coords(-0.18,0.0)
    ax6.set_ylabel('')
    ax6.set_yticklabels([])    
    ax7.set_ylabel('New asthma cases due \nto NO$_\mathregular{2}$ '+\
        '[per 100000]', loc='bottom', fontsize=14)    
    ax7.yaxis.set_label_coords(-0.18,0.0)
    ax8.set_ylabel('')
    ax8.set_yticklabels([])    
    # Y-axis labels for secondary axes
    ax1t.set_ylabel('')
    ax1t.set_yticklabels([])
    ax2t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax2t.yaxis.set_label_coords(1.25,0.5)
    ax3t.set_ylabel('')
    ax3t.set_yticklabels([])
    ax4t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax4t.yaxis.set_label_coords(1.25,0.5)
    ax5t.set_ylabel('')
    ax5t.set_yticklabels([])
    ax6t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax6t.yaxis.set_label_coords(1.25,0.5)
    ax7t.set_ylabel('')
    ax7t.set_yticklabels([])
    ax8t.set_ylabel('Relative disparities [$\mathregular{\cdot}$]', 
        rotation=270, fontsize=14)
    ax8t.yaxis.set_label_coords(1.25,0.5)
    # Generate legend 
    patch2 = mpatches.Patch(color=color2, label='Black')
    patch1 = mpatches.Patch(color=color3, label='White')
    all_handles = (patch2, patch1)
    ax7.legend(handles=all_handles, loc=8, frameon=False, ncol=2,
        bbox_to_anchor=(0.5, -0.5), fontsize=14)
    patch1 = mpatches.Patch(color=color2, label='Hispanic')
    patch2 = mpatches.Patch(color=color3, label='Non-Hispanic')
    all_handles = (patch1, patch2)
    ax8.legend(handles=all_handles, loc=8, frameon=False, ncol=2,
        bbox_to_anchor=(0.5, -0.5), fontsize=14)
    # Subplot labels
    ax1.set_title('(A) Racial disparities', fontsize=14, loc='left')
    ax2.set_title('(B) Ethnic disparities', fontsize=14, loc='left')
    ax3.set_title('(C)', fontsize=14, loc='left')
    ax4.set_title('(D)', fontsize=14, loc='left')
    ax5.set_title('(E)', fontsize=14, loc='left')
    ax6.set_title('(F)', fontsize=14, loc='left')
    ax7.set_title('(G)', fontsize=14, loc='left')
    ax8.set_title('(H)', fontsize=14, loc='left')
    plt.subplots_adjust(wspace=0.2, top=0.95, bottom=0.1, hspace=0.3)
    plt.savefig(DIR_FIG+'figS7.png', dpi=600)
    return

def figS8(burdents):
    """
    Parameters
    ----------
    burdents : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import matplotlib.patches as mpatches
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp    
    # Select burdens from beginning, midpoint, and final years of analysis
    # (2010, 2015, 2019) 
    burden10 = burdents.loc[burdents['YEAR']=='2006-2010'].copy(deep=True)
    burden15 = burdents.loc[burdents['YEAR']=='2011-2015'].copy(deep=True)
    burden19 = burdents.loc[burdents['YEAR']=='2015-2019'].copy(deep=True)
    # Calculate racial characteristics
    fracnh10 = ((burden10[['race_nh']].sum(axis=1))/burden10['race_tot'])
    burden10['fracnh10'] = fracnh10
    fracnh15 = ((burden15[['race_nh']].sum(axis=1))/burden15['race_tot'])
    burden15['fracnh15'] = fracnh15
    fracnh19 = ((burden19[['race_nh']].sum(axis=1))/burden19['race_tot'])
    burden19['fracnh19'] = fracnh19
    # Find tracts with largest NO2-, PM25-, or PM25- and NO2-attributable 
    # health burdens
    # For 2010
    whereleast10 = burden10.loc[
        (burden10.BURDENASTHMARATE<=np.nanpercentile(burden10.BURDENASTHMARATE,10)) & 
        (burden10.BURDENPMALLRATE<=np.nanpercentile(burden10.BURDENPMALLRATE,10))]
    wheremost10 = burden10.loc[
        (burden10.BURDENASTHMARATE>=np.nanpercentile(burden10.BURDENASTHMARATE,90)) & 
        (burden10.BURDENPMALLRATE>=np.nanpercentile(burden10.BURDENPMALLRATE,90))]
    wheremostpm10 = burden10.loc[burden10.BURDENPMALLRATE>=
        np.nanpercentile(burden10.BURDENPMALLRATE,90)]
    wheremostasthma10 = burden10.loc[burden10.BURDENASTHMARATE>=
        np.nanpercentile(burden10.BURDENASTHMARATE,90)]
    whereleastpm10 = burden10.loc[burden10.BURDENPMALLRATE<=
        np.nanpercentile(burden10.BURDENPMALLRATE,10)]
    whereleastasthma10 = burden10.loc[burden10.BURDENASTHMARATE<=
        np.nanpercentile(burden10.BURDENASTHMARATE,10)]
    # For 2015
    whereleast15 = burden15.loc[
        (burden15.BURDENASTHMARATE<=np.nanpercentile(burden15.BURDENASTHMARATE,10)) & 
        (burden15.BURDENPMALLRATE<=np.nanpercentile(burden15.BURDENPMALLRATE,10))]
    wheremost15 = burden15.loc[
        (burden15.BURDENASTHMARATE>=np.nanpercentile(burden15.BURDENASTHMARATE,90)) & 
        (burden15.BURDENPMALLRATE>=np.nanpercentile(burden15.BURDENPMALLRATE,90))]
    wheremostpm15 = burden15.loc[burden15.BURDENPMALLRATE>=
        np.nanpercentile(burden15.BURDENPMALLRATE,90)]
    wheremostasthma15 = burden15.loc[burden15.BURDENASTHMARATE>=
        np.nanpercentile(burden15.BURDENASTHMARATE,90)]
    whereleastpm15 = burden15.loc[burden15.BURDENPMALLRATE<=
        np.nanpercentile(burden15.BURDENPMALLRATE,10)]
    whereleastasthma15 = burden15.loc[burden15.BURDENASTHMARATE<=
        np.nanpercentile(burden15.BURDENASTHMARATE,10)]
    # For 2019 
    whereleast19 = burden19.loc[
        (burden19.BURDENASTHMARATE<=np.nanpercentile(burden19.BURDENASTHMARATE,10)) & 
        (burden19.BURDENPMALLRATE<=np.nanpercentile(burden19.BURDENPMALLRATE,10))]
    wheremost19 = burden19.loc[
        (burden19.BURDENASTHMARATE>=np.nanpercentile(burden19.BURDENASTHMARATE,90)) & 
        (burden19.BURDENPMALLRATE>=np.nanpercentile(burden19.BURDENPMALLRATE,90))]
    wheremostpm19 = burden19.loc[burden19.BURDENPMALLRATE>=
        np.nanpercentile(burden19.BURDENPMALLRATE,90)]
    wheremostasthma19 = burden19.loc[burden19.BURDENASTHMARATE>=
        np.nanpercentile(burden19.BURDENASTHMARATE,90)]
    whereleastpm19 = burden19.loc[burden19.BURDENPMALLRATE<=
        np.nanpercentile(burden19.BURDENPMALLRATE,10)]
    whereleastasthma19 = burden19.loc[burden19.BURDENASTHMARATE<=
        np.nanpercentile(burden19.BURDENASTHMARATE,10)]
    colorsyear = [color1, color2, color3]
    # Plotting
    fig = plt.figure(figsize=(8,5))
    ax1 = plt.subplot2grid((1,1), (0,0))
    # Denote median percent white in years 
    ax1.axhline(burden10.fracnh10.median(), ls='-', color=colorsyear[0], 
        zorder=0, lw=0.75)
    ax1.axhline(burden15.fracnh15.median(), ls='-', color=colorsyear[1], 
        zorder=0, lw=0.75)
    ax1.axhline(burden19.fracnh19.median(), ls='-', color=colorsyear[2], 
        zorder=0, lw=0.75)
    # Racial composition of tracts with the smallest PM2.5-
    # attributable burdens
    heights = [whereleastpm10.fracnh10, whereleastpm15.fracnh15, 
        whereleastpm19.fracnh19]
    pos = [0,1,2]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance 
    ks_1015 = ks_2samp(whereleastpm10.fracnh10, whereleastpm15.fracnh15)
    ks_1519 = ks_2samp(whereleastpm15.fracnh15, whereleastpm19.fracnh19)
    ks_1019 = ks_2samp(whereleastpm10.fracnh10, whereleastpm19.fracnh19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the smallest NO2-attributable 
    # burdens
    heights = [whereleastasthma10.fracnh10, whereleastasthma15.fracnh15, 
        whereleastasthma19.fracnh19]
    pos = [4,5,6]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'),
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(whereleastasthma10.fracnh10, 
        whereleastasthma15.fracnh15)
    ks_1519 = ks_2samp(whereleastasthma15.fracnh15, 
        whereleastasthma19.fracnh19)
    ks_1019 = ks_2samp(whereleastasthma10.fracnh10, 
        whereleastasthma19.fracnh19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the smallest NO2- and PM2.5-
    # attributable burdens
    heights = [whereleast10.fracnh10, whereleast15.fracnh15, 
        whereleast19.fracnh19]
    pos = [8,9,10]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'), 
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(whereleast10.fracnh10, whereleast15.fracnh15)
    ks_1519 = ks_2samp(whereleast15.fracnh15, whereleast19.fracnh19)
    ks_1019 = ks_2samp(whereleast10.fracnh10, whereleast19.fracnh19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the largest PM2.5-attributable burdens    
    heights = [wheremostpm10.fracnh10, wheremostpm15.fracnh15, 
        wheremostpm19.fracnh19]
    pos = [13,14,15]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'),
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(wheremostpm10.fracnh10, wheremostpm15.fracnh15)
    ks_1519 = ks_2samp(wheremostpm15.fracnh15, wheremostpm19.fracnh19)
    ks_1019 = ks_2samp(wheremostpm10.fracnh10, wheremostpm19.fracnh19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.07, maxasterix=(3))
    # Racial composition of tracts with the largest NO2-attributable burdens    
    heights = [wheremostasthma10.fracnh10, 
        wheremostasthma15.fracnh15, wheremostasthma19.fracnh19]
    pos = [17,18,19]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'),
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(wheremostasthma10.fracnh10, 
        wheremostasthma15.fracnh15)
    ks_1519 = ks_2samp(wheremostasthma15.fracnh15, 
        wheremostasthma19.fracnh19)
    ks_1019 = ks_2samp(wheremostasthma10.fracnh10, 
        wheremostasthma19.fracnh19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, ks_1019.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.08, maxasterix=(3))
    # Racial composition of tracts with the largest NO2- and PM2.5-attributable burdens    
    heights = [wheremost10.fracnh10, wheremost15.fracnh15, 
        wheremost19.fracnh19]
    pos = [21,22,23]
    box = ax1.boxplot(heights, positions=pos, whis=0, showfliers=False, 
        widths=0.6, patch_artist=True, medianprops=dict(color='w'),
        capprops=dict(color='None'), whiskerprops=dict(color='None'))
    for patch, color in zip(box['boxes'], colorsyear):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    # Calculate significance
    ks_1015 = ks_2samp(wheremost10.fracnh10, wheremost15.fracnh15)
    ks_1519 = ks_2samp(wheremost15.fracnh15, wheremost19.fracnh19)
    ks_1019 = ks_2samp(wheremost10.fracnh10, wheremost19.fracnh19)
    barplot_annotate_brackets(0, 1, ks_1015.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(1, 2, ks_1519.pvalue, pos, 
        [np.nanpercentile(x, 25) for x in heights], maxasterix=(3))
    barplot_annotate_brackets(0, 2, np.round(ks_1019.pvalue,2), pos, 
        [np.nanpercentile(x, 25) for x in heights], dh=.11, maxasterix=(3))
    # Aesthetics
    ax1.set_xlim([-0.75, 23.75])
    ax1.set_xticks([1,5,9,14,18,22])
    ax1.set_xticklabels([
        'PM$_\mathregular{2.5}$-\nattributable',
        'NO$_\mathregular{2}$-\nattributable',
        'NO$_\mathregular{2}$- and PM$_{\mathregular{2.5}}$-\nattributable',
        'PM$_\mathregular{2.5}$-\nattributable',
        'NO$_\mathregular{2}$-\nattributable',    
        'NO$_\mathregular{2}$- and PM$_{\mathregular{2.5}}$-\nattributable'])
    ax1.tick_params(axis=u'x', which=u'both',length=0)
    ax1.set_ylim([0,1])
    ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.])
    ax1.set_yticklabels(['0','20','40','60','80','100'])
    ax1.set_ylabel('Proportion of non-Hispanic population [%]', fontsize=14)
    # Hide the right and top spines
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # Create legend
    ax1.text(5, 1.05,'L e a s t    b u r d e n e d', fontsize=18, ha='center')
    ax1.text(18, 1.05,'M o s t    b u r d e n e d', fontsize=18, ha='center')
    patch1 = mpatches.Patch(color=colorsyear[0], label='2010')
    patch2 = mpatches.Patch(color=colorsyear[1], label='2015')
    patch3 = mpatches.Patch(color=colorsyear[2], label='2019')
    all_handles = (patch1, patch2, patch3)
    ax1.legend(handles=all_handles, frameon=False, ncol=1,
        bbox_to_anchor=(0.2,0.25))
    plt.savefig(DIR_FIG+'figS8.pdf', dpi=500)
    # Calculate some pithy statistics
    print('Median non-Hispanic population [%]')
    print('2010 = %.2f'%(burden10.fracnh10.median()*100.))
    print('2015 = %.2f'%(burden15.fracnh15.median()*100.))
    print('2019 = %.2f'%(burden19.fracnh19.median()*100.))
    print('\n')
    print('White population in tracts with smallest '+\
        'PM2.5-attributable burdens [%]')
    print('2010 = %.2f'%(whereleastpm10.fracnh10.median()*100.))
    print('2015 = %.2f'%(whereleastpm15.fracnh15.median()*100.))
    print('2019 = %.2f'%(whereleastpm19.fracnh19.median()*100.)) 
    print('White population in tracts with largest '+\
        'PM2.5-attributable burdens [%]')
    print('2010 = %.2f'%(wheremostpm10.fracnh10.median()*100.))
    print('2015 = %.2f'%(wheremostpm15.fracnh15.median()*100.))
    print('2019 = %.2f'%(wheremostpm19.fracnh19.median()*100.))    
    print('\n')
    print('White population in tracts with smallest '+\
        'NO2-attributable burdens [%]')
    print('2010 = %.2f'%(whereleastasthma10.fracnh10.median()*100.))
    print('2015 = %.2f'%(whereleastasthma15.fracnh15.median()*100.))
    print('2019 = %.2f'%(whereleastasthma19.fracnh19.median()*100.))
    print('White population in tracts with largest '+\
        'NO2-attributable burdens [%]')
    print('2010 = %.2f'%(wheremostasthma10.fracnh10.median()*100.))
    print('2015 = %.2f'%(wheremostasthma15.fracnh15.median()*100.))
    print('2019 = %.2f'%(wheremostasthma19.fracnh19.median()*100.))
    print('\n')
    print('White population in tracts with smallest '+\
        'PM2.5- AND NO2-attributable burdens [%]')
    print('2010 = %.2f'%(whereleast10.fracnh10.median()*100.), 
        '(%d tracts)'%whereleast10.shape[0])
    print('2015 = %.2f'%(whereleast15.fracnh15.median()*100.), 
        '(%d tracts)'%whereleast15.shape[0])
    print('2019 = %.2f'%(whereleast19.fracnh19.median()*100.), 
        '(%d tracts)'%whereleast19.shape[0]) 
    print('White population in tracts with largest '+\
        'PM2.5- AND NO2-attributable burdens [%]')
    print('2010 = %.2f'%(wheremost10.fracnh10.median()*100.), 
        '(%d tracts)'%wheremost10.shape[0])
    print('2015 = %.2f'%(wheremost15.fracnh15.median()*100.), 
        '(%d tracts)'%wheremost15.shape[0])
    print('2019 = %.2f'%(wheremost19.fracnh19.median()*100.), 
        '(%d tracts)'%wheremost19.shape[0])
    return

import pandas as pd
import math
import time
from datetime import datetime
import numpy as np   
from scipy import stats
import sys
sys.path.append(DIR)
import edf_open, edf_calculate
sys.path.append('/Users/ghkerr/GW/tropomi_ej/')
import tropomi_census_utils
sys.path.append('/Users/ghkerr/GW/edf/')
import pm25no2_constants

# Load crosswalk to enable subsampling of MSAs
crosswalk = pd.read_csv(DIR_CROSS+'qcew-county-msa-csa-crosswalk.csv', 
    engine='python', encoding='latin1')
# Add a leading zero to FIPS codes 0-9
crosswalk['County Code'] = crosswalk['County Code'].map(lambda x:
    f'{x:0>5}')
# Open 2010-2019 harmonized tables and calculate burdens
harmts = []
burdents = []
for year in np.arange(2010, 2020, 1):
    print(year)
    vintage = '%d-%d'%(year-4, year)
    harm = edf_open.load_vintageharmonized(vintage)
    burden = edf_calculate.calculate_pm25no2burden(harm)
    # Total PM2.5-attributable deaths and new cases of NO2-attributable asthma
    print('sum(Stroke) = %d'%round(burden.BURDENST.sum()))
    print('sum(COPD) = %d'%round(burden.BURDENCOPD.sum()))
    print('sum(Lung cancer) = %d'%round(burden.BURDENLC.sum()))
    print('sum(Type 2 diabetes) = %d'%round(burden.BURDENDM.sum()))
    print('sum(Total IHD) = %d'%round(burden.BURDENIHD.sum()))
    print('sum(Lower respiratory infection) = %d'%round(burden.BURDENLRI.sum()))
    print('sum(Pediatric asthma) = %d'%round(burden.BURDENASTHMA.sum()))
    print('Total PM deaths = %d'%round(burden.BURDENST.sum()+
        burden.BURDENCOPD.sum()+burden.BURDENLC.sum()+burden.BURDENDM.sum()+
        burden.BURDENIHD.sum()+burden.BURDENLRI.sum()))
    print('\n')
    harmts.append(harm)
    burdents.append(burden)
harmts = pd.concat(harmts)
burdents = pd.concat(burdents)

# # # # Subset harmonized tables in MSAs
harm_msa, burden_msa = [], []
geoids = harm.index.values
for i, msa in enumerate(pm25no2_constants.majors):
    crosswalk_msa = crosswalk.loc[crosswalk['MSA Title']==msa]
    # Umlaut and accent screw with .loc
    if msa=='Mayagüez, PR':
        crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C3242']
    if msa=='San Germán, PR':
        crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C4190']
    # Find GEOIDs in MSA
    geoids_msa = []
    for prefix in crosswalk_msa['County Code'].values: 
        prefix = str(prefix).zfill(5)
        incounty = [x for x in geoids if x.startswith(prefix)]
        geoids_msa.append(incounty)
    geoids_msa = sum(geoids_msa, [])
    harm_imsa = harmts.loc[harmts.index.isin(geoids_msa)]
    harm_msa.append(harm_imsa)
    burden_imsa = burdents.loc[burdents.index.isin(geoids_msa)]
    burden_msa.append(burden_imsa)
harm_msa = pd.concat(harm_msa)
burden_msa = pd.concat(burden_msa)

pmburden_allmsa, asthmaburden_allmsa = [], []
pmburdenrate_allmsa, asthmaburdenrate_allmsa = [], []
lng_allmsa, lat_allmsa, name_allmsa = [], [], []
burden19 = burdents.loc[burdents.YEAR=='2015-2019']
geoids = burden19.index.values
# Loop through MSAs in U.S. 
for i, msa in enumerate(pm25no2_constants.majors):
    crosswalk_msa = crosswalk.loc[crosswalk['MSA Title']==msa]
    # Umlaut and accent screw with .loc
    if msa=='Mayagüez, PR':
        crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C3242']
    if msa=='San Germán, PR':
        crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C4190']
    # Find GEOIDs in MSA
    geoids_msa = []
    for prefix in crosswalk_msa['County Code'].values: 
        prefix = str(prefix).zfill(5)
        incounty = [x for x in geoids if x.startswith(prefix)]
        geoids_msa.append(incounty)
    geoids_msa = sum(geoids_msa, [])
    # Select NO2-attributable burdens for most recent year available (2019)
    burden_imsa = burden19.loc[burden19.index.isin(geoids_msa)]
    asthmaburdenrate_allmsa.append(burden_imsa['BURDENASTHMARATE'].mean())
    asthmaburden_allmsa.append(burden_imsa['BURDENASTHMA'].sum())
    pmburdenrate_allmsa.append(burden_imsa['BURDENPMALLRATE'].mean())
    pmburden_allmsa.append(burden_imsa['BURDENPMALL'].sum())
    lng_allmsa.append(burden_imsa['LNG_CENTROID'].mean())
    lat_allmsa.append(burden_imsa['LAT_CENTROID'].mean())
    name_allmsa.append(msa)
asthmaburdenrate_allmsa = np.array(asthmaburdenrate_allmsa)
pmburdenrate_allmsa = np.array(pmburdenrate_allmsa)


# fig1(burdents, pmburdenrate_allmsa, asthmaburdenrate_allmsa, lng_allmsa, 
#     lat_allmsa)
# fig2(burdents)
# fig3(burdents)
# table1(burdents)


# figS1(burdents)
# figS2()
# figS3()
# figS4()
# Figure S5 is the population-weighted version of Figure 2
# figS6(burdents)
# figS7(burdents)
# figS8(burdents)


# burden19 = burdents.loc[burdents['YEAR']=='2015-2019']
# frac_hisp19 = (burden19['race_h']/burden19['race_tot'])
# frac_white19 = ((burden19[['race_nh_white','race_h_white']].sum(axis=1))/
#     burden19['race_tot'])
# mosthisp = burden19.iloc[np.where(frac_hisp19 >= 
#     np.nanpercentile(frac_hisp19, 90))]
# leasthisp = burden19.iloc[np.where(frac_hisp19 <= 
#     np.nanpercentile(frac_hisp19, 10))]
# mostwhite = burden19.iloc[np.where(frac_white19 >= 
#     np.nanpercentile(frac_white19, 90))]
# leastwhite = burden19.iloc[np.where(frac_white19 <= 
#     np.nanpercentile(frac_white19, 10))]
# mwp = mosthisp[['pop_m_lt5', 'pop_m_5-9', 'pop_m_10-14',
# 'pop_m_15-17', 'pop_m_18-19', 'pop_m_20', 'pop_m_21', 'pop_m_22-24', 'pop_f_lt5',
# 'pop_f_5-9', 'pop_f_10-14', 'pop_f_15-17', 'pop_f_18-19', 'pop_f_20', 'pop_f_21', 'pop_f_22-24']].sum(axis=1)/mosthisp['pop_tot']
# lwp = leasthisp[['pop_m_lt5', 'pop_m_5-9', 'pop_m_10-14',
# 'pop_m_15-17', 'pop_m_18-19', 'pop_m_20', 'pop_m_21', 'pop_m_22-24', 'pop_f_lt5',
# 'pop_f_5-9', 'pop_f_10-14', 'pop_f_15-17', 'pop_f_18-19', 'pop_f_20', 'pop_f_21', 'pop_f_22-24']].sum(axis=1)/leasthisp['pop_tot'] 

# areas = []
# msa = harm_msa.index.unique()
# import shapefile
# from shapely.geometry import shape, Point
# fips = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', 
#     '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
#     '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
#     '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', 
#     '48', '49', '50', '51', '53', '54', '55', '56', '72']
# for statefips in fips:
#     print(fips)        
#     fname = DIR_GEO+'tract_2010/tl_2019_%s_tract/'%statefips
#     r = shapefile.Reader(fname+'tl_2019_%s_tract.shp'%statefips)
#     # Get shapes, records
#     tracts = r.shapes()
#     records = r.records()
#     for tract in np.arange(0, len(tracts), 1):
#         record = records[tract]  
#         # In older iterations of this script, I employed a 0.75˚ search radius
#         # to find intersecting netCDF grid cells within the tract. This 
#         # approach might not be perfect for very large tracts, such as those 
#         # in the Western U.S.; however, it is very slow. The new approach 
#         # defines a different search radius for each tract based on its total 
#         # area (in meters squared) of land and water
#         area = (record['ALAND']+record['AWATER'])/(1000*1000)
#         geoid = record['GEOID']  
#         # if geoid in msa:
#         areas.append(area)
# # Load crosswalk to enable subsampling of MSAs
# crosswalk = pd.read_csv(DIR_CROSS+'qcew-county-msa-csa-crosswalk.csv', 
#     engine='python', encoding='latin1')
# # Add a leading zero to FIPS codes 0-9
# crosswalk['County Code'] = crosswalk['County Code'].map(lambda x:
#     f'{x:0>5}')
# geoids = harm.index.values
# # Subset harmonized tables in urban versus rural areas
# harm_urban, geoids_urban = [], []
# geoids = burden19.index.values
# for i, msa in enumerate(pm25no2_constants.majors):
#     crosswalk_msa = crosswalk.loc[crosswalk['MSA Title']==msa]
#     # Umlaut and accent screw with .loc
#     if msa=='Mayagüez, PR':
#         crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C3242']
#     if msa=='San Germán, PR':
#         crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C4190']
#     # Find GEOIDs in MSA
#     geoids_msa = []
#     for prefix in crosswalk_msa['County Code'].values: 
#         prefix = str(prefix).zfill(5)
#         incounty = [x for x in geoids if x.startswith(prefix)]
#         geoids_msa.append(incounty)
#     geoids_msa = sum(geoids_msa, [])
#     harm_iurban = burden19.loc[burden19.index.isin(geoids_msa)]
#     geoids_urban.append(geoids_msa)
#     harm_urban.append(harm_iurban)
# harm_urban = pd.concat(harm_urban)
# geoids_urban = np.hstack(geoids_urban)
# harm_rural = harm.loc[~harm.index.isin(geoids_urban)]




