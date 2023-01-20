#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:43:39 2021

@author: ghkerr
"""
import sys
DIR = '/Users/ghkerr/GW/edf/'
DIR_HARM = '/Users/ghkerr/GW/data/anenberg_mohegh_no2/harmonizedtables/'
#DIR_CENSUS = DIR_ROOT+'acs/'
DIR_CROSS = '/Users/ghkerr/GW/data/geography/'
DIR_GEO = DIR_CROSS+'tigerline/'
DIR_AQ = '/Users/ghkerr/GW/data/aq/aqs/'
#DIR_OUT = DIR_ROOT+'harmonizedtables/'
DIR_TYPEFACE = '/Users/ghkerr/Library/Fonts/'
DIR_GBD = '/Users/ghkerr/GW/data/gbd/'
DIR_FIG = '/Users/ghkerr/GW/edf/figs/'

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
    if text == 'ns':
        mid = ((lx+rx)/2, (y)-barh-0.015)
    plt.plot(barx, bary, c='black', lw=0.75)
    kwargs = dict(ha='center', va='center')
    if fs is not None:
        kwargs['fontsize'] = fs
    if text == 'ns':
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
    pollutant_aqs['TROPOMI_VAL'] = np.nan    
    # Find the tract code corresponding to each AQS monitor measurement; code
    # adapted from https://stackoverflow.com/questions/68665273/
    # mapping-gps-coordinates-with-census-tract-python
    for index, row in pollutant_aqs.iterrows():
        print(index, row['Parameter Name'], row['Year'])
        # Avoid monitors across the border
        if (row['State Name'] != 'Country Of Mexico') and \
                (row['State Name'] != 'Virgin Islands'):
            try:
                block = requests.get(
                    url, params={"latitude": row["Latitude"],
                                 "longitude": row["Longitude"], "format": "json",
                                 "censusYear": 2010}).json(
                )["Block"]["FIPS"]
                pollutant_aqs.loc[index, 'GEOID'] = block[:11]
                # try:
                pollutant_aqs.loc[index, 'DATASET_VAL'] = \
                    harmty.loc[block[:11]][pollutant]
                pollutant_aqs.loc[index, 'TROPOMI_VAL'] = \
                    harmty.loc[block[:11]]['TROPOMINO2']                    
                # except KeyError:
                # pollutant_aqs.loc[index,'DATASET_VAL'] = np.nan
            # Look up tract-average
            except ValueError:
                print('No code for %.2f, %.2f' % (row['Latitude'],
                                                  row['Longitude']))
                pass
    # Save output file
    pollutant_aqs.to_csv(DIR_AQ+'%s_%d_bymonitor_harmonized.csv' % (
        pollutant, year), sep=',', encoding='utf-8')
    return

def agestandardize(refpop, subgroup, endpoint):
    """Calculate the age standardized incidence or death rates for a given 
    population subgroup. 

    Parameters
    ----------
    refpop : pandas.core.frame.DataFrame
        The reference population (i.e., nationwide tract-level ACS data) which
        will be used to calculate the age standardization. 
    subgroup : pandas.core.frame.DataFrame
        Dataframe with burdens for a particular year and demographic subgroup.
    endpoint : str
        The abbreviation of the endpoint, which should match the column naming
        convention in the burden tables (e.g., PA, IHD, ST, LC, COPD, DM, LRI).

    Returns
    -------
    subgroup_as : float
        The age standardized incidence or death rates for the population 
        subgroup of interest
    """
    # Age standardization for asthma burden rate
    refpop = refpop.copy(deep=True)
    if 'PA' in endpoint:
        # If intended to standardize by age, simply multiply the results 
        # by a common age distribution (i.e., full age distribution for the 
        # U.S. for that year). Reference: https://sphweb.bumc.bu.edu/otlt/
        # mph-modules/ep/ep713_standardizedrates/
        # ep713_standardizedrates_print.html
        pop = ['pop_m_lt5','pop_f_lt5','pop_m_5-9','pop_f_5-9', 
            'pop_m_10-14','pop_f_10-14','pop_m_15-17','pop_m_18-19',
            'pop_f_15-17','pop_f_18-19']
        refpop['FRAC_LT5'] = (refpop[['pop_m_lt5','pop_f_lt5']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_5-9'] = (refpop[['pop_m_5-9','pop_f_5-9']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_10-14'] = (refpop[['pop_m_10-14','pop_f_10-14']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_15-19'] = (refpop[['pop_m_15-17','pop_m_18-19',
            'pop_f_15-17','pop_f_18-19']].sum(axis=1)/refpop[pop].sum(axis=1))        
        # Calculate age standardized rate based on fraction of each age 
        # group to total population 
        subgroup_as = ((subgroup['BURDEN'+endpoint+'_LT5RATE'].mean()*
              refpop['FRAC_LT5'].mean())+
            (subgroup['BURDEN'+endpoint+'_5RATE'].mean()*
              refpop['FRAC_5-9'].mean())+
            (subgroup['BURDEN'+endpoint+'_10RATE'].mean()*
              refpop['FRAC_10-14'].mean())+
            (subgroup['BURDEN'+endpoint+'_15RATE'].mean()*
              refpop['FRAC_15-19'].mean()))
    elif 'AC' in endpoint: 
        pop = ['pop_m_30-34', 'pop_m_35-39', 'pop_m_40-44', 'pop_m_45-49', 
            'pop_m_50-54', 'pop_m_55-59', 'pop_m_60-61', 'pop_m_62-64', 
            'pop_m_65-66', 'pop_m_67-69', 'pop_m_70-74', 'pop_m_75-79', 
            'pop_m_80-84', 'pop_m_gt85', 'pop_f_30-34', 'pop_f_35-39', 
            'pop_f_40-44', 'pop_f_45-49', 'pop_f_50-54', 'pop_f_55-59', 
            'pop_f_60-61', 'pop_f_62-64', 'pop_f_65-66', 'pop_f_67-69', 
            'pop_f_70-74', 'pop_f_75-79', 'pop_f_80-84', 'pop_f_gt85']    
        refpop['FRAC_30-34'] = (refpop[['pop_m_30-34','pop_f_30-34']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_35-39'] = (refpop[['pop_m_35-39','pop_f_35-39']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_40-44'] = (refpop[['pop_m_40-44','pop_f_40-44']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_45-49'] = (refpop[['pop_m_45-49','pop_f_45-49']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_50-54'] = (refpop[['pop_m_50-54','pop_f_50-54']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_55-59'] = (refpop[['pop_m_55-59','pop_f_55-59']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_60-64'] = (refpop[['pop_m_60-61','pop_m_62-64',
            'pop_f_60-61','pop_f_62-64']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_65-69'] = (refpop[['pop_m_65-66','pop_m_67-69',
            'pop_f_65-66','pop_f_67-69']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_70-74'] = (refpop[['pop_m_70-74','pop_f_70-74']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_75-79'] = (refpop[['pop_m_75-79','pop_f_75-79']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_80-84'] = (refpop[['pop_m_80-84','pop_f_80-84']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_GT85'] = (refpop[['pop_m_gt85','pop_f_gt85']
            ].sum(axis=1)/refpop[pop].sum(axis=1))        
        # Weight
        subgroup_as = ((subgroup['BURDEN'+endpoint+'_30RATE'].mean()*
              refpop['FRAC_30-34'].mean())+
            (subgroup['BURDEN'+endpoint+'_35RATE'].mean()*
              refpop['FRAC_35-39'].mean())+
            (subgroup['BURDEN'+endpoint+'_40RATE'].mean()*
              refpop['FRAC_40-44'].mean())+
            (subgroup['BURDEN'+endpoint+'_45RATE'].mean()*
              refpop['FRAC_45-49'].mean())+
            (subgroup['BURDEN'+endpoint+'_50RATE'].mean()*
              refpop['FRAC_50-54'].mean())+
            (subgroup['BURDEN'+endpoint+'_55RATE'].mean()*
              refpop['FRAC_55-59'].mean())+
            (subgroup['BURDEN'+endpoint+'_60RATE'].mean()*
              refpop['FRAC_60-64'].mean())+
            (subgroup['BURDEN'+endpoint+'_65RATE'].mean()*
              refpop['FRAC_65-69'].mean())+
            (subgroup['BURDEN'+endpoint+'_70RATE'].mean()*
              refpop['FRAC_70-74'].mean())+
            (subgroup['BURDEN'+endpoint+'_75RATE'].mean()*
              refpop['FRAC_75-79'].mean())+
            (subgroup['BURDEN'+endpoint+'_80RATE'].mean()*
              refpop['FRAC_80-84'].mean())+
            (subgroup['BURDEN'+endpoint+'_85RATE'].mean()*
              refpop['FRAC_GT85'].mean()))
    elif 'IEc' in endpoint: 
        endpoint = 'AC'
        pop = ['pop_m_25-29', 'pop_f_25-29', 'pop_m_30-34', 'pop_m_35-39', 
            'pop_m_40-44', 'pop_m_45-49', 'pop_m_50-54', 'pop_m_55-59', 
            'pop_m_60-61', 'pop_m_62-64', 'pop_m_65-66', 'pop_m_67-69', 
            'pop_m_70-74', 'pop_m_75-79', 'pop_m_80-84', 'pop_m_gt85', 
            'pop_f_30-34', 'pop_f_35-39', 'pop_f_40-44', 'pop_f_45-49', 
            'pop_f_50-54', 'pop_f_55-59', 'pop_f_60-61', 'pop_f_62-64', 
            'pop_f_65-66', 'pop_f_67-69', 'pop_f_70-74', 'pop_f_75-79', 
            'pop_f_80-84', 'pop_f_gt85']    
        refpop['FRAC_25-34'] = (refpop[['pop_m_25-29', 'pop_f_25-29',
            'pop_m_30-34','pop_f_30-34']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_35-44'] = (refpop[['pop_m_35-39','pop_f_35-39',
            'pop_m_40-44','pop_f_40-44']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_45-54'] = (refpop[['pop_m_45-49','pop_f_45-49',
            'pop_m_50-54','pop_f_50-54']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_55-64'] = (refpop[['pop_m_55-59','pop_f_55-59',
            'pop_m_60-61','pop_m_62-64','pop_f_60-61','pop_f_62-64']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_65-74'] = (refpop[['pop_m_65-66','pop_m_67-69',
            'pop_f_65-66','pop_f_67-69','pop_m_70-74','pop_f_70-74']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_75-84'] = (refpop[['pop_m_75-79','pop_f_75-79',
            'pop_m_80-84','pop_f_80-84']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_GT85'] = (refpop[['pop_m_gt85','pop_f_gt85']
            ].sum(axis=1)/refpop[pop].sum(axis=1))        
        # Weight
        subgroup_as = ((subgroup['BURDEN'+endpoint+'_25RATE'].mean()*
              refpop['FRAC_25-34'].mean())+
            (subgroup['BURDEN'+endpoint+'_35RATE'].mean()*
              refpop['FRAC_35-44'].mean())+
            (subgroup['BURDEN'+endpoint+'_45RATE'].mean()*
              refpop['FRAC_45-54'].mean())+
            (subgroup['BURDEN'+endpoint+'_55RATE'].mean()*
              refpop['FRAC_55-64'].mean())+
            (subgroup['BURDEN'+endpoint+'_65RATE'].mean()*
              refpop['FRAC_65-74'].mean())+
            (subgroup['BURDEN'+endpoint+'_75RATE'].mean()*
              refpop['FRAC_75-84'].mean())+
            (subgroup['BURDEN'+endpoint+'_85RATE'].mean()*
              refpop['FRAC_GT85'].mean()))        
    elif 'LRI' in endpoint:
        refpop['FRAC_LT5'] = (refpop[['pop_m_lt5','pop_f_lt5']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_5-9'] = (refpop[['pop_m_5-9','pop_f_5-9']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_10-14'] = (refpop[['pop_m_10-14','pop_f_10-14']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_15-19'] = (refpop[['pop_m_15-17','pop_m_18-19',
            'pop_f_15-17','pop_f_18-19']].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_20-24'] = (refpop[['pop_m_20','pop_m_21','pop_m_22-24',
            'pop_f_20','pop_f_21','pop_f_22-24']].sum(axis=1)/
            refpop['pop_tot'])
        refpop['FRAC_25-29'] = (refpop[['pop_m_25-29','pop_f_25-29']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_30-34'] = (refpop[['pop_m_30-34','pop_f_30-34']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_35-39'] = (refpop[['pop_m_35-39','pop_f_35-39']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_40-44'] = (refpop[['pop_m_40-44','pop_f_40-44']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_45-49'] = (refpop[['pop_m_45-49','pop_f_45-49']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_50-54'] = (refpop[['pop_m_50-54','pop_f_50-54']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_55-59'] = (refpop[['pop_m_55-59','pop_f_55-59']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_60-64'] = (refpop[['pop_m_60-61','pop_m_62-64',
            'pop_f_60-61','pop_f_62-64']].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_65-69'] = (refpop[['pop_m_65-66','pop_m_67-69',
            'pop_f_65-66','pop_f_67-69']].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_70-74'] = (refpop[['pop_m_70-74','pop_f_70-74']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_75-79'] = (refpop[['pop_m_75-79','pop_f_75-79']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_80-84'] = (refpop[['pop_m_80-84','pop_f_80-84']
            ].sum(axis=1)/refpop['pop_tot'])
        refpop['FRAC_GT85'] = (refpop[['pop_m_gt85','pop_f_gt85']
            ].sum(axis=1)/refpop['pop_tot'])
        # Weight        
        subgroup_as = ((subgroup['BURDEN'+endpoint+'_LT5RATE'].mean()*
              refpop['FRAC_LT5'].mean())+
            (subgroup['BURDEN'+endpoint+'_5RATE'].mean()*
              refpop['FRAC_5-9'].mean())+
            (subgroup['BURDEN'+endpoint+'_10RATE'].mean()*
              refpop['FRAC_10-14'].mean())+
            (subgroup['BURDEN'+endpoint+'_15RATE'].mean()*
              refpop['FRAC_15-19'].mean())+
            (subgroup['BURDEN'+endpoint+'_20RATE'].mean()*
              refpop['FRAC_20-24'].mean())+
            (subgroup['BURDEN'+endpoint+'_25RATE'].mean()*
              refpop['FRAC_25-29'].mean())+
            (subgroup['BURDEN'+endpoint+'_30RATE'].mean()*
              refpop['FRAC_30-34'].mean())+
            (subgroup['BURDEN'+endpoint+'_35RATE'].mean()*
              refpop['FRAC_35-39'].mean())+
            (subgroup['BURDEN'+endpoint+'_40RATE'].mean()*
              refpop['FRAC_40-44'].mean())+
            (subgroup['BURDEN'+endpoint+'_45RATE'].mean()*
              refpop['FRAC_45-49'].mean())+
            (subgroup['BURDEN'+endpoint+'_50RATE'].mean()*
              refpop['FRAC_50-54'].mean())+
            (subgroup['BURDEN'+endpoint+'_55RATE'].mean()*
              refpop['FRAC_55-59'].mean())+
            (subgroup['BURDEN'+endpoint+'_60RATE'].mean()*
              refpop['FRAC_60-64'].mean())+
            (subgroup['BURDEN'+endpoint+'_65RATE'].mean()*
              refpop['FRAC_65-69'].mean())+
            (subgroup['BURDEN'+endpoint+'_70RATE'].mean()*
              refpop['FRAC_70-74'].mean())+
            (subgroup['BURDEN'+endpoint+'_75RATE'].mean()*
              refpop['FRAC_75-79'].mean())+
            (subgroup['BURDEN'+endpoint+'_80RATE'].mean()*
              refpop['FRAC_80-84'].mean())+
            (subgroup['BURDEN'+endpoint+'_85RATE'].mean()*
              refpop['FRAC_GT85'].mean()))         
    else:   
        pop = ['pop_m_25-29','pop_f_25-29','pop_m_30-34','pop_f_30-34',
            'pop_m_35-39','pop_f_35-39','pop_m_40-44','pop_f_40-44',
            'pop_m_45-49','pop_f_45-49','pop_m_50-54','pop_f_50-54',
            'pop_m_55-59','pop_f_55-59','pop_m_60-61','pop_m_62-64',
            'pop_f_60-61','pop_f_62-64','pop_m_65-66','pop_m_67-69',
            'pop_f_65-66','pop_f_67-69','pop_m_70-74','pop_f_70-74',
            'pop_m_75-79','pop_f_75-79','pop_m_80-84','pop_f_80-84',
            'pop_m_gt85','pop_f_gt85']
        refpop['FRAC_25-29'] = (refpop[['pop_m_25-29','pop_f_25-29']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_30-34'] = (refpop[['pop_m_30-34','pop_f_30-34']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_35-39'] = (refpop[['pop_m_35-39','pop_f_35-39']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_40-44'] = (refpop[['pop_m_40-44','pop_f_40-44']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_45-49'] = (refpop[['pop_m_45-49','pop_f_45-49']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_50-54'] = (refpop[['pop_m_50-54','pop_f_50-54']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_55-59'] = (refpop[['pop_m_55-59','pop_f_55-59']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_60-64'] = (refpop[['pop_m_60-61','pop_m_62-64',
            'pop_f_60-61','pop_f_62-64']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_65-69'] = (refpop[['pop_m_65-66','pop_m_67-69',
            'pop_f_65-66','pop_f_67-69']].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_70-74'] = (refpop[['pop_m_70-74','pop_f_70-74']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_75-79'] = (refpop[['pop_m_75-79','pop_f_75-79']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_80-84'] = (refpop[['pop_m_80-84','pop_f_80-84']
            ].sum(axis=1)/refpop[pop].sum(axis=1))
        refpop['FRAC_GT85'] = (refpop[['pop_m_gt85','pop_f_gt85']
            ].sum(axis=1)/refpop[pop].sum(axis=1))        
        # Weight
        subgroup_as = ((subgroup['BURDEN'+endpoint+'_25RATE'].mean()*
              refpop['FRAC_25-29'].mean())+
            (subgroup['BURDEN'+endpoint+'_30RATE'].mean()*
              refpop['FRAC_30-34'].mean())+
            (subgroup['BURDEN'+endpoint+'_35RATE'].mean()*
              refpop['FRAC_35-39'].mean())+
            (subgroup['BURDEN'+endpoint+'_40RATE'].mean()*
              refpop['FRAC_40-44'].mean())+
            (subgroup['BURDEN'+endpoint+'_45RATE'].mean()*
              refpop['FRAC_45-49'].mean())+
            (subgroup['BURDEN'+endpoint+'_50RATE'].mean()*
              refpop['FRAC_50-54'].mean())+
            (subgroup['BURDEN'+endpoint+'_55RATE'].mean()*
              refpop['FRAC_55-59'].mean())+
            (subgroup['BURDEN'+endpoint+'_60RATE'].mean()*
              refpop['FRAC_60-64'].mean())+
            (subgroup['BURDEN'+endpoint+'_65RATE'].mean()*
              refpop['FRAC_65-69'].mean())+
            (subgroup['BURDEN'+endpoint+'_70RATE'].mean()*
              refpop['FRAC_70-74'].mean())+
            (subgroup['BURDEN'+endpoint+'_75RATE'].mean()*
              refpop['FRAC_75-79'].mean())+
            (subgroup['BURDEN'+endpoint+'_80RATE'].mean()*
              refpop['FRAC_80-84'].mean())+
            (subgroup['BURDEN'+endpoint+'_85RATE'].mean()*
              refpop['FRAC_GT85'].mean()))
    return subgroup_as

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
    no2_aqs2010 = aqs2010.loc[(aqs2010['Parameter Code'] == 42602) &
        (aqs2010['Metric Used'] == 'Observed values')].copy(deep=True)
    no2_aqs2015 = aqs2015.loc[(aqs2015['Parameter Code'] == 42602) &
        (aqs2015['Metric Used'] == 'Observed values')].copy(deep=True)
    no2_aqs2019 = aqs2019.loc[(aqs2019['Parameter Code'] == 42602) &
        (aqs2019['Metric Used'] == 'Observed values')].copy(deep=True)
    pm25_aqs2010 = aqs2010.loc[(aqs2010['Parameter Code'] == 88101) &
        (aqs2010['Metric Used'] == 'Daily Mean') &
        (aqs2010['Pollutant Standard'] == 'PM25 24-hour 2012')].copy(deep=True)
    pm25_aqs2015 = aqs2015.loc[(aqs2015['Parameter Code'] == 88101) &
        (aqs2015['Metric Used'] == 'Daily Mean') &
        (aqs2015['Pollutant Standard'] == 'PM25 24-hour 2012')].copy(deep=True)
    pm25_aqs2019 = aqs2019.loc[(aqs2019['Parameter Code'] == 88101) &
        (aqs2019['Metric Used'] == 'Daily Mean') &
        (aqs2019['Pollutant Standard'] == 'PM25 24-hour 2012')].copy(deep=True)
    # Harmonize
    harm2010 = harmts.loc[harmts.YEAR == '2006-2010']
    harm2015 = harmts.loc[harmts.YEAR == '2011-2015']
    harm2019 = harmts.loc[harmts.YEAR == '2015-2019']
    harmonize_aqstract_yearpoll(2010, harm2010, no2_aqs2010, 'NO2')
    harmonize_aqstract_yearpoll(2015, harm2015, no2_aqs2015, 'NO2')
    harmonize_aqstract_yearpoll(2019, harm2019, no2_aqs2019, 'NO2')
    harmonize_aqstract_yearpoll(2010, harm2010, pm25_aqs2010, 'PM25')
    harmonize_aqstract_yearpoll(2015, harm2015, pm25_aqs2015, 'PM25')
    harmonize_aqstract_yearpoll(2019, harm2019, pm25_aqs2019, 'PM25')
    return

def add_insetmap(axes_extent, map_extent, state_name, geometry, lng, lat,
                 quant, proj, fc='#f2f2f2', fips=None, harmonized=None, vara=None,
                 cmap=None, norm=None, sc=None, ec='None', linewidth=0.):
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
    if cmap is not None:
        sub_ax.scatter(lng, lat, s=quant, c=sc, zorder=15, ec=ec, 
            linewidth=linewidth, clip_on=True, cmap=cmap, norm=norm, 
            transform=ccrs.PlateCarree())
    else:
        sub_ax.scatter(lng, lat, s=quant, alpha=0.4, c=cscat, ec='None',
            zorder=15, clip_on=True, transform=ccrs.PlateCarree())
        sub_ax.scatter(lng, lat, s=quant, c='None', linewidth=.25, ec=cscat,
            transform=proj, zorder=15, clip_on=True)
    sub_ax.axis('off')
    if fips is not None:
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO +
                                 'tract_2010/tl_2019_%s_tract/tl_2019_%s_tract.shp' % (fips, fips))
        records = shp.records()
        tracts = shp.geometries()
        for record, tract in zip(records, tracts):
            # Find GEOID of tract
            gi = record.attributes['GEOID']
            # Look up harmonized NO2-census data for tract
            harmonized_tract = harmonized.loc[harmonized.index.isin([gi])]
            if harmonized_tract.empty == True:
                sub_ax.add_geometries([tract], proj, facecolor='none',
                                      edgecolor="none", alpha=1., linewidth=0., rasterized=True,
                                      zorder=10)
            else:
                var_tract = harmonized_tract['%s' % vara].values[0]
                if np.isnan(var_tract) == True:
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
        x = [0] + np.cos(np.linspace(previous, this, 30)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 30)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker': xy, 's': np.abs(xy).max() **
                       2*np.array(sizes), 'facecolor': color})
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


def w_avg(df, values, weights):
    """Calculate population-weighted mean timeseries

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.
    weights : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()

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
        ].sum(axis=1)).sum()/burden[['race_nh_black', 'race_h_black']
        ].sum(axis=1).sum())
    no2black = ((burden['NO2']*burden[['race_nh_black', 'race_h_black']
        ].sum(axis=1)).sum()/burden[['race_nh_black', 'race_h_black']
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
    asthmablack = ((burden['BURDENPARATE']*burden[['race_nh_black',
        'race_h_black']].sum(axis=1)).sum()/burden[['race_nh_black',
        'race_h_black']].sum(axis=1).sum())
    allmortblack = ((burden['BURDENPMALLRATE']*burden[[
        'race_nh_black', 'race_h_black']].sum(axis=1)).sum()/burden[[
        'race_nh_black', 'race_h_black']].sum(axis=1).sum())
    # For whites
    pm25white = ((burden['PM25']*burden[['race_nh_white', 'race_h_white']
        ].sum(axis=1)).sum()/burden[['race_nh_white', 'race_h_white']
        ].sum(axis=1).sum())
    no2white = ((burden['NO2']*burden[['race_nh_white', 'race_h_white']
        ].sum(axis=1)).sum()/burden[['race_nh_white', 'race_h_white']
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
    asthmawhite = ((burden['BURDENPARATE']*burden[['race_nh_white',
       'race_h_white']].sum(axis=1)).sum()/burden[['race_nh_white',
       'race_h_white']].sum(axis=1).sum())
    allmortwhite = ((burden['BURDENPMALLRATE']*burden[
        ['race_nh_white', 'race_h_white']].sum(axis=1)).sum()/burden[
        ['race_nh_white', 'race_h_white']].sum(axis=1).sum())
    # For non-Hispanic
    no2nh = ((burden['NO2']*burden['race_nh']).sum()/burden['race_nh'].sum())
    pm25nh = ((burden['PM25']*burden['race_nh']).sum()/burden['race_nh'].sum())
    copdnh = ((burden['BURDENCOPDRATE']*burden['race_nh']).sum() /
        burden['race_nh'].sum())
    ihdnh = ((burden['BURDENIHDRATE']*burden['race_nh']).sum() /
        burden['race_nh'].sum())
    lrinh = ((burden['BURDENLRIRATE']*burden['race_nh']).sum() /
        burden['race_nh'].sum())
    dmnh = ((burden['BURDENDMRATE']*burden['race_nh']).sum() /
        burden['race_nh'].sum())
    lcnh = ((burden['BURDENLCRATE']*burden['race_nh']).sum() /
        burden['race_nh'].sum())
    stnh = ((burden['BURDENSTRATE']*burden['race_nh']).sum() /
        burden['race_nh'].sum())
    asthmanh = ((burden['BURDENPARATE']*burden['race_nh']).sum() /
        burden['race_nh'].sum())
    allmortnh = ((burden['BURDENPMALLRATE']*burden[
        'race_nh']).sum()/burden['race_nh'].sum())
    # For Hispanic
    no2h = ((burden['NO2']*burden['race_h']).sum()/burden['race_h'].sum())
    pm25h = ((burden['PM25']*burden['race_h']).sum()/burden['race_h'].sum())
    copdh = ((burden['BURDENCOPDRATE']*burden['race_h']).sum() /
        burden['race_h'].sum())
    ihdh = ((burden['BURDENIHDRATE']*burden['race_h']).sum() /
        burden['race_h'].sum())
    lrih = ((burden['BURDENLRIRATE']*burden['race_h']).sum() /
        burden['race_h'].sum())
    dmh = ((burden['BURDENDMRATE']*burden['race_h']).sum() /
        burden['race_h'].sum())
    lch = ((burden['BURDENLCRATE']*burden['race_h']).sum() /
        burden['race_h'].sum())
    sth = ((burden['BURDENSTRATE']*burden['race_h']).sum() /
        burden['race_h'].sum())
    asthmah = ((burden['BURDENPARATE']*burden['race_h']).sum() /
        burden['race_h'].sum())
    allmorth = ((burden['BURDENPMALLRATE']*burden['race_h']).sum()/burden[
        'race_h'].sum())
    return (pm25black, no2black, copdblack, ihdblack, lriblack, dmblack,
        lcblack, stblack, asthmablack, allmortblack, pm25white,
        no2white, copdwhite, ihdwhite, lriwhite, dmwhite, lcwhite, stwhite,
        asthmawhite, allmortwhite, pm25nh, no2nh, copdnh, ihdnh,
        lrinh, dmnh, lcnh, stnh, asthmanh, allmortnh, pm25h, no2h,
        copdh, ihdh, lrih, dmh, lch, sth, asthmah, allmorth)

def fig1():
    """

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.io import shapereader
    import cartopy.feature as cfeature
    from operator import itemgetter
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from scipy.interpolate import interpn
    from pylr2 import regress2
    
    def density_scatter(x, y, fig=None, ax=None, sort=True, vmin=None,
                        vmax=None, bins=20, **kwargs):
        """
        Scatter plot colored by 2d histogram
        """
        if ax is None:
            fig, ax = plt.subplots()
        mask = ~np.isnan(x) & ~np.isnan(y)
        data, x_e, y_e = np.histogram2d(
            x[mask], y[mask], bins=bins, density=True)
        z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                    data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0
        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, **kwargs)
        # norm = Normalize(vmin = vmin, vmax=vmax)#np.min(z), vmax = np.max(z))
        # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        # # cbar.ax.set_ylabel('Density')
        return ax
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
    # Plotting
    proj = ccrs.AlbersEqualArea(central_longitude=-98,
        central_latitude=39.5, standard_parallels=(29.5, 45.5))
    fig = plt.figure(figsize=(9, 9))
    axtl = plt.subplot2grid((3, 6), (0, 0), colspan=3, projection=proj)
    axtr = plt.subplot2grid((3, 6), (0, 3), colspan=3, projection=proj)
    ax1 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
    ax3 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
    ax4 = plt.subplot2grid((3, 6), (2, 0), colspan=2)
    ax5 = plt.subplot2grid((3, 6), (2, 2), colspan=2)
    ax6 = plt.subplot2grid((3, 6), (2, 4), colspan=2)
    # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()
    usa = [x.attributes['ADM0_A3'] for x in countries]
    usa = np.where(np.array(usa) == 'USA')[0][0]
    usa = list(reader.geometries())[usa].geoms
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shapereader.Reader(lakes)
    lakes = lakes_reader.records()
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names) == 'Lake Superior') |
                            (np.array(lake_names) == 'Lake Michigan') |
                            (np.array(lake_names) == 'Lake Huron') |
                            (np.array(lake_names) == 'Lake Erie') |
                            (np.array(lake_names) == 'Lake Ontario'))[0]
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
    alaska = states_all[np.where(np.array(states_all_name) == 'Alaska')[0]][0]
    hawaii = states_all[np.where(np.array(states_all_name) == 'Hawaii')[0]][0]
    puertorico = states_all[np.where(np.array(states_all_name) ==
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
    norm = mpl.colors.Normalize(vmin=-4, vmax=4)
    # Plot location of monitors
    for ax, monitors in zip([axtl, axtr], [no2_aqs2019, pm25_aqs2019]):
        ax.scatter(monitors['Longitude'], monitors['Latitude'], 
            transform=ccrs.PlateCarree(), zorder=30, clip_on=True, s=7, 
            alpha=0.4, linewidth=0.25, c=(monitors['Arithmetic Mean']-
            monitors['DATASET_VAL']), ec='k', cmap=cmap, norm=norm)
        ax.set_extent([-125, -66.5, 24.5, 49.48], crs=ccrs.PlateCarree())
        for usai in usa:
            ax.add_geometries([usai], crs=ccrs.PlateCarree(), lw=0.25, 
                fc='None', ec='k', zorder=1)
        ax.add_geometries(great_lakes, crs=ccrs.PlateCarree(), facecolor='w',
            lw=0.25, edgecolor='k', alpha=1., zorder=12)
        ax.add_feature(cfeature.NaturalEarthFeature('physical',
            'ocean', '10m', edgecolor='None', facecolor='w', alpha=1.),
            zorder=11)
        ax.axis('off')
        # Add state outlines
        for astate in states_shp.records():
            if astate.attributes['sr_adm0_a3'] == 'USA':
                geometry = astate.geometry
                ax.add_geometries([geometry], crs=ccrs.PlateCarree(),
                    fc='#f2f2f2', lw=0.5, ec='w', alpha=1., zorder=0)
        # # Hawaii
        axes_extent = (ax.get_position().x0+0.018, ax.get_position().y0-0.005,
            (ax.get_position().x1-ax.get_position().x0)*0.2,
            (ax.get_position().x1-ax.get_position().x0)*0.2)
        add_insetmap(axes_extent, (-162, -154, 18.75, 23), '',
            hawaii.geometry, monitors['Longitude'], monitors['Latitude'],
            6, proj, sc=(monitors['Arithmetic Mean']-monitors['DATASET_VAL']),
            cmap=cmap, norm=norm, ec='k', linewidth=0.25)
        # Alaska
        axes_extent = (ax.get_position().x0-0.06, ax.get_position().y0-0.01,
            (ax.get_position().x1-ax.get_position().x0)*0.25,
            (ax.get_position().x1-ax.get_position().x0)*0.25)
        add_insetmap(axes_extent, (-179.99, -130, 49, 73), '',
            alaska.geometry, monitors['Longitude'], monitors['Latitude'],
            6, proj, sc=(monitors['Arithmetic Mean']-monitors['DATASET_VAL']),
            cmap=cmap, norm=norm, ec='k', linewidth=0.25)
        # Puerto Rico
        axes_extent = (ax.get_position().x0+0.09, ax.get_position().y0-0.01,
            (ax.get_position().x1-ax.get_position().x0)*0.18,
            (ax.get_position().x1-ax.get_position().x0)*0.18)
        add_insetmap(axes_extent, (-68., -65., 17.5, 19.), '',
            puertorico.geometry, monitors['Longitude'],
            monitors['Latitude'], 6, proj, sc=(monitors['Arithmetic Mean'] -
            monitors['DATASET_VAL']), cmap=cmap, norm=norm, ec='k', linewidth=0.25)
    density_scatter(no2_aqs2010['Arithmetic Mean'], no2_aqs2010['DATASET_VAL'],
        fig=fig, ax=ax1, vmin=0, vmax=0.02, bins=20, s=5,
        cmap=plt.get_cmap('inferno'), zorder=10)
    density_scatter(no2_aqs2015['Arithmetic Mean'], no2_aqs2015['DATASET_VAL'],
        fig=fig, ax=ax2, vmin=0, vmax=0.02, bins=20, s=5,
        cmap=plt.get_cmap('inferno'), zorder=10)
    density_scatter(no2_aqs2019['Arithmetic Mean'], no2_aqs2019['DATASET_VAL'],
        fig=fig, ax=ax3, vmin=0, vmax=0.02, bins=20, s=5,
        cmap=plt.get_cmap('inferno'), zorder=10)
    density_scatter(pm25_aqs2010['Arithmetic Mean'], pm25_aqs2010['DATASET_VAL'],
        fig=fig, ax=ax4, bins=20, s=5, vmin=0, vmax=0.08,
        cmap=plt.get_cmap('inferno'), zorder=10)
    density_scatter(pm25_aqs2015['Arithmetic Mean'], pm25_aqs2015['DATASET_VAL'],
        fig=fig, ax=ax5, bins=20, s=5, vmin=0, vmax=0.08,
        cmap=plt.get_cmap('inferno'), zorder=10)
    density_scatter(pm25_aqs2019['Arithmetic Mean'], pm25_aqs2019['DATASET_VAL'],
        fig=fig, ax=ax6, bins=20, s=5, vmin=0, vmax=0.08,
        cmap=plt.get_cmap('inferno'), zorder=10)
    # Ticks and labels for NO2 plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([0, 30])
        ax.set_xticks(np.linspace(0, 30, 7))
        ax.set_ylim([0, 30])
        ax.set_yticks(np.linspace(0, 30, 7))
        ax.set_yticklabels([])
        ax.set_xlabel('Observed NO$_{2}$ [ppbv]', loc='left')
    # Add 1:1 and 2:1/1:2 lines, statistics, etc
    for ax, df in zip([ax1, ax2, ax3, ax4, ax5, ax6], [no2_aqs2010, no2_aqs2015,
        no2_aqs2019, pm25_aqs2010, pm25_aqs2015, pm25_aqs2019]):
        # 1:1 and 2:1/1:2 lines
        ax.plot(np.linspace(0, 30, 100), np.linspace(0, 30, 100), '--',
                lw=1., color='grey', zorder=0, label='1:1')
        ax.plot(np.linspace(0, 30, 100), 2*np.linspace(0, 30, 100), '-',
                lw=0.5, color='grey', zorder=0)
        ax.plot(np.linspace(0, 30, 100), 0.5*np.linspace(0, 30, 100), '-',
                lw=0.5, color='grey', zorder=0, label='1:2 and 2:1')
        # Calculate reduced major axis linear regression to account for the
        # attenuation in slope that occurs when there is uncertainty in the
        # x-axis (which in this case will be driven by representativeness bias
        # between a point and an area average). The approximate effect will
        # be an increase in slope by 1/r.
        idx = np.isfinite(df['Arithmetic Mean']
            ) & np.isfinite(df['DATASET_VAL'])
        results = regress2(df['Arithmetic Mean'][idx], df['DATASET_VAL'][idx],
            _method_type_2="reduced major axis")
        # Correlation coefficient
        r = np.corrcoef(df['Arithmetic Mean'][idx], 
            df['DATASET_VAL'][idx])[0, 1]
        # Normalized mean bias
        nmb = (np.nansum(df['DATASET_VAL'][idx]-df['Arithmetic Mean'][idx]) /
                np.nansum(df['Arithmetic Mean'][idx]))
        ax.plot(np.linspace(0, 30, 100), (np.linspace(0, 30, 100) *
                                          results['slope'] + results['intercept']), '-', color=cmap(0),
                zorder=1, label='Fit')
        ax.text(0.03, 0.94, 'm=%.2f, b=%.2f' % (results['slope'],
                                                results['intercept']), ha='left', va='center', transform=ax.transAxes,
                fontsize=8)
        ax.text(0.03, 0.86, 'N=%d' % np.where(idx == True)[0].shape[0], ha='left',
                va='center', transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.78, 'NMB=%.2f' % nmb, ha='left', va='center',
                transform=ax.transAxes, fontsize=8)
        ax.text(0.03, 0.7, 'r=%.2f' % r, ha='left', va='center',
                transform=ax.transAxes, fontsize=8)
    ax1.set_ylabel('Anenberg, Mohegh, et al.\n(2022) NO$_{2}$ [ppbv]',
                    loc='bottom')
    ax1.set_yticklabels([int(x) for x in np.linspace(0, 30, 7)])
    # Ticks and labels for PM2.5 plots
    for ax in [ax4, ax5, ax6]:
        ax.set_xlim([0, 20])
        ax.set_xticks(np.linspace(0, 20, 6))
        ax.set_ylim([0, 20])
        ax.set_yticks(np.linspace(0, 20, 6))
        ax.set_yticklabels([])
        ax.set_xlabel('Observed PM$_{2.5}$ [$\mathregular{\mu}$g ' +
            'm$^{\mathregular{-3}}$]', loc='left')
    ax4.set_ylabel('van Donkelaar et al. (2021)\nPM$_{2.5}$' +
            ' [$\mathregular{\mu}$g m$^{\mathregular{-3}}$]', loc='bottom')
    ax4.set_yticklabels([int(x) for x in np.linspace(0, 20, 6)])
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    # Add colorbars
    # For maps
    cax = fig.add_axes([axtr.get_position().x1+0.01, axtr.get_position().y0, 0.01,
        (axtr.get_position().y1-axtr.get_position().y0)])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, spacing='proportional', 
        orientation='vertical', extend='both',
        label='[$\mathregular{\mu}$g m$^{\mathregular{-3}}$ | ppbv]')
    # For NO2 scatter
    cax = fig.add_axes([ax3.get_position().x1+0.01, ax3.get_position().y0, 
        0.01, (ax3.get_position().y1 -ax3.get_position().y0)])
    norm = Normalize(vmin=0, vmax=0.02)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('inferno'),
        norm=norm, spacing='proportional', orientation='vertical',
        extend='max', label='Density', ticks=[0, 0.005, 0.01, 0.015, 0.02])
    cbar.set_ticklabels(['0', '', '0.01', '', '0.02'])
    # For PM2.5 scatter
    cax = fig.add_axes([ax6.get_position().x1+0.01, ax6.get_position().y0, 
        0.01, (ax3.get_position().y1-ax3.get_position().y0)])
    norm = Normalize(vmin=0, vmax=0.08)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap('inferno'),
        norm=norm, spacing='proportional', orientation='vertical',
        extend='max', label='Density', ticks=[0, 0.02, 0.04, 0.06, 0.08])
    cbar.set_ticklabels(['0', '', '0.04', '', '0.08'])
    # Add legend
    ax5.legend(ncol=3, frameon=False, bbox_to_anchor=(0.4, -0.7), loc=8,
        fontsize=14)
    plt.savefig(DIR_FIG+'fig1_REVISED.png', dpi=1000)
    return

def fig2(burdents):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    from cartopy.io import shapereader
    import matplotlib.patches as mpatches
    from scipy import stats
    # # # # Constants
    years = np.arange(2010, 2020, 1)
    pm25_pwm, no2_pwm = [], []
    for year in years:
        pm25_pwm.append(w_avg(burdents.loc[burdents.YEAR == '%d-%d' % (
            year-4, year)], 'PM25', 'pop_tot'))
        no2_pwm.append(w_avg(burdents.loc[burdents.YEAR == '%d-%d' % (
            year-4, year)], 'NO2', 'pop_tot'))
    # Calculate some pithy statistics
    trend_pm25 = stats.linregress(years, pm25_pwm)
    trend_no2 = stats.linregress(years, no2_pwm)
    print('# # # # Nationwide PM2.5 trends')
    print('Slope/p-value = ', trend_pm25.slope, '/', trend_pm25.pvalue)
    pc_pm25 = ((pm25_pwm[-1]-pm25_pwm[0])/pm25_pwm[0])*100
    print('Percent change', pc_pm25, '%')
    print('# # # # Nationwide NO2 trends')
    print('Slope/p-value = ', trend_no2.slope, '/', trend_no2.pvalue)
    pc_no2 = ((no2_pwm[-1]-no2_pwm[0])/no2_pwm[0])*100
    print('Percent change', pc_no2, '%')
    for state in np.unique(burdents.STATE):
        burdents_state = burdents.loc[burdents.STATE.isin([state])]
        pm25_spwm, no2_spwm = [], []
        for year in years:
            pm25_spwm.append(w_avg(burdents_state.loc[
                burdents_state.YEAR == '%d-%d' % (year-4, year)], 'PM25', 'pop_tot'))
            no2_spwm.append(w_avg(burdents_state.loc[
                burdents_state.YEAR == '%d-%d' % (year-4, year)], 'NO2', 'pop_tot'))
        trend_pm25 = stats.linregress(years, pm25_spwm)
        trend_no2 = stats.linregress(years, no2_spwm)
        print('# # # # %s PM2.5 trends' % state)
        print('Slope/p-value = ', trend_pm25.slope, '/', trend_pm25.pvalue)
        pc_pm25 = ((pm25_spwm[-1]-pm25_spwm[0])/pm25_spwm[0])*100
        print('Percent change', pc_pm25, '%')
        print('# # # # %s NO2 trends' % state)
        print('Slope/p-value = ', trend_no2.slope, '/', trend_no2.pvalue)
        pc_no2 = ((no2_spwm[-1]-no2_spwm[0])/no2_spwm[0])*100
        print('Percent change', pc_no2, '%')
        print('\n')
    # Open burdens calculated without TMREL 
    notmrel = []
    for year in np.arange(2010, 2020, 1):
        print(year)
        notmrelty = pd.read_parquet(DIR_HARM+'burdens_%d_TROPOMIv2_noTMREL.gzip'%(year))
        notmrel.append(notmrelty)
    notmrel = pd.concat(notmrel)
    copd, lri, lc, ihd, dm, st, asthma = [], [], [], [], [], [], []
    copdin, lriin, lcin, ihdin, dmin, stin, asthmain = [], [], [], [], [], [], []
    copdnotmrel, lrinotmrel, lcnotmrel, ihdnotmrel = [], [], [], []
    dmnotmrel, stnotmrel, asthmanotmrel = [], [], []
    popped = ['pop_m_lt5','pop_f_lt5','pop_m_5-9','pop_f_5-9', 
        'pop_m_10-14','pop_f_10-14','pop_m_15-17','pop_m_18-19',
        'pop_f_15-17','pop_f_18-19']
    pop25p = ['pop_m_25-29', 'pop_f_25-29', 'pop_m_30-34', 'pop_m_35-39', 
        'pop_m_40-44', 'pop_m_45-49', 'pop_m_50-54', 'pop_m_55-59', 
        'pop_m_60-61', 'pop_m_62-64', 'pop_m_65-66', 'pop_m_67-69', 
        'pop_m_70-74', 'pop_m_75-79', 'pop_m_80-84', 'pop_m_gt85', 
        'pop_f_30-34', 'pop_f_35-39', 'pop_f_40-44', 'pop_f_45-49', 
        'pop_f_50-54', 'pop_f_55-59', 'pop_f_60-61', 'pop_f_62-64', 
        'pop_f_65-66', 'pop_f_67-69', 'pop_f_70-74', 'pop_f_75-79', 
        'pop_f_80-84', 'pop_f_gt85']
    for year in np.arange(2010, 2020, 1):
        vintage = '%d-%d' % (year-4, year)
        burdenty = burdents.loc[burdents['YEAR'] == vintage]
        notmrelty = notmrel.loc[notmrel['YEAR'] == vintage]
        # Burdens for health endpoints
        copd.append(burdenty.BURDENCOPD.sum())
        lri.append(burdenty.BURDENLRI.sum())
        lc.append(burdenty.BURDENLC.sum())
        ihd.append(burdenty.BURDENIHD.sum())
        dm.append(burdenty.BURDENDM.sum())
        st.append(burdenty.BURDENST.sum())
        asthma.append(burdenty.BURDENPA.sum())
        copdnotmrel.append(notmrelty.BURDENCOPD.sum())
        lrinotmrel.append(notmrelty.BURDENLRI.sum())
        lcnotmrel.append(notmrelty.BURDENLC.sum())
        ihdnotmrel.append(notmrelty.BURDENIHD.sum())
        dmnotmrel.append(notmrelty.BURDENDM.sum())
        stnotmrel.append(notmrelty.BURDENST.sum())
        asthmanotmrel.append(notmrelty.BURDENPA.sum())    
        # Incidence
        asthmain.append(((burdenty.BURDENPA/burdenty[popped].sum(
            axis=1))*100000).mean())
        copdin.append(((burdenty.BURDENCOPD/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        lcin.append(((burdenty.BURDENLC/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        ihdin.append(((burdenty.BURDENIHD/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        dmin.append(((burdenty.BURDENDM/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        stin.append(((burdenty.BURDENST/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        lriin.append(((burdenty.BURDENST/burdenty.pop_tot)*100000).mean())
    # Convert lists to arrays
    pm25_mean = np.array(pm25_pwm)
    no2_mean = np.array(no2_pwm)
    copd = np.array(copd)
    lri = np.array(lri)
    lc = np.array(lc)
    ihd = np.array(ihd)
    dm = np.array(dm)
    st = np.array(st)
    asthma = np.array(asthma)
    copdnotmrel = np.array(copdnotmrel)
    lrinotmrel = np.array(lrinotmrel)
    lcnotmrel = np.array(lcnotmrel)
    ihdnotmrel = np.array(ihdnotmrel)
    dmnotmrel = np.array(dmnotmrel)
    stnotmrel = np.array(stnotmrel)
    asthmanotmrel = np.array(asthmanotmrel)
    copdin = np.array(copdin)
    lriin = np.array(lriin)
    lcin = np.array(lcin)
    ihdin = np.array(ihdin)
    dmin = np.array(dmin)
    stin = np.array(stin)
    asthmain = np.array(asthmain)
    # Plotting
    fig = plt.figure(figsize=(9, 3.5))
    axts1 = plt.subplot2grid((1, 2), (0, 0))
    axts2 = plt.subplot2grid((1, 2), (0, 1))
    ax1t = axts1.twinx()
    ax2t = axts2.twinx()
    axts1.set_title('(A) Premature deaths due to PM$_\mathregular{2.5}$',
        loc='left')
    axts2.set_title('(B) New asthma cases due to NO$_\mathregular{2}$',
        loc='left')
    # PM25 and PM25-attributable mortality
    for i in np.arange(0, len(np.arange(2010, 2020, 1)), 1):
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
    axts1.set_ylim([0, 75000])
    axts1.set_yticks(np.linspace(0, 75000, 6))
    ax1t.yaxis.set_label_coords(1.15, 0.5)
    ax1t.set_ylim([0, 10])
    ax1t.set_yticks(np.linspace(0, 10, 6))
    ax1t.set_ylabel('PM$_{\mathregular{2.5}}$ [$\mathregular{\mu}$g m$' +
                    '^{\mathregular{-3}}$]', rotation=270)
    # NO2 and NO2-attributable new cases
    axts2.bar(years, asthma, color=color7, zorder=10)
    ax2t.errorbar(years, no2_mean, ls='-', marker='o', color='k')
    axts2.set_ylim([0, 200000])
    axts2.set_yticks(np.linspace(0, 200000, 5))
    ax2t.set_ylim([0, 12])
    ax2t.set_yticks(np.linspace(0, 12, 5))
    ax2t.set_ylabel('NO$_{\mathregular{2}}$ [ppbv]', rotation=270)
    ax2t.yaxis.set_label_coords(1.15, 0.5)
    for ax in [ax1t, ax2t]:
        ax.set_xlim([2009.25, 2019.75])
        ax.set_xticks(np.arange(2010, 2020, 1))
        ax.set_xticklabels(['2010', '', '2012', '', '2014', '', '2016', '', '2018', ''])
    # Add legend denoting PM2.5 timeseries
    ax1t.annotate('PM$_{\mathregular{2.5}}$', xy=(2019, pm25_mean[-3]),
        xycoords='data', xytext=(2017.6, pm25_mean[-3]+1.2),
        textcoords='data', arrowprops=dict(arrowstyle='->', color='k'),
        fontsize=12)
    ax2t.annotate('NO$_{\mathregular{2}}$', xy=(2019, no2_mean[-3]),
        xycoords='data', xytext=(2018.2, no2_mean[-3]+2.1),
        textcoords='data', arrowprops=dict(arrowstyle='->', color='k'),
        fontsize=12)
    plt.subplots_adjust(left=0.08, wspace=0.44, bottom=0.3, right=0.92)
    # Legends (first create dummy)
    pihd = mpatches.Patch(color=color1, label='Ischemic heart disease')
    pst = mpatches.Patch(color=color2, label='Stroke')
    plc = mpatches.Patch(color=color3, label='Lung cancer')
    pcopd = mpatches.Patch(color=color4, label='COPD')
    pdm = mpatches.Patch(color=color5, label='Type 2 diabetes')
    plri = mpatches.Patch(color=color6, label='Lower respiratory infection')
    axts1.legend(handles=[pihd, pst, plc, pcopd, pdm, plri],
        bbox_to_anchor=(1.25, -0.12), ncol=2, frameon=False)
    plt.savefig(DIR_FIG+'fig2_REVISED.pdf', dpi=500)
    return 

def fig3(lng_allmsa, lat_allmsa, asthmaburdenrate_allmsa, pmburdenrate_allmsa):
    from cartopy.io import shapereader
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    # With Shapely 1.8, deprecation warnings will show that cannot be avoided 
    # (depending on the geometry type, NumPy tries to access the array 
    # interface of the objects or check if an object is iterable or has a 
    # length, and those operations are all deprecated now. The end result 
    # is still correct, but the warnings appear nonetheless). Ignore those 
    # warnings
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
    
    # # # # # Constants
    proj = ccrs.AlbersEqualArea(central_longitude=-98,
        central_latitude=39.5, standard_parallels=(29.5, 45.5))
    cscat = 'dodgerblue'
    
    # # # # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()
    usaidx = [x.attributes['ADM0_A3'] for x in countries]
    usaidx = np.where(np.in1d(np.array(usaidx), ['PRI', 'USA']) == True)
    usa = list(reader.geometries())
    usa = np.array(usa, dtype=object)[usaidx[0]]
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shpreader.Reader(lakes)
    lakes = lakes_reader.records()
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names) == 'Lake Superior') |
                            (np.array(lake_names) == 'Lake Michigan') |
                            (np.array(lake_names) == 'Lake Huron') |
                            (np.array(lake_names) == 'Lake Erie') |
                            (np.array(lake_names) == 'Lake Ontario'))[0]
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
    burdenrateb1 = np.where(asthmaburdenrate_allmsa < no2p30)[0]
    burdenrateb2 = np.where((asthmaburdenrate_allmsa >= no2p30) &
                            (asthmaburdenrate_allmsa < no2p60))[0]
    burdenrateb3 = np.where((asthmaburdenrate_allmsa >= no2p60) &
                            (asthmaburdenrate_allmsa < no2p90))[0]
    burdenrateb4 = np.where((asthmaburdenrate_allmsa >= no2p90) &
                            (asthmaburdenrate_allmsa < no2p95))[0]
    burdenrateb5 = np.where((asthmaburdenrate_allmsa >= no2p95))[0]
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
    burdenrateb1 = np.where(pmburdenrate_allmsa < pmp30)[0]
    burdenrateb2 = np.where((pmburdenrate_allmsa >= pmp30) &
                            (pmburdenrate_allmsa < pmp60))[0]
    burdenrateb3 = np.where((pmburdenrate_allmsa >= pmp60) &
                            (pmburdenrate_allmsa < pmp90))[0]
    burdenrateb4 = np.where((pmburdenrate_allmsa >= pmp90) &
                            (pmburdenrate_allmsa < pmp95))[0]
    burdenrateb5 = np.where((pmburdenrate_allmsa >= pmp95))[0]
    pmburdenrated_allmsa[burdenrateb1] = 6
    pmburdenrated_allmsa[burdenrateb2] = 20
    pmburdenrated_allmsa[burdenrateb3] = 40
    pmburdenrated_allmsa[burdenrateb4] = 70
    pmburdenrated_allmsa[burdenrateb5] = 200
    
    # # # # Plotting
    fig = plt.figure(figsize=(11, 4))
    ax2 = plt.subplot2grid((1, 2), (0, 0), projection=proj)
    ax1 = plt.subplot2grid((1, 2), (0, 1), projection=proj)
    ax2.set_title('(A) Premature deaths due to PM$_\mathregular{2.5}$ ' +
                  'per 100000', loc='left')
    ax1.set_title('(B) New asthma cases due to NO$_\mathregular{2}$ ' +
                  'per 100000', loc='left')
    # Add borders, set map extent, etc.
    for ax in [ax1, ax2]:
        ax.set_extent([-125, -66.5, 24.5, 49.48], crs=ccrs.PlateCarree())
        for usai in usa:
            ax.add_geometries(list(usai), crs=ccrs.PlateCarree(), fc='None', 
                lw=0.25, ec='k', alpha=1., zorder=15)    
        ax.add_geometries(great_lakes, crs=ccrs.PlateCarree(), fc='w', 
            lw=0.25, ec='k', alpha=1., zorder=17)
        ax.axis('off')
    ax1.scatter(lng_allmsa, lat_allmsa, s=asthmaburdenrated_allmsa, alpha=0.4,
        c=cscat, ec='None', transform=ccrs.PlateCarree(), zorder=30, clip_on=True)
    ax1.scatter(lng_allmsa, lat_allmsa, s=asthmaburdenrated_allmsa, fc='None',
        linewidth=.15, ec=cscat, transform=ccrs.PlateCarree(), zorder=30, clip_on=True)
    ax2.scatter(lng_allmsa, lat_allmsa, s=pmburdenrated_allmsa, alpha=0.4,
        c=cscat, ec='None', transform=ccrs.PlateCarree(), zorder=30, clip_on=True)
    ax2.scatter(lng_allmsa, lat_allmsa, s=pmburdenrated_allmsa, fc='None',
        linewidth=.15, ec=cscat, transform=ccrs.PlateCarree(), zorder=30, clip_on=True)
    
    b1 = ax1.scatter([], [], s=2, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b2 = ax1.scatter([], [], s=4, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b3 = ax1.scatter([], [], s=20, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b4 = ax1.scatter([], [], s=50, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b5 = ax1.scatter([], [], s=300, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    ax1.legend((b1, b2, b3, b4, b5),
                ('< %d' % (no2p30), '%d-%d' % (no2p30, no2p60), '%d-%d' % (no2p60, no2p90),
                '%d-%d' % (no2p90, no2p95), '> %d' % (no2p90)), scatterpoints=1,
                labelspacing=0.8, loc='center right', bbox_to_anchor=(1.04, 0.32),
                ncol=1, frameon=False, fontsize=8)
    b1 = ax2.scatter([], [], s=2, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b2 = ax2.scatter([], [], s=4, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b3 = ax2.scatter([], [], s=20, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b4 = ax2.scatter([], [], s=50, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    b5 = ax2.scatter([], [], s=300, marker='o', ec=cscat, fc=cscat, linewidth=.15,
                      alpha=0.4)
    ax2.legend((b1, b2, b3, b4, b5),
                ('< %d' % (pmp30), '%d-%d' % (pmp30, pmp60), '%d-%d' % (pmp60, pmp90),
                '%d-%d' % (pmp90, pmp95), '> %d' % (pmp95)), scatterpoints=1,
                labelspacing=0.8, loc='center right', bbox_to_anchor=(1.04, 0.32),
                ncol=1, frameon=False, fontsize=8)
    # Adjust plots and make maps a little bigger
    plt.subplots_adjust(left=0.02, right=0.98, top=1., bottom=0.)
    # Add inset maps
    for astate in states_shp.records():
        if astate.attributes['name'] == 'Alaska':
            # Alaska asthma
            # lonmin, lonmax, latmin, latmax
            map_extent = (-179.99, -130, 49, 73)
            axes_extent = (ax1.get_position().x0-0.05, ax1.get_position().y0-0.02,
                            0.16, 0.16)  # LLx, LLy, width, height
            geometry = astate.geometry
            add_insetmap(axes_extent, map_extent, '', astate.geometry,
                          lng_allmsa, lat_allmsa, asthmaburdenrated_allmsa, proj)
            # Alaska PM2.5
            axes_extent = (ax2.get_position().x0-0.05, ax2.get_position().y0-0.02,
                            0.16, 0.16)  # LLx, LLy, width, height
            add_insetmap(axes_extent, map_extent, '', astate.geometry,
                          lng_allmsa, lat_allmsa, pmburdenrated_allmsa, proj)
        elif astate.attributes['name'] == 'Hawaii':
            map_extent = (-162, -154, 18.75, 23)
            geometry = astate.geometry
            axes_extent = (ax1.get_position().x0+0.03, ax1.get_position().y0-0.01,
                            0.10, 0.10)
            add_insetmap(axes_extent, map_extent, '', astate.geometry,
                          lng_allmsa, lat_allmsa, asthmaburdenrated_allmsa, proj)
            axes_extent = (ax2.get_position().x0+0.03, ax2.get_position().y0-0.01,
                            0.10, 0.10)
            add_insetmap(axes_extent, map_extent, '', astate.geometry,
                          lng_allmsa, lat_allmsa, pmburdenrated_allmsa, proj)
        elif astate.attributes['name'] == 'PRI-00 (Puerto Rico aggregation)':
            map_extent = (-68., -65., 17.5, 18.8)
            geometry = astate.geometry
            axes_extent = (ax1.get_position().x0+0.09, ax1.get_position().y0-0.015,
                            0.10, 0.10)
            add_insetmap(axes_extent, map_extent, '', astate.geometry,
                          lng_allmsa, lat_allmsa, asthmaburdenrated_allmsa, proj)
            axes_extent = (ax2.get_position().x0+0.09, ax2.get_position().y0-0.015,
                            0.10, 0.10)
            add_insetmap(axes_extent, map_extent, '', astate.geometry,
                          lng_allmsa, lat_allmsa, pmburdenrated_allmsa, proj)
        elif astate.attributes['sr_adm0_a3'] == 'USA':
            geometry = astate.geometry
            for ax in [ax1, ax2]:
                ax.add_geometries([geometry], crs=ccrs.PlateCarree(),
                                  facecolor='#f2f2f2', lw=0.5, edgecolor='w', alpha=1.,
                                  zorder=0)
    plt.savefig(DIR_FIG+'fig3_REVISED.pdf', dpi=1000)
    return 

def fig4(burdents):
    """

    Returns
    -------
    None.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
    jitter = 0.04
    # Pollution concentrations and demographics for years of interest
    burden19 = burdents.loc[burdents['YEAR'] == '2015-2019']
    (pm25black19, no2black19, copdblack19, ihdblack19, lriblack19, dmblack19,
      lcblack19, stblack19, asthmablack19, allmortblack19, pm25white19, 
      no2white19, copdwhite19, ihdwhite19, lriwhite19, dmwhite19, lcwhite19, 
      stwhite19, asthmawhite19, allmortwhite19, pm25nh19, no2nh19, copdnh19, 
      ihdnh19, lrinh19, dmnh19, lcnh19, stnh19, asthmanh19, allmortnh19, 
      pm25h19, no2h19, copdh19, ihdh19, lrih19, dmh19, lch19, sth19, 
      asthmah19, allmorth19) = pollhealthdisparities(burden19)
    # Calculate most/least white pollutants and burdens as in Kerr et al. (2021)
    frac_white19 = ((burden19[['race_nh_white', 'race_h_white']].sum(axis=1)) /
        burden19['race_tot'])
    frac_hisp19 = (burden19['race_h']/burden19['race_tot'])
    mostwhite = burden19.iloc[np.where(frac_white19 >=
        np.nanpercentile(frac_white19, 90))]
    leastwhite = burden19.iloc[np.where(frac_white19 <=
        np.nanpercentile(frac_white19, 10))]
    mosthisp = burden19.iloc[np.where(frac_hisp19 >=
        np.nanpercentile(frac_hisp19, 90))]
    leasthisp = burden19.iloc[np.where(frac_hisp19 <=
        np.nanpercentile(frac_hisp19, 10))]
    # Pollutants averaged over all MSAs
    mostwhitepm25all = burden19.iloc[np.where(frac_white19 >=
        np.nanpercentile(frac_white19, 90))].PM25.mean()
    leastwhitepm25all = burden19.iloc[np.where(frac_white19 <=
        np.nanpercentile(frac_white19, 10))].PM25.mean()
    mostwhiteno2all = burden19.iloc[np.where(frac_white19 >=
        np.nanpercentile(frac_white19, 90))].NO2.mean()
    leastwhiteno2all = burden19.iloc[np.where(frac_white19 <=
        np.nanpercentile(frac_white19, 10))].NO2.mean()
    mosthisppm25all = burden19.iloc[np.where(frac_hisp19 >=
        np.nanpercentile(frac_hisp19, 90))].PM25.mean()
    leasthisppm25all = burden19.iloc[np.where(frac_hisp19 <=
        np.nanpercentile(frac_hisp19, 10))].PM25.mean()
    mosthispno2all = burden19.iloc[np.where(frac_hisp19 >=
        np.nanpercentile(frac_hisp19, 90))].NO2.mean()
    leasthispno2all = burden19.iloc[np.where(frac_hisp19 <=
        np.nanpercentile(frac_hisp19, 10))].NO2.mean()
    # Rates averaged over all MSAs
    mostwhiteasthmaall = agestandardize(burden19, mostwhite, 'PA')
    leastwhiteasthmaall = agestandardize(burden19, leastwhite, 'PA')
    mostwhiteihdall = agestandardize(burden19, mostwhite, 'IHD')
    leastwhiteihdall = agestandardize(burden19, leastwhite, 'IHD')
    mostwhitestall = agestandardize(burden19, mostwhite, 'ST')
    leastwhitestall = agestandardize(burden19, leastwhite, 'ST')
    mostwhitelcall = agestandardize(burden19, mostwhite, 'LC')
    leastwhitelcall = agestandardize(burden19, leastwhite, 'LC')
    mostwhitedmall = agestandardize(burden19, mostwhite, 'DM')
    leastwhitedmall = agestandardize(burden19, leastwhite, 'DM')
    mostwhitecopdall = agestandardize(burden19, mostwhite, 'COPD')
    leastwhitecopdall = agestandardize(burden19, leastwhite, 'COPD')
    mostwhitelriall = agestandardize(burden19, mostwhite, 'LRI')
    leastwhitelriall = agestandardize(burden19, leastwhite, 'LRI')
    mostwhitemortall = (mostwhiteihdall+mostwhitestall+mostwhitelcall+
        mostwhitedmall+mostwhitecopdall+mostwhitelriall)
    leastwhitemortall = (leastwhiteihdall+leastwhitestall+leastwhitelcall+
        leastwhitedmall+leastwhitecopdall+leastwhitelriall)
    mosthispasthmaall = agestandardize(burden19, mosthisp, 'PA')
    leasthispasthmaall = agestandardize(burden19, leasthisp, 'PA')
    mosthispihdall = agestandardize(burden19, mosthisp, 'IHD')
    leasthispihdall = agestandardize(burden19, leasthisp, 'IHD')
    mosthispstall = agestandardize(burden19, mosthisp, 'ST')
    leasthispstall = agestandardize(burden19, leasthisp, 'ST')
    mosthisplcall = agestandardize(burden19, mosthisp, 'LC')
    leasthisplcall = agestandardize(burden19, leasthisp, 'LC')
    mosthispdmall = agestandardize(burden19, mosthisp, 'DM')
    leasthispdmall = agestandardize(burden19, leasthisp, 'DM')
    mosthispcopdall = agestandardize(burden19, mosthisp, 'COPD') 
    leasthispcopdall = agestandardize(burden19, leasthisp, 'COPD') 
    mosthisplriall = agestandardize(burden19, mosthisp, 'LRI')
    leasthisplriall = agestandardize(burden19, leasthisp, 'LRI')
    mosthispmortall = (mosthispihdall+mosthispstall+mosthisplcall+
        mosthispdmall+mosthispcopdall+mosthisplriall)
    leasthispmortall = (leasthispihdall+leasthispstall+leasthisplcall+
        leasthispdmall+leasthispcopdall+leasthisplriall)
    
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
        print(msa)
        msas.append(msa)
        crosswalk_msa = crosswalk.loc[crosswalk['MSA Title']==msa]
        # Umlaut and accent screw with .loc
        if msa=='Mayagüez, PR':
            crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C3242']
        if msa=='San Germán, PR':
            crosswalk_msa = crosswalk.loc[crosswalk['MSA Code']=='C4190']
        geoids_msa = []
        for prefix in crosswalk_msa['County Code'].values:
            prefix = str(prefix).zfill(5)
            incounty = [x for x in geoids if x.startswith(prefix)]
            geoids_msa.append(incounty)
        geoids_msa = sum(geoids_msa, [])
        # MSA-specific tracts and demographics
        harm_imsa19 = burden19.loc[burden19.index.isin(geoids_msa)]
        frac_white19 = ((harm_imsa19[['race_nh_white', 'race_h_white']].sum(axis=1)) /
                        harm_imsa19['race_tot'])
        frac_hisp19 = (harm_imsa19['race_h_white']/harm_imsa19['race_tot'])
        mostwhite = harm_imsa19.iloc[np.where(frac_white19 >= 
            np.nanpercentile(frac_white19, 90))]
        leastwhite = harm_imsa19.iloc[np.where(frac_white19 <= 
            np.nanpercentile(frac_white19, 10))]
        mosthisp = harm_imsa19.iloc[np.where(frac_hisp19 >=
            np.nanpercentile(frac_hisp19, 90))]
        leasthisp = harm_imsa19.iloc[np.where(frac_hisp19 <=
            np.nanpercentile(frac_hisp19, 10))]
        # Concentrations and burdens for population subgroup extremes
        mostwhitepm25msa.append(mostwhite.PM25.mean())
        leastwhitepm25msa.append(leastwhite.PM25.mean())
        mostwhiteno2msa.append(mostwhite.NO2.mean())
        leastwhiteno2msa.append(leastwhite.NO2.mean())
        mostwhiteihdmsa.append(agestandardize(harm_imsa19, mostwhite, 'IHD'))
        leastwhiteihdmsa.append(agestandardize(harm_imsa19, leastwhite, 'IHD'))
        mostwhitestmsa.append(agestandardize(harm_imsa19, mostwhite, 'ST'))
        leastwhitestmsa.append(agestandardize(harm_imsa19, leastwhite, 'ST'))
        mostwhitelcmsa.append(agestandardize(harm_imsa19, mostwhite, 'LC'))
        leastwhitelcmsa.append(agestandardize(harm_imsa19, leastwhite, 'LC'))
        mostwhitedmmsa.append(agestandardize(harm_imsa19, mostwhite, 'DM'))
        leastwhitedmmsa.append(agestandardize(harm_imsa19, leastwhite, 'DM'))
        mostwhitecopdmsa.append(agestandardize(harm_imsa19, mostwhite, 'COPD'))
        leastwhitecopdmsa.append(agestandardize(harm_imsa19, leastwhite, 'COPD'))
        mostwhitelrimsa.append(agestandardize(harm_imsa19, mostwhite, 'LRI'))
        leastwhitelrimsa.append(agestandardize(harm_imsa19, leastwhite, 'LRI'))
        mostwhiteasthmamsa.append(agestandardize(harm_imsa19, mostwhite, 'PA'))
        leastwhiteasthmamsa.append(agestandardize(harm_imsa19, leastwhite, 'PA'))
        mostwhitemortmsa.append(mostwhitelrimsa+mostwhitecopdmsa+
            mostwhitedmmsa+mostwhitelcmsa+mostwhitestmsa+mostwhiteihdmsa)
        leastwhitemortmsa.append(leastwhitelrimsa+leastwhitecopdmsa+
            leastwhitedmmsa+leastwhitelcmsa+leastwhitestmsa+leastwhiteihdmsa)
        # Concentrations and burdens for ethnic groups
        mosthisppm25msa.append(mosthisp.PM25.mean())
        leasthisppm25msa.append(leasthisp.PM25.mean())
        mosthispno2msa.append(mosthisp.NO2.mean())
        leasthispno2msa.append(leasthisp.NO2.mean())
        mosthispihdmsa.append(agestandardize(harm_imsa19, mosthisp, 'IHD'))
        leasthispihdmsa.append(agestandardize(harm_imsa19, leasthisp, 'IHD'))
        mosthispstmsa.append(agestandardize(harm_imsa19, mosthisp, 'ST'))
        leasthispstmsa.append(agestandardize(harm_imsa19, leasthisp, 'ST'))
        mosthisplcmsa.append(agestandardize(harm_imsa19, mosthisp, 'LC'))
        leasthisplcmsa.append(agestandardize(harm_imsa19, leasthisp, 'LC'))
        mosthispdmmsa.append(agestandardize(harm_imsa19, mosthisp, 'DM'))
        leasthispdmmsa.append(agestandardize(harm_imsa19, leasthisp, 'DM'))
        mosthispcopdmsa.append(agestandardize(harm_imsa19, mosthisp, 'COPD'))
        leasthispcopdmsa.append(agestandardize(harm_imsa19, leasthisp, 'COPD'))
        mosthisplrimsa.append(agestandardize(harm_imsa19, mosthisp, 'LRI'))
        leasthisplrimsa.append(agestandardize(harm_imsa19, leasthisp, 'LRI'))
        mosthispasthmamsa.append(agestandardize(harm_imsa19, mosthisp, 'PA'))
        leasthispasthmamsa.append(agestandardize(harm_imsa19, leasthisp, 'PA'))
        mosthispmortmsa.append(mosthisplrimsa+mosthispcopdmsa+
            mosthispdmmsa+mosthisplcmsa+mosthispstmsa+mosthispihdmsa)
        leasthispmortmsa.append(leasthisplrimsa+leasthispcopdmsa+
            leasthispdmmsa+leasthisplcmsa+leasthispstmsa+leasthispihdmsa)
    
        # Population-weighted concentrations and burdens
        (pm25black19i, no2black19i, copdblack19i, ihdblack19i, lriblack19i,
          dmblack19i, lcblack19i, stblack19i, asthmablack19i, allmortblack19i, 
          pm25white19i, no2white19i, copdwhite19i, ihdwhite19i, lriwhite19i, 
          dmwhite19i, lcwhite19i, stwhite19i, asthmawhite19i, allmortwhite19i, 
          pm25nh19i, no2nh19i, copdnh19i, ihdnh19i, lrinh19i, dmnh19i, lcnh19i, 
          stnh19i, asthmanh19i, allmortnh19i, pm25h19i, no2h19i, copdh19i, ihdh19i,
          lrih19i, dmh19i, lch19i, sth19i, asthmah19i,
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
        hpwasthmamsa.append(asthmah19i)
        nhpwasthmamsa.append(asthmanh19i)
        hpwmortmsa.append(allmorth19i)
        nhpwmortmsa.append(allmortnh19i)
    
    # # # # For population-weighted
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot2grid((10, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((10, 2), (2, 0), rowspan=6)
    ax3 = plt.subplot2grid((10, 2), (8, 0), rowspan=2)
    ax4 = plt.subplot2grid((10, 2), (0, 1), rowspan=2)
    ax5 = plt.subplot2grid((10, 2), (2, 1), rowspan=6)
    ax6 = plt.subplot2grid((10, 2), (8, 1), rowspan=2)
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
        ax1.axhspan(ypos-0.75, ypos+0.75, alpha=0.3,
                    color='lightgrey', zorder=0)
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
    ax4.text(0.5, ypos+0.2, 'Non-Hispanic',
              va='center', color=color3, zorder=11)
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
    ax5.plot(reject_outliers(nhpwihdmsa), y1, '.', color=color3, alpha=0.05, 
        zorder=10)
    ax5.plot(reject_outliers(hpwihdmsa), y2, '.', color=color2, alpha=0.05, 
        zorder=10)
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
        ax.set_xlim([0, 14])
        ax.set_xticks([0, 3.5, 7, 10.5, 14])
        ax.set_xticklabels(['0', '', '7', '', '14'])
        ax.set_xlabel('Concentration [$\mathregular{\mu}$g m$^' +
                      '{\mathregular{-3}}$  |  ppbv]', loc='left')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0]*0.95, ylim[1]*1.05])
    for ax in [ax2, ax5]:
        ax.set_xlim([0, 16])
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_xticklabels(['0', '', '8', '', '16'])
        ax.set_xlabel('PM$_{\mathregular{2.5}}$-attributable mortality rate' +
                      ' [per 100,000 population]', loc='left')
    for ax in [ax3, ax6]:
        ax.set_xlim([0, 400])
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_xticklabels(['0', '', '200', '', '400'])
        ax.set_xlabel('NO$_{\mathregular{2}}$-attributable incidence rate' +
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
    plt.savefig(DIR_FIG+'figS8_REVISED.pdf', dpi=600)
    
    # # # # For subgroup extremes
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.subplot2grid((10, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((10, 2), (2, 0), rowspan=6)
    ax3 = plt.subplot2grid((10, 2), (8, 0), rowspan=2)
    ax4 = plt.subplot2grid((10, 2), (0, 1), rowspan=2)
    ax5 = plt.subplot2grid((10, 2), (2, 1), rowspan=6)
    ax6 = plt.subplot2grid((10, 2), (8, 1), rowspan=2)
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
    ax1.text(11, ypos-0.2, 'Least\nwhite',
              va='center', color=color2, zorder=11)
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
    ax4.text(11, ypos-0.2, 'Most\nHispanic',
              va='center', color=color2, zorder=11)
    ax4.text(1, ypos+0.2, 'Least\nHispanic',
              va='center', color=color3, zorder=11)
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
        ax.set_xlim([0, 14])
        ax.set_xticks([0, 3.5, 7, 10.5, 14])
        ax.set_xticklabels(['0', '', '7', '', '14'])
        ax.set_xlabel('Concentration [$\mathregular{\mu}$g m$^' +
            '{\mathregular{-3}}$  |  ppbv]', loc='left')
        ylim = ax.get_ylim()
        ax.set_ylim([ylim[0]*0.95, ylim[1]*1.05])
    for ax in [ax2, ax5]:
        ax.set_xlim([0, 16])
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_xticklabels(['0', '', '8', '', '16'])
        ax.set_xlabel('PM$_{\mathregular{2.5}}$-attributable mortality rate' +
            ' [per 100,000 population]', loc='left')
    for ax in [ax3, ax6]:
        ax.set_xlim([0, 400])
        ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_xticklabels(['0', '', '200', '', '400'])
        ax.set_xlabel('NO$_{\mathregular{2}}$-attributable incidence rate' +
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
    plt.savefig(DIR_FIG+'fig4_REVISED.pdf', dpi=1000)
    return

def fig5(burdents):
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
    import matplotlib.pyplot as plt
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
        print(year)
        yearst = '%d-%d' % (year-4, year)
        burdenty = burdents.loc[burdents['YEAR'] == yearst].copy(deep=True)
        # Define ethnoracial groups
        burdenty['fracwhite'] = ((burdenty[['race_nh_white', 'race_h_white'
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
        PM25_mosthisp.append(mosthisp.PM25.mean())
        PM25_leasthisp.append(leasthisp.PM25.mean())
        NO2_mosthisp.append(mosthisp.NO2.mean())
        NO2_leasthisp.append(leasthisp.NO2.mean())    
        # Age standardized burdens    
        mostwhitepm, leastwhitepm, mosthisppm, leasthisppm = [], [], [], []
        for pmendpoint in ['IHD', 'DM', 'ST', 'COPD', 'LRI', 'LC']:
            mostwhitepm.append(agestandardize(burdenty, mostwhite, pmendpoint))
            leastwhitepm.append(agestandardize(burdenty, leastwhite, pmendpoint))
            mosthisppm.append(agestandardize(burdenty, mosthisp, pmendpoint))
            leasthisppm.append(agestandardize(burdenty, leasthisp, pmendpoint))        
        mostwhitepm = np.sum(mostwhitepm)
        leastwhitepm = np.sum(leastwhitepm)
        mosthisppm = np.sum(mosthisppm)
        leasthisppm = np.sum(leasthisppm)    
        mostwhiteasthma = agestandardize(burdenty, mostwhite, 'PA')
        leastwhiteasthma = agestandardize(burdenty, leastwhite, 'PA')
        mosthispasthma = agestandardize(burdenty, mosthisp, 'PA')
        leasthispasthma = agestandardize(burdenty, leasthisp, 'PA')
        pd_mostwhite.append(mostwhitepm)
        pd_leastwhite.append(leastwhitepm)
        asthma_mostwhite.append(mostwhiteasthma)
        asthma_leastwhite.append(leastwhiteasthma)
        pd_mosthisp.append(mosthisppm)
        pd_leasthisp.append(leasthisppm)
        asthma_mosthisp.append(mosthispasthma)
        asthma_leasthisp.append(leasthispasthma)
        # Relative disparities
        PM25_race_relative.append(leastwhite.PM25.mean()/mostwhite.PM25.mean())
        NO2_race_relative.append(leastwhite.NO2.mean()/mostwhite.NO2.mean())
        PM25_ethnic_relative.append(mosthisp.PM25.mean()/leasthisp.PM25.mean())
        NO2_ethnic_relative.append(mosthisp.NO2.mean()/leasthisp.NO2.mean())
        pd_race_relative.append(leastwhitepm/mostwhitepm)
        asthma_race_relative.append(leastwhiteasthma/mostwhiteasthma)
        pd_ethnic_relative.append(mosthisppm/leasthisppm)
        asthma_ethnic_relative.append(mosthispasthma/leasthispasthma)
    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax1t = plt.subplot2grid((10, 2), (0, 0), rowspan=3)
    ax1b = plt.subplot2grid((10, 2), (3, 0), rowspan=2)
    ax2t = plt.subplot2grid((10, 2), (5, 0), rowspan=3)
    ax2b = plt.subplot2grid((10, 2), (8, 0), rowspan=2)
    ax3t = plt.subplot2grid((10, 2), (0, 1), rowspan=3)
    ax3b = plt.subplot2grid((10, 2), (3, 1), rowspan=2)
    ax4t = plt.subplot2grid((10, 2), (5, 1), rowspan=3)
    ax4b = plt.subplot2grid((10, 2), (8, 1), rowspan=2)
    years = np.arange(2010, 2020, 1)
    # Racial NO2-attributable pediatric asthma
    for i, year in enumerate(years):
        ax1t.vlines(x=year, ymin=asthma_mostwhite[i], ymax=asthma_leastwhite[i],
                    colors='darkgrey', ls='-', lw=1)
    ax1t.scatter(years, asthma_mostwhite, color=color3, zorder=10,
                 label='Most white')
    ax1t.scatter(years, asthma_leastwhite, color=color2, zorder=10,
                 label='Least white')
    # Text for first and last years
    ax1t.text(years[0], asthma_mostwhite[0]-30, '%d' %np.round(
        asthma_mostwhite[0]), ha='center', va='top', color=color3, fontsize=8)
    ax1t.text(years[0], asthma_leastwhite[0]+25, '%d' %np.round(
        asthma_leastwhite[0]), ha='center', va='bottom', color=color2, fontsize=8)
    ax1t.text(years[-1], asthma_mostwhite[-1]-30, '%d' %np.round(
        asthma_mostwhite[-1]), ha='center', va='top', color=color3, fontsize=8)
    ax1t.text(years[-1], asthma_leastwhite[-1]+25, '%d' %np.round(
        asthma_leastwhite[-1]), ha='center', va='bottom', color=color2, fontsize=8)
    ax1b.plot(years, asthma_race_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(asthma_race_relative):
        if i == 3:
            ax1b.text(years[i], asthma_race_relative[i]*1.065, '%.2f' % txt,
                      ha='center', fontsize=8, clip_on=False)
        else:
            ax1b.text(years[i], asthma_race_relative[i]*1.05, '%.2f' % txt,
                      ha='center', fontsize=8, clip_on=False)
    
    # Racial PM2.5-attributable premature mortality
    for i, year in enumerate(years):
        ax2t.vlines(x=year, ymin=pd_mostwhite[i], ymax=pd_leastwhite[i],
                    colors='darkgrey', ls='-', lw=1)
    ax2t.scatter(years, pd_mostwhite, color=color3, zorder=10,
                 label='Most white')
    ax2t.scatter(years, pd_leastwhite, color=color2, zorder=10,
                 label='Least white')
    ax2t.text(years[0], pd_mostwhite[0]-5, '%d' %np.round(pd_mostwhite[0]),
              ha='center', va='bottom', color=color3, fontsize=8)
    ax2t.text(years[0], pd_leastwhite[0]+4, '%d' %np.round(pd_leastwhite[0]),
              ha='center', va='top', color=color2, fontsize=8)
    ax2t.text(years[-1], pd_mostwhite[-1]-1.5, '%d' %np.round(pd_mostwhite[-1]),
              ha='center', va='top', color=color3, fontsize=8)
    ax2t.text(years[-1], pd_leastwhite[-1]+1.2, '%d' %np.round(pd_leastwhite[-1]),
              ha='center', va='bottom', color=color2, fontsize=8)
    ax2b.plot(years, pd_race_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(pd_race_relative):
        ax2b.text(years[i], pd_race_relative[i]*1.04, '%.2f' % txt,
                  fontsize=8, ha='center', clip_on=False)
    
    # Ethnic NO2-attributable pediatric asthma
    for i, year in enumerate(years):
        ax3t.vlines(x=year, ymin=asthma_leasthisp[i], ymax=asthma_mosthisp[i],
                    colors='darkgrey', ls='-', lw=1)
    ax3t.scatter(years, asthma_leasthisp, color=color3, zorder=10,
                 label='Least Hispanic')
    ax3t.scatter(years, asthma_mosthisp, color=color2, zorder=10,
                 label='Most Hispanic')
    ax3t.text(years[0], asthma_leasthisp[0]-30, '%d' %np.round(asthma_leasthisp[0]),
              ha='center', va='top', color=color3, fontsize=8)
    ax3t.text(years[0], asthma_mosthisp[0]+25, '%d' %np.round(asthma_mosthisp[0]),
              ha='center', va='bottom', color=color2, fontsize=8)
    ax3t.text(years[-1], asthma_leasthisp[-1]-30, '%d' %np.round(asthma_leasthisp[-1]),
              ha='center', va='top', color=color3, fontsize=8)
    ax3t.text(years[-1], asthma_mosthisp[-1]+25, '%d' %np.round(asthma_mosthisp[-1]),
              ha='center', va='bottom', color=color2, fontsize=8)
    ax3b.plot(years, asthma_ethnic_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(asthma_ethnic_relative):
        if i == 0:
            ax3b.text(years[i]-0.1, asthma_ethnic_relative[i]*1.03, '%.2f' % txt, fontsize=8,
                      ha='center', clip_on=False)
        elif i == 2:
            ax3b.text(years[i]+0.05, asthma_ethnic_relative[i]*1.028, '%.2f' % txt, fontsize=8,
                      ha='center', clip_on=False)
        else:
            ax3b.text(years[i], asthma_ethnic_relative[i]*1.023, '%.2f' % txt, fontsize=8,
                      ha='center', clip_on=False)
    
    # Ethnic PM2.5-attributable premature mortality
    for i, year in enumerate(years):
        ax4t.vlines(x=year, ymin=pd_leasthisp[i], ymax=pd_mosthisp[i],
            colors='darkgrey', ls='-', lw=1)
    ax4t.scatter(years, pd_leasthisp, color=color3, zorder=10,
        label='Least Hispanic')
    ax4t.scatter(years, pd_mosthisp, color=color2, zorder=10,
        label='Most Hispanic')
    ax4t.text(years[0], pd_leasthisp[0]+1.5, '%d' %np.round(pd_leasthisp[0]),
        ha='center', va='bottom', color=color3, fontsize=8)
    ax4t.text(years[0], pd_mosthisp[0]-1.9, '%d' %np.round(pd_mosthisp[0]),
        ha='center', va='top', color=color2, fontsize=8)
    ax4t.text(years[-1], pd_leasthisp[-1]-4.5, '%d' %np.round(pd_leasthisp[-1]),
        ha='center', va='bottom', color=color3, fontsize=8)
    ax4t.text(years[-1], pd_mosthisp[-1]+4.3, '%d' %np.round(pd_mosthisp[-1]),
              ha='center', va='top', color=color2, fontsize=8)
    ax4b.plot(years, pd_ethnic_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(pd_ethnic_relative):
        if i == 0:
            ax4b.text(years[i], pd_ethnic_relative[i]*1.1, '%.2f' % txt, fontsize=8,
                      ha='center', clip_on=False)
        elif i == 8:
            ax4b.text(years[i], pd_ethnic_relative[i]*1.05, '%.2f' % txt, fontsize=8,
                      ha='center', clip_on=False)
        else:
            ax4b.text(years[i], pd_ethnic_relative[i]*1.07, '%.2f' % txt, fontsize=8,
                ha='center', clip_on=False)
    # Aesthetics
    ax1t.set_title('(A) Racial disparities', fontsize=14, loc='left', y=1.07)
    ax2t.set_title('(C)', fontsize=14, loc='left', y=1.07)
    ax3t.set_title('(B) Ethnic disparities', fontsize=14, loc='left', y=1.07)
    ax4t.set_title('(D)', fontsize=14, loc='left', y=1.07)
    ax1t.set_ylabel('NO$_{\mathregular{2}}$-attributable pediatric\n asthma ' +
                    'cases [per 100000]')
    ax1t.get_yaxis().set_label_coords(-0.15, 0.5)
    ax2t.set_ylabel('PM$_{\mathregular{2.5}}$-attributable premature\ndeaths ' +
                    '[per 100000]')
    ax2t.get_yaxis().set_label_coords(-0.15, 0.5)
    # Axis limits
    for ax in [ax1t, ax3t]:
        ax.set_ylim([0, 500])
        ax.set_yticks(np.linspace(0, 500, 6))
        ax.set_yticklabels([])
    ax1t.set_yticklabels([int(x) for x in np.linspace(0, 500, 6)])
    for ax in [ax2t, ax4t]:
        ax.set_ylim([18, 46])
        ax.set_yticks(np.linspace(18, 46, 5))
        ax.set_yticklabels([])
    ax2t.set_yticklabels([int(x) for x in np.linspace(18, 46, 5)])
    # Relative disparities plots
    ax1b.set_ylim([min(asthma_race_relative)*0.92,
        max(asthma_race_relative)*1.03])
    ax2b.set_ylim([min(pd_race_relative)*0.92,
        max(pd_race_relative)*1.0])
    ax3b.set_ylim([min(asthma_ethnic_relative)*0.98,
        max(asthma_ethnic_relative)*1.0])
    ax4b.set_ylim([min(pd_ethnic_relative)*0.9,
        max(pd_ethnic_relative)*0.95])
    plt.subplots_adjust(wspace=0.2, hspace=15.5, bottom=0.08, top=0.92, 
        right=0.96)
    ax2t.legend(ncol=2, frameon=False, bbox_to_anchor=(0.95, -1.))
    ax4t.legend(ncol=2, frameon=False, bbox_to_anchor=(1.01, -1.))
    for ax in [ax1t, ax2t, ax3t, ax4t]:
        ax.set_xlim([2009.5, 2019.5])
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False)
    for ax in [ax1b, ax2b, ax3b, ax4b]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlim([2009.5, 2019.5])
        # Only draw spine between the y-ticks
        ax.spines.bottom.set_bounds((2010, 2019))
        ax.set_xticks(years)
        ax.set_xticklabels(['2010', '', '', '2013', '', '', '2016', '', '',
                            '2019'])
        ax.set_yticks([])
        # Move relative disparities subplots up
        box = ax.get_position()
        box.y0 = box.y0 + 0.04
        box.y1 = box.y1 + 0.04
        ax.set_position(box)
    plt.savefig(DIR_FIG+'fig5_REVISED.pdf', dpi=1000)
    return

def fig6(burdents):
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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    # Select burdens/sensitivity scenarios for 2019
    burden19 = pd.read_parquet(DIR_HARM+'burdens_%d_withsensitivity.gzip'%(year))
    # Population subgroups
    fracwhite = ((burden19[['race_nh_white', 'race_h_white']].sum(axis=1)) /
        burden19['race_tot'])
    burden19['fracwhite'] = fracwhite
    mostwhite = burden19.iloc[np.where(burden19.fracwhite >=
       np.nanpercentile(burden19.fracwhite, 90))]
    leastwhite = burden19.iloc[np.where(burden19.fracwhite <=
       np.nanpercentile(burden19.fracwhite, 10))]
    # Since PM2.5 concentrations for the NAAQS8 scenario were not saved off,
    # calculate them here
    burden19['PM25NAAQS8'] = burden19.PM25
    burden19.loc[burden19['PM25NAAQS8'] > 8., 'PM25NAAQS8'] = 8.
    
    # Names of PM2.5 and NO2 policy scenarios
    pmscenarios = ['', 'NAAQS12', 'WHO10', 'NAAQS8', 'WHO5']
    no2scenarios = ['', 'WHO40', 'WHO30', 'WHO20', 'WHO10']
    # Loop through scenarios and calculate burdens and PM2.5
    copd, lri, lc, ihd, dm, st = [], [], [], [], [], []
    pm25_pwm = []
    pd_mostwhite, pd_leastwhite = [], []
    for scenario in pmscenarios:
        pm25_pwm.append(w_avg(burden19, 'PM25'+scenario, 'pop_tot'))
        copd.append(burden19['BURDENCOPD'+scenario].sum())
        lri.append(burden19['BURDENLRI'+scenario].sum())
        lc.append(burden19['BURDENLC'+scenario].sum())
        ihd.append(burden19['BURDENIHD'+scenario].sum())
        dm.append(burden19['BURDENDM'+scenario].sum())
        st.append(burden19['BURDENST'+scenario].sum())
        mostwhitetemp, leastwhitetemp = [], []
        for pmendpoint in ['IHD', 'DM', 'ST', 'COPD', 'LRI', 'LC']:
            mostwhitetemp.append(agestandardize(burden19, mostwhite, 
                pmendpoint+scenario))
            leastwhitetemp.append(agestandardize(burden19, leastwhite, 
                pmendpoint+scenario))
        pd_mostwhite.append(np.sum(mostwhitetemp))
        pd_leastwhite.append(np.sum(leastwhitetemp))
    # Convert lists to arrays
    copd = np.array(copd)
    lri = np.array(lri)
    lc = np.array(lc)
    ihd = np.array(ihd)
    dm = np.array(dm)
    st = np.array(st)
    pm25_mean = np.array(pm25_pwm)
    # Same as above but for NO2-attributable pediatric asthma burdens
    asthma, asthma_mostwhite, asthma_leastwhite = [], [], []
    no2_pwm = []
    for scenario in no2scenarios:
        no2_pwm.append(w_avg(burden19, 'NO2'+scenario, 'pop_tot'))
        asthma.append(burden19['BURDENPA'+scenario].sum())
        asthma_mostwhite.append(agestandardize(burden19, mostwhite, 
            'PA'+scenario))
        asthma_leastwhite.append(agestandardize(burden19, leastwhite, 
            'PA'+scenario))
    asthma = np.array(asthma)
    no2_pwm = np.array(no2_pwm)
    asthma_leastwhite = np.array(asthma_leastwhite)
    asthma_mostwhite = np.array(asthma_mostwhite)
    # Initialize figure, axes
    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1t = ax1.twinx()
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax3 = plt.subplot2grid((2, 2), (0, 1))
    ax3t = ax3.twinx()
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    # PM25 and PM25-attributable mortality
    for i in np.arange(0, len(pmscenarios)):
        copdty = copd[i]
        lrity = lri[i]
        lcty = lc[i]
        ihdty = ihd[i]
        dmty = dm[i]
        stty = st[i]
        ax1.bar(i, ihdty, color=color1, bottom=0, zorder=10)
        ax1.bar(i, stty, color=color2, bottom=ihdty, zorder=10)
        ax1.bar(i, lcty, color=color3, bottom=ihdty+stty, zorder=10)
        ax1.bar(i, copdty, color=color4, bottom=ihdty+stty+lcty, zorder=10)
        ax1.bar(i, dmty, color=color5, bottom=ihdty+stty+lcty+copdty,
                zorder=10)
        ax1.bar(i, lrity, color=color6, bottom=ihdty+stty+lcty+copdty+dmty,
                zorder=10)
    ax1t.plot(np.arange(0, len(pmscenarios)), pm25_pwm, ls='-', marker='o',
              color='k', lw=2)
    # Racial PM2.5-attributable premature mortality
    for i, s in enumerate(pmscenarios):
        ax2.vlines(x=i, ymin=pd_mostwhite[i], ymax=pd_leastwhite[i],
                   colors='darkgrey', ls='-', lw=1)
        ax2.scatter(i, pd_mostwhite[i], color=color3, zorder=10,
                    label='Most white')
        ax2.scatter(i, pd_leastwhite[i], color=color2, zorder=10,
                    label='Least white')
        if i == 4:
            ax2.text(i, pd_mostwhite[i]+4, '%d' %np.round(pd_mostwhite[i]),
                     ha='center', va='top', color=color3, fontsize=10)
        else:
            ax2.text(i, pd_mostwhite[i]-5, '%d' %np.round(pd_mostwhite[i]),
                     ha='center', va='bottom', color=color3, fontsize=10)
        if i == 4:
            ax2.text(i, pd_leastwhite[i]-2.5, '%d' %np.round(pd_leastwhite[i]),
                     ha='center', va='top', color=color2, fontsize=10)
        else:
            ax2.text(i, pd_leastwhite[i]+4, '%d' %np.round(pd_leastwhite[i]),
                     ha='center', va='top', color=color2, fontsize=10)
    for i in np.arange(0, len(no2scenarios)):
        asthmaty = asthma[i]
        ax3.bar(i, asthmaty, color=color7, bottom=0, zorder=10)
    ax3t.plot(np.arange(0, len(no2scenarios)), no2_pwm, ls='-', marker='o',
              color='k', lw=2)
    # Racial NO2-attributable pediatric asthma
    for i, s in enumerate(no2scenarios):
        ax4.vlines(x=i, ymin=asthma_mostwhite[i], ymax=asthma_leastwhite[i],
            colors='darkgrey', ls='-', lw=1)
        ax4.scatter(i, asthma_mostwhite[i], color=color3, zorder=10,
            label='Most white', clip_on=False)
        ax4.scatter(i, asthma_leastwhite[i], color=color2, zorder=10,
            label='Least white', clip_on=False)
        ax4.text(i, asthma_mostwhite[i]-40, '%d' %np.round(asthma_mostwhite[i]),
            ha='center', va='bottom', color=color3, fontsize=10, clip_on=False)
        ax4.text(i, asthma_leastwhite[i]+40, '%d' %np.round(asthma_leastwhite[i]),
            ha='center', va='top', color=color2, fontsize=10, clip_on=False)
    # Aesthetics
    for ax in [ax1, ax1t, ax2, ax3, ax3t, ax4]:
        ax.set_xlim([-0.5, 4.5])
    for ax in [ax2, ax4]:
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False)
    # For ax1
    ax1.set_title('(A) PM$_{\mathregular{2.5}}$ standards', loc='left')
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_ylabel('PM$_{2.5}$-attributable\npremature deaths')
    ax1.set_ylim([0, 60000])
    ax1.set_yticks(np.linspace(0, 60000, 5))
    ax1.yaxis.set_label_coords(-0.26, 0.5)
    ax1t.set_ylim([0, 8])
    ax1t.set_yticks(np.linspace(0, 8, 5))
    ax1t.set_ylabel('PM$_{\mathregular{2.5}}$ [$\mathregular{\mu}$g m$' +
                    '^{\mathregular{-3}}$]', rotation=270)
    ax1t.yaxis.set_label_coords(1.2, 0.5)
    # For ax2
    ax2.set_ylim([2, 18])
    ax2.set_yticks(np.linspace(0, 30, 6))
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(['Base', 'EPA\nNAAQS', 'WHO\nIT-4',
                         'EPA\nCASAC', 'WHO\nAQG'], rotation=0, fontsize=9)
    ax2.tick_params(axis='x', pad=15)
    ax2.set_ylabel('PM$_{\mathregular{2.5}}$-attributable premature\ndeaths ' +
        '[per 100000]')
    ax2.get_yaxis().set_label_coords(-0.26, 0.5)
    # For ax3
    ax3.set_title('(B) NO$_{\mathregular{2}}$ standards', loc='left')
    ax3.set_xticks([])
    ax3.set_ylabel('NO$_{2}$-attributable\npediatric asthma')
    ax3.set_ylim([0, 120000])
    ax3.set_yticks(np.linspace(0, 120000, 5))
    ax3.yaxis.set_label_coords(-0.26, 0.5)
    ax3t.set_ylim([0, 9])
    ax3t.set_yticks(np.linspace(0, 9, 4))
    ax3t.set_ylabel('NO$_{\mathregular{2}}$ [ppbv]', rotation=270)
    ax3t.yaxis.set_label_coords(1.2, 0.5)
    # For ax4
    ax4.set_ylim([0, 18])
    ax4.set_yticks(np.linspace(0, 300, 6))
    ax4.set_xticks([0, 1, 2, 3, 4])
    ax4.set_xticklabels(['Base', 'WHO\nIT-4', 'WHO\nIT-3',
        'WHO\nIT-2', 'WHO\nAQG'], rotation=0, fontsize=9)
    ax4.tick_params(axis='x', pad=15)
    ax4.set_ylabel('NO$_{\mathregular{2}}$-attributable pediatric\n asthma ' +
        'cases [per 100000]')
    ax4.get_yaxis().set_label_coords(-0.26, 0.5)
    # Add legend denoting PM2.5 timeseries
    ax1t.annotate('PM$_{\mathregular{2.5}}$', xy=(4, pm25_pwm[-1]+0.3),
        xycoords='data', xytext=(3.3, pm25_pwm[-1]+1.8), textcoords='data', 
        arrowprops=dict(arrowstyle='->', color='k'), fontsize=12)
    ax3t.annotate('NO$_{\mathregular{2}}$', xy=(4, no2_pwm[-1]+0.2),
        xycoords='data', xytext=(3.7, no2_pwm[-1]+2.1), textcoords='data', 
        arrowprops=dict(arrowstyle='->', color='k'), fontsize=12)
    # # Add legend for different endpoints
    # pihd = mpatches.Patch(color=color1, label='Ischemic heart disease')
    # pst = mpatches.Patch(color=color2, label='Stroke')
    # plc = mpatches.Patch(color=color3, label='Lung cancer')
    # pcopd = mpatches.Patch(color=color4, label='COPD')
    # pdm = mpatches.Patch(color=color5, label='Type 2 diabetes')
    # plri = mpatches.Patch(color=color6, label='Lower respiratory infection')
    # ax1.legend(handles=[pihd, pst, plc, pcopd, pdm, plri],
    #     bbox_to_anchor=(1.3, -0.4), ncol=2, frameon=False)
    plt.subplots_adjust(hspace=0.2, wspace=0.85, bottom=0.12)
    plt.savefig(DIR_FIG+'fig6_REVISED.pdf', dpi=1000)
    return

def figS1():
    """
    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    cfpm = 4.15
    cfno2 = 5.3675
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
    afno2asthma.loc[afno2asthma < 0] = 0.004
    afno2asthmaupper.loc[afno2asthmaupper < 0] = 0.
    # afno2asthmalower.loc[afno2asthmalower<0]=0.
    afpmlri.loc[afpmlri < 0] = 0.004
    afpmdm.loc[afpmdm < 0] = 0.004
    afpmcopd.loc[afpmcopd < 0] = 0.004
    afpmlc.loc[afpmlc < 0] = 0.004
    afpmst.loc[afpmst < 0] = 0.004
    afpmihd.loc[afpmihd < 0] = 0.004
    # Plotting
    colorupper = '#FF7043'
    colorlower = '#0095A8'
    fig = plt.figure(figsize=(6, 8))
    ax1 = plt.subplot2grid((4, 2), (0, 0))
    ax2 = plt.subplot2grid((4, 2), (1, 0))
    ax3 = plt.subplot2grid((4, 2), (2, 0))
    ax4 = plt.subplot2grid((4, 2), (2, 1))
    ax5 = plt.subplot2grid((4, 2), (0, 1))
    ax6 = plt.subplot2grid((4, 2), (1, 1))
    ax7 = plt.subplot2grid((4, 2), (3, 0))
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
    ax7.plot(rrno2asthma.loc[rrno2asthma['exposure'] > cfno2]['exposure'],
             afno2asthmalower.values[54:]*100.,
             color=colorlower, ls='-', zorder=20, label='Lower')
    ax7.plot(rrno2asthma['exposure'], afno2asthmaupper.values*100.,
             color=colorupper, ls='-', zorder=20, label='Upper')
    ax7.set_xlabel('NO$_{\mathregular{2}}$ [ppbv]')
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim([0, 20])
        ax.set_xticks(np.linspace(0, 20, 5))
        ax.set_xticklabels([])
        ax.set_ylim([0, 25.])
        ax.set_yticks(np.linspace(0, 25, 6))
        ax.set_yticklabels([])
        ax.grid(axis='both', which='major', zorder=0, color='grey', ls='-',
                lw=0.25)
        for k, spine in ax.spines.items():
            spine.set_zorder(30)
        # Denote TMREL
        ax.axvspan(0, cfpm, alpha=1., color='lightgrey', zorder=10)
    for ax in [ax4, ax3]:
        ax.set_xlabel('PM$_{\mathregular{2.5}}$ [${\mu}$g m$' +
                      '^{\mathregular{-3}}$]')
        ax.set_xticklabels(['0', '5', '10', '15', '20'])
    for ax in [ax1, ax2, ax3]:
        ax.set_yticklabels(['0', '5', '10', '15', '20', '25'])
    fig.text(0.02, 0.5, 'Population attributable fraction [%]', va='center',
             rotation='vertical', fontsize=14)
    ax7.set_xlim([0, 40])
    ax7.set_xticks(np.linspace(0, 40, 5))
    ax7.set_ylim([-50, 50])
    ax7.set_yticks(np.linspace(-50, 50, 5))
    ax7.legend(ncol=1, frameon=False, bbox_to_anchor=(1.4, 1.1), fontsize=14)
    ax7.axvspan(0, cfno2, alpha=1., color='lightgrey', zorder=10)
    ax7.grid(axis='both', which='major', zorder=0,
             color='grey', ls='-', lw=0.25)
    for k, spine in ax7.spines.items():
        spine.set_zorder(30)
    plt.subplots_adjust(hspace=1., right=0.95)
    plt.savefig(DIR_FIG+'figS1.pdf', dpi=1000)
    return

def figS2():
    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from cartopy.io import shapereader
    import matplotlib
    import pandas as pd
    # Open GBD 2019 incidence rates
    ihd = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-cvd_ihd_agestrat.csv',
        engine='python', sep=',')
    stroke = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-cvd_stroke_agestrat.csv', 
        engine='python', sep=',')
    lri = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-lri_agestrat.csv',
        engine='python', sep=',')
    lung = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-neo_lung_agestrat.csv', 
        engine='python', sep=',')
    copd = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-resp_copd_agestrat.csv', 
        engine='python', sep=',')
    dm = pd.read_csv(DIR_GBD+'IHME-GBD_2019_DATA-t2_dm_agestrat.csv',
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
    pc_ihd = ((ihdyr.val.values[-1]-ihdyr.val.values[0]) /
              ihdyr.val.values[0])*100
    print('Percent change', pc_ihd, '%')
    print('\n')
    print('# # # # Nationwide stroke incidence trends')
    print('Slope/p-value = ', trend_strokeyr.slope, '/', trend_strokeyr.pvalue)
    pc_stroke = ((strokeyr.val.values[-1]-strokeyr.val.values[0]) /
                 strokeyr.val.values[0])*100
    print('Percent change', pc_stroke, '%')
    print('\n')
    print('# # # # Nationwide lower respiratory infection incidence trends')
    print('Slope/p-value = ', trend_lriyr.slope, '/', trend_lriyr.pvalue)
    
    pc_lri = ((lriyr.val.values[-1]-lriyr.val.values[0]) /
              lriyr.val.values[0])*100
    print('Percent change', pc_lri, '%')
    print('\n')
    print('# # # # Nationwide lung cancer incidence trends')
    print('Slope/p-value = ', trend_lungyr.slope, '/', trend_lungyr.pvalue)
    pc_lung = ((lungyr.val.values[-1]-lungyr.val.values[0]) /
               lungyr.val.values[0])*100
    print('Percent change', pc_lung, '%')
    print('\n')
    print('# # # # Nationwide COPD incidence trends')
    print('Slope/p-value = ', trend_copdyr.slope, '/', trend_copdyr.pvalue)
    pc_copd = ((copdyr.val.values[-1]-copdyr.val.values[0]) /
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
    pc_asthma = ((asthmayr.val.values[-1]-asthmayr.val.values[0]) /
                 asthmayr.val.values[0])*100
    print('Percent change', pc_asthma, '%')
    print('\n')
    # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural', 
        'admin_0_countries')
    reader = shapereader.Reader(shpfilename)
    countries = reader.records()
    usaidx = [x.attributes['ADM0_A3'] for x in countries]
    usaidx = np.where(np.in1d(np.array(usaidx), ['PRI', 'USA']) == True)
    usa = list(reader.geometries())
    usa = np.array(usa, dtype=object)[usaidx[0]]
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shapereader.Reader(lakes)
    lakes = lakes_reader.records()
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names) == 'Lake Superior') |
        (np.array(lake_names) == 'Lake Michigan') | (np.array(lake_names) 
        == 'Lake Huron') | (np.array(lake_names) == 'Lake Erie') |
        (np.array(lake_names) == 'Lake Ontario'))[0]
    great_lakes = np.array(list(lakes_reader.geometries()), 
        dtype=object)[great_lakes]
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shapereader.natural_earth(resolution='10m', 
        category='cultural', name=shapename)
    states_shp = shapereader.Reader(states_shp)
    # Find shapefiles of states of interest for inset maps
    states_all = list(states_shp.records())
    states_all_name = []
    for s in states_all:
        states_all_name.append(s.attributes['name'])
    states_all = np.array(states_all)
    alaska = states_all[np.where(np.array(states_all_name) == 'Alaska')[0]][0]
    hawaii = states_all[np.where(np.array(states_all_name) == 'Hawaii')[0]][0]
    puertorico = states_all[np.where(np.array(states_all_name) ==
        'PRI-00 (Puerto Rico aggregation)')[0]][0]
    # Define colorscheme
    cmap = plt.get_cmap('magma_r', 7)
    normlri = matplotlib.colors.Normalize(vmin=40, vmax=90)
    normstroke = matplotlib.colors.Normalize(vmin=100, vmax=200)
    normdm = matplotlib.colors.Normalize(vmin=20, vmax=70)
    normihd = matplotlib.colors.Normalize(vmin=300, vmax=600)
    normcopd = matplotlib.colors.Normalize(vmin=100, vmax=200)
    normlung = matplotlib.colors.Normalize(vmin=50, vmax=200)
    normasthma = matplotlib.colors.Normalize(vmin=2000, vmax=4000)
    # Define endpoint order
    endpoints = [lri, stroke, dm, ihd, copd, lung, asthma]
    norms = [normlri, normstroke, normdm, normihd, normcopd, normlung, 
        normasthma]
    endpoints_names = ['Lower respir-\natory infection', 'Stroke\n',
        'Type 2\ndiabetes', 'Ischemic\nheart disease', 'COPD\n', 
        'Lung cancer\n', 'Pediatric\nasthma']
    # Initialize figure, subplots
    proj = ccrs.PlateCarree()
    fig, axs = plt.subplots(7,3, figsize=(8.5,11), subplot_kw=dict(
        projection=proj))
    plt.subplots_adjust(left=0.12, top=0.95, bottom=0.05)
    for i, ax in enumerate(axs.flatten()):
        # Select endpoint and year
        endpointloc = int(np.round((i/3)-(1./3)))
        endpoint = endpoints[endpointloc]
        endpoint_name = endpoints_names[endpointloc]
        if i in [0, 3, 6, 9, 12, 15, 18]:
            year = 2010
        elif i in [1, 4, 7, 10, 13, 16, 19]:
            year = 2015
        else:
            year = 2019
        endpoint = endpoint.loc[endpoint.year == year]
        # Add endpoint name as y-label
        if i in [0, 3, 6, 9, 12, 15, 18]:
            ax.text(-0.2, -0.05, endpoint_name, va='bottom', ha='left',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=14)
        ax.set_extent([-125, -66.5, 24.5, 49.48], proj)
        ax.add_geometries(usa, crs=proj, lw=0.25, facecolor='None',
            edgecolor='k', zorder=1)
        ax.add_geometries(great_lakes, crs=proj, facecolor='w', lw=0.25,
            edgecolor='k', alpha=1., zorder=12)
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
            edgecolor='None', facecolor='w', alpha=1.), zorder=11)
        ax.axis('off')
        # Add colorbar
        if i in [2, 5, 8, 11, 14, 17, 20]:
            cax = fig.add_axes([axs.flatten()[i].get_position().x1+0.015,
                axs.flatten()[i].get_position().y0, 0.012, (axs.flatten(
                )[i].get_position().y1 -axs.flatten()[i].get_position().y0)])
            mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norms[endpointloc],
                orientation='vertical', extend='both', label='', 
                ticks=np.linspace(norms[endpointloc].vmin,
                norms[endpointloc].vmax, 3))
        # Color states by incidence rates and add state borders
        for astate in states_shp.records():
            if astate.attributes['name'] in ['Alaska', 'Hawaii',
                'PRI-00 (Puerto Rico aggregation)']:
                # Parse information about rates in states
                if astate.attributes['name'] == 'Alaska':
                    fcak = endpoint.loc[endpoint['location_name'] == 'Alaska']
                if astate.attributes['name'] == 'Hawaii':
                    fchi = endpoint.loc[endpoint['location_name'] == 'Hawaii']
                if astate.attributes['name'] == 'PRI-00 (Puerto Rico aggregation)':
                    fcpr = endpoint.loc[endpoint['location_name']
                                        == 'Puerto Rico']
                pass
            elif astate.attributes['sr_adm0_a3'] == 'USA':
                inrate = endpoint.loc[endpoint['location_name'] ==
                    astate.attributes['name']]
                geometry = astate.geometry
                ax.add_geometries([geometry], crs=proj, zorder=100, ec='k',
                    fc=cmap(norms[endpointloc](inrate.val.mean())), lw=0.25)
        # Hawaii
        axes_extent = (ax.get_position().x0-0.0, ax.get_position().y0-0.015,
            (ax.get_position().x1-ax.get_position().x0)*0.22, (ax.get_position(
            ).x1-ax.get_position().x0)*0.22)
        add_insetmap(axes_extent, (-162, -154, 18.75, 23), '', hawaii.geometry,
            [0.], [0.], [0.], proj, fc=matplotlib.colors.to_hex(cmap(
            norms[endpointloc](fchi.val.mean()))))
        # Alaska
        axes_extent = (ax.get_position().x0-0.04, ax.get_position().y0-0.01,
            (ax.get_position().x1-ax.get_position().x0)*0.22, (ax.get_position(
            ).x1-ax.get_position().x0)*0.22)
        add_insetmap(axes_extent, (-179.99, -130, 49, 73), '', alaska.geometry,
            [0.], [0.], [0.], proj, fc=matplotlib.colors.to_hex(cmap(
            norms[endpointloc](fcak.val.mean()))))
        # Puerto Rico
        axes_extent = (ax.get_position().x0+0.048, ax.get_position().y0-0.015,
            (ax.get_position().x1-ax.get_position().x0)*0.16, (ax.get_position(
            ).x1-ax.get_position().x0)*0.16)
        add_insetmap(axes_extent, (-68., -65., 17.5, 19.), '', 
            puertorico.geometry, [0.], [0.], [0.], proj, fc=\
            matplotlib.colors.to_hex(cmap(norms[endpointloc](fcpr.val.mean()))))
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
    plt.savefig(DIR_FIG+'figS2.png', dpi=1000)
    return

def figS3(burdents):
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
    harm2010 = burdents.loc[(burdents.YEAR == '2006-2010')]
    harm2015 = burdents.loc[(burdents.YEAR == '2011-2015')]
    harm2019 = burdents.loc[(burdents.YEAR == '2015-2019')]
    # Load shapefiles
    shpfilename = shapereader.natural_earth('10m', 'cultural',
                                            'admin_0_countries')
    reader = shapereader.Reader(shpfilename)
    countries = reader.records()
    usa = [x.attributes['ADM0_A3'] for x in countries]
    # usaidx = np.where(np.in1d(np.array(usaidx), ['PRI','USA'])==True)
    # usa = list(reader.geometries())
    # usa = np.array(usa, dtype=object)[usaidx[0]]
    usa = np.where(np.array(usa) == 'USA')[0][0]
    usa = list(reader.geometries())[usa].geoms
    lakes = shapereader.natural_earth('10m', 'physical', 'lakes')
    lakes_reader = shapereader.Reader(lakes)
    lakes = lakes_reader.records()
    lake_names = [x.attributes['name'] for x in lakes]
    great_lakes = np.where((np.array(lake_names) == 'Lake Superior') |
                           (np.array(lake_names) == 'Lake Michigan') |
                           (np.array(lake_names) == 'Lake Huron') |
                           (np.array(lake_names) == 'Lake Erie') |
                           (np.array(lake_names) == 'Lake Ontario'))[0]
    great_lakes = itemgetter(*great_lakes)(list(lakes_reader.geometries()))
    shapename = 'admin_1_states_provinces_lakes_shp'
    states_shp = shapereader.natural_earth(resolution='10m', category='cultural',
                                           name=shapename)
    states_shp = shapereader.Reader(states_shp)
    # Constants
    proj = ccrs.PlateCarree(central_longitude=0.0)
    # Initialize figure, subplots
    fig = plt.figure(figsize=(12, 4.25))
    ax1 = plt.subplot2grid((2, 3), (0, 0), projection=proj)
    ax2 = plt.subplot2grid((2, 3), (0, 1), projection=proj)
    ax3 = plt.subplot2grid((2, 3), (0, 2), projection=proj)
    ax4 = plt.subplot2grid((2, 3), (1, 0), projection=proj)
    ax5 = plt.subplot2grid((2, 3), (1, 1), projection=proj)
    ax6 = plt.subplot2grid((2, 3), (1, 2), projection=proj)
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
    for FIPS_i in FIPS:
        print(FIPS_i)
        # Tigerline shapefile for state
        shp = shapereader.Reader(DIR_GEO+'tract_2010/'+
            'tl_2019_%s_tract/tl_2019_%s_tract.shp' % (FIPS_i, FIPS_i))
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
            if harm2010_tract.empty == True:
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
            if harm2015_tract.empty == True:
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
            if harm2019_tract.empty == True:
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
        ax.set_extent([-125, -66.5, 24.5, 49.48], proj)
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
            elif astate.attributes['sr_adm0_a3'] == 'USA':
                geometry = astate.geometry
                ax.add_geometries([geometry], crs=proj, facecolor='None',
                    lw=0.25, edgecolor='k', zorder=100)
    # Find shapefiles of states of interest
    states_all = list(states_shp.records())
    states_all_name = []
    for s in states_all:
        states_all_name.append(s.attributes['name'])
    states_all = np.array(states_all)
    alaska = states_all[np.where(np.array(states_all_name) == 'Alaska')[0]][0]
    hawaii = states_all[np.where(np.array(states_all_name) == 'Hawaii')[0]][0]
    puertorico = states_all[np.where(np.array(states_all_name) ==
                                     'PRI-00 (Puerto Rico aggregation)')[0]][0]
    # Select harmonized dataset in states in inset maps
    alaska2010 = burdents.loc[(burdents.YEAR == '2006-2010') &
                              (burdents.STATE == 'Alaska')]
    alaska2014 = burdents.loc[(burdents.YEAR == '2010-2014') &
                              (burdents.STATE == 'Alaska')]
    alaska2015 = burdents.loc[(burdents.YEAR == '2011-2015') &
                              (burdents.STATE == 'Alaska')]
    alaska2017 = burdents.loc[(burdents.YEAR == '2013-2017') &
                              (burdents.STATE == 'Alaska')]
    alaska2019 = burdents.loc[(burdents.YEAR == '2015-2019') &
                              (burdents.STATE == 'Alaska')]
    hawaii2010 = burdents.loc[(burdents.YEAR == '2006-2010') &
                              (burdents.STATE == 'Hawaii')]
    hawaii2014 = burdents.loc[(burdents.YEAR == '2010-2014') &
                              (burdents.STATE == 'Hawaii')]
    hawaii2015 = burdents.loc[(burdents.YEAR == '2011-2015') &
                              (burdents.STATE == 'Hawaii')]
    hawaii2017 = burdents.loc[(burdents.YEAR == '2013-2017') &
                              (burdents.STATE == 'Hawaii')]
    hawaii2019 = burdents.loc[(burdents.YEAR == '2015-2019') &
                              (burdents.STATE == 'Hawaii')]
    puertorico2010 = burdents.loc[(burdents.YEAR == '2006-2010') &
                                  (burdents.STATE == 'Puerto Rico')]
    puertorico2014 = burdents.loc[(burdents.YEAR == '2010-2014') &
                                  (burdents.STATE == 'Puerto Rico')]
    puertorico2015 = burdents.loc[(burdents.YEAR == '2011-2015') &
                                  (burdents.STATE == 'Puerto Rico')]
    puertorico2017 = burdents.loc[(burdents.YEAR == '2013-2017') &
                                  (burdents.STATE == 'Puerto Rico')]
    puertorico2019 = burdents.loc[(burdents.YEAR == '2015-2019') &
                                  (burdents.STATE == 'Puerto Rico')]
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
        add_insetmap(axes_extent, (-68., -65., 17.5, 19.), '', 
            puertorico.geometry, [0.], [0.], [0.], proj, fips='72', 
            harmonized=harm[2], vara=vara, cmap=cmap, norm=norm)
    # Add colorbars
    caxno2 = fig.add_axes([ax3.get_position().x1+0.01, ax3.get_position().y0, 
        0.01, (ax3.get_position().y1-ax3.get_position().y0)])
    mpl.colorbar.ColorbarBase(caxno2, cmap=cmapno2, norm=normno2,
        spacing='proportional', orientation='vertical', extend='max',
        label='[ppbv]', ticks=np.linspace(0, 24, 5))
    caxpm = fig.add_axes([ax6.get_position().x1+0.01, ax6.get_position().y0, 
        0.01, (ax6.get_position().y1-ax6.get_position().y0)])
    mpl.colorbar.ColorbarBase(caxpm, cmap=cmappm, norm=normpm,
        spacing='proportional', orientation='vertical', extend='max',
        label='[$\mathregular{\mu}$g m$^{\mathregular{-3}}$]', 
        ticks=np.linspace(0, 12, 5))
    plt.savefig(DIR_FIG+'figS3.png', dpi=1000)
    return

def figS4(burdents):
    """

    Parameters
    ----------
    burdents : TYPE
        DESCRIPTION.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    burden19 = burdents.loc[burdents['YEAR']=='2015-2019'].copy(deep=True)
    # Define ethnoracial groups
    burden19['fracwhite'] = ((burden19[['race_nh_white','race_h_white'
        ]].sum(axis=1))/burden19['race_tot'])
    burden19['frachisp'] = (burden19['race_h']/burden19['race_tot'])
    # Define extreme ethnoracial subgroups
    mostwhite = burden19.iloc[np.where(burden19.fracwhite >=
        np.nanpercentile(burden19.fracwhite, 90))]
    leastwhite = burden19.iloc[np.where(burden19.fracwhite <=
        np.nanpercentile(burden19.fracwhite, 10))]
    mosthisp = burden19.iloc[np.where(burden19.frachisp >
        np.nanpercentile(burden19.frachisp, 90))]
    leasthisp = burden19.iloc[np.where(burden19.frachisp <
        np.nanpercentile(burden19.frachisp, 10))]
    # Lists to be filled with age fractions
    mostwhite_af, leastwhite_af, mosthisp_af, leasthisp_af = [], [], [], []
    for age in [['pop_m_lt5','pop_f_lt5'],
        ['pop_m_5-9','pop_f_5-9'],
        ['pop_m_10-14','pop_f_10-14'],
        ['pop_m_15-17','pop_m_18-19','pop_f_15-17','pop_f_18-19'],
        ['pop_m_20','pop_m_21','pop_m_22-24','pop_f_20','pop_f_21','pop_f_22-24',],
        ['pop_m_25-29','pop_f_25-29'],
        ['pop_m_30-34','pop_f_30-34'],
        ['pop_m_35-39','pop_f_35-39'],
        ['pop_m_40-44','pop_f_40-44'],
        ['pop_m_45-49','pop_f_45-49'],
        ['pop_m_50-54','pop_f_50-54'],
        ['pop_m_55-59','pop_f_55-59'],
        ['pop_m_60-61','pop_m_62-64','pop_f_60-61','pop_f_62-64'],
        ['pop_m_65-66', 'pop_m_67-69','pop_f_65-66', 'pop_f_67-69'],
        ['pop_m_70-74','pop_f_70-74'],
        ['pop_m_75-79','pop_f_75-79'],
        ['pop_m_80-84','pop_f_80-84'],
        ['pop_m_gt85','pop_f_gt85']]:
        mostwhite_af.append((mostwhite[age].sum(axis=1)/
            mostwhite.pop_tot).mean())
        leastwhite_af.append((leastwhite[age].sum(axis=1)/
            leastwhite.pop_tot).mean())
        mosthisp_af.append((mosthisp[age].sum(axis=1)/
            mosthisp.pop_tot).mean())
        leasthisp_af.append((leasthisp[age].sum(axis=1)/
            leasthisp.pop_tot).mean())
    fig = plt.figure(figsize=(7, 8.5))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    # Titles
    ax1.set_title('(A) Most white', loc='left')
    ax2.set_title('(B) Least white', loc='left')
    ax3.set_title('(C) Least Hispanic', loc='left')
    ax4.set_title('(D) Most Hispanic', loc='left')
    # Create horizontal bars
    y_pos = np.arange(len(mostwhite_af))
    ax1.barh(y_pos, mostwhite_af, color=color3)
    ax2.barh(y_pos, leastwhite_af, color=color2)
    ax3.barh(y_pos, leasthisp_af, color=color3)
    ax4.barh(y_pos, mosthisp_af, color=color2)
    ylab = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34',
            '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
            '70-74', '75-79', '80-84', '$\geq$85']
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([0, 0.09])
        ax.set_xticks([0, 0.015, 0.03, 0.045, 0.06, 0.075, 0.09])
        ax.set_xticklabels(['0.0', '', '0.03', '', '0.06', '', '0.09'])
        ax.set_ylim([-0.5, len(mostwhite_af)-0.5])
        ax.set_yticks(np.arange(0, len(mostwhite_af)))
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Draw shared regions for premature mortality/pediatric asthma
        # ax.axhspan(4.5, 17.5, alpha=0.5, color='gainsboro', zorder=0)
        ax.axhspan(-0.5, 3.5, alpha=0.5, color='wheat', zorder=0)
    # Denote shaded regions
    # ax1.text(0.088, 17, 'Premature\nmortality\nsusceptible', color='grey',
              # ha='right', va='top')
    ax1.text(0.088, 3., 'Pediatric\nasthma\nsusceptible',
              color='darkorange', ha='right', va='top')
    ax1.set_yticklabels(ylab)
    ax3.set_yticklabels(ylab)
    ax1.set_ylabel('Age [years]')
    ax3.set_xlabel('Proportion of population')
    ax3.set_ylabel('Age [years]')
    ax4.set_xlabel('Proportion of population')
    plt.subplots_adjust(top=0.95, bottom=0.08)
    plt.savefig(DIR_FIG+'figS4.pdf', dpi=1000)
    return

def (burdents):
    """

    Returns
    -------
    None
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
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
    # # # Burdens in 2015 calculated as in main text 
    burden = burdents.loc[burdents.YEAR=='2011-2015'].copy(deep=True)
    # Burdens using Turner et al. RR and GBD incidence for ages 30+
    burdent = edf_calculate.calculate_pm25burden_turner()
    # Burdens using Turner et al. RR and IEc incidence rates for ages 25+ 
    burdenti = edf_calculate.calculate_pm25burden_turneriec()
    # Calculate most/least white as in Kerr et al. (2021)
    burden['FRACWHITE'] = ((burden[['race_nh_white', 'race_h_white']
        ].sum(axis=1))/burden['race_tot'])
    burden['FRACHISP'] = (burden['race_h']/burden['race_tot'])
    burdent['FRACWHITE'] = ((burdent[['race_nh_white', 'race_h_white']
        ].sum(axis=1))/burdent['race_tot'])
    burdent['FRACHISP'] = (burdent['race_h']/burdent['race_tot'])
    burdenti['FRACWHITE'] = ((burdenti[['race_nh_white', 'race_h_white']
        ].sum(axis=1))/burdenti['race_tot'])
    burdenti['FRACHISP'] = (burdenti['race_h']/burdenti['race_tot'])
    # Population subgroups
    mostwhite = burden.loc[burden['FRACWHITE'] >=
        np.nanpercentile(burden['FRACWHITE'], 90)]
    leastwhite = burden.loc[burden['FRACWHITE'] <=
        np.nanpercentile(burden['FRACWHITE'], 10)]
    mosthisp = burden.loc[burden['FRACHISP'] >=
        np.nanpercentile(burden['FRACHISP'], 90)]
    leasthisp = burden.loc[burden['FRACHISP'] <=
        np.nanpercentile(burden['FRACHISP'], 10)]
    mostwhitet = burdent.loc[burdent['FRACWHITE'] >=
        np.nanpercentile(burdent['FRACWHITE'], 90)]
    leastwhitet = burdent.loc[burdent['FRACWHITE'] <=
        np.nanpercentile(burdent['FRACWHITE'], 10)]
    mosthispt = burdent.loc[burdent['FRACHISP'] >=
        np.nanpercentile(burdent['FRACHISP'], 90)]
    leasthispt = burdent.loc[burdent['FRACHISP'] <=
        np.nanpercentile(burdent['FRACHISP'], 10)]
    mostwhiteti = burdenti.loc[burdenti['FRACWHITE'] >=
        np.nanpercentile(burdenti['FRACWHITE'], 90)]
    leastwhiteti = burdenti.loc[burdenti['FRACWHITE'] <=
        np.nanpercentile(burdenti['FRACWHITE'], 10)]
    mosthispti = burdenti.loc[burdenti['FRACHISP'] >=
        np.nanpercentile(burdenti['FRACHISP'], 90)]
    leasthispti = burdenti.loc[burdenti['FRACHISP'] <=
        np.nanpercentile(burdenti['FRACHISP'], 10)]
    # Burdens from the main text
    leastwhite_rate = (agestandardize(burden, leastwhite, 'IHD')+
        agestandardize(burden, leastwhite, 'ST')+
        agestandardize(burden, leastwhite, 'LC')+
        agestandardize(burden, leastwhite, 'DM')+
        agestandardize(burden, leastwhite, 'COPD')+
        agestandardize(burden, leastwhite, 'LRI'))
    mostwhite_rate = (agestandardize(burden, mostwhite, 'IHD')+
        agestandardize(burden, mostwhite, 'ST')+
        agestandardize(burden, mostwhite, 'LC')+
        agestandardize(burden, mostwhite, 'DM')+
        agestandardize(burden, mostwhite, 'COPD')+
        agestandardize(burden, mostwhite, 'LRI'))
    leasthisp_rate = (agestandardize(burden, leasthisp, 'IHD')+
        agestandardize(burden, leasthisp, 'ST')+
        agestandardize(burden, leasthisp, 'LC')+
        agestandardize(burden, leasthisp, 'DM')+
        agestandardize(burden, leasthisp, 'COPD')+
        agestandardize(burden, leasthisp, 'LRI'))
    mosthisp_rate = (agestandardize(burden, mosthisp, 'IHD')+
        agestandardize(burden, mosthisp, 'ST')+
        agestandardize(burden, mosthisp, 'LC')+
        agestandardize(burden, mosthisp, 'DM')+
        agestandardize(burden, mosthisp, 'COPD')+
        agestandardize(burden, mosthisp, 'LRI'))
    leastwhitet_rate = agestandardize(burden, leastwhitet, 'AC')
    mostwhitet_rate = agestandardize(burden, mostwhitet, 'AC')
    leasthispt_rate = agestandardize(burden, leasthispt, 'AC')
    mosthispt_rate = agestandardize(burden, mosthispt, 'AC')
    leastwhiteti_rate = agestandardize(burden, leastwhiteti, 'IEc')
    mostwhiteti_rate = agestandardize(burden, mostwhiteti, 'IEc')
    leasthispti_rate = agestandardize(burden, leasthispti, 'IEc')
    mosthispti_rate = agestandardize(burden, mosthispti, 'IEc')
    # Rates but no age standardization 
    leastwhite_rate_noas = (leastwhite.BURDENIHDRATE.mean()+
        leastwhite.BURDENSTRATE.mean()+leastwhite.BURDENLCRATE.mean()+
        leastwhite.BURDENDMRATE.mean()+leastwhite.BURDENCOPDRATE.mean()+
        leastwhite.BURDENLRIRATE.mean())
    mostwhite_rate_noas = (mostwhite.BURDENIHDRATE.mean()+
        mostwhite.BURDENSTRATE.mean()+mostwhite.BURDENLCRATE.mean()+
        mostwhite.BURDENDMRATE.mean()+mostwhite.BURDENCOPDRATE.mean()+
        mostwhite.BURDENLRIRATE.mean())
    leasthisp_rate_noas = (leasthisp.BURDENIHDRATE.mean()+
        leasthisp.BURDENSTRATE.mean()+leasthisp.BURDENLCRATE.mean()+
        leasthisp.BURDENDMRATE.mean()+leasthisp.BURDENCOPDRATE.mean()+
        leasthisp.BURDENLRIRATE.mean())
    mosthisp_rate_noas = (mosthisp.BURDENIHDRATE.mean()+
        mosthisp.BURDENSTRATE.mean()+mosthisp.BURDENLCRATE.mean()+
        mosthisp.BURDENDMRATE.mean()+mosthisp.BURDENCOPDRATE.mean()+
        mosthisp.BURDENLRIRATE.mean())
    leastwhitet_rate_noas = leastwhitet.BURDENACRATE.mean()
    mostwhitet_rate_noas = mostwhitet.BURDENACRATE.mean()
    leasthispt_rate_noas = leasthispt.BURDENACRATE.mean()
    mosthispt_rate_noas = mosthispt.BURDENACRATE.mean()
    leastwhiteti_rate_noas = leastwhiteti.BURDENACRATE.mean()
    mostwhiteti_rate_noas = mostwhiteti.BURDENACRATE.mean()
    leasthispti_rate_noas = leasthispti.BURDENACRATE.mean()
    mosthispti_rate_noas = mosthispti.BURDENACRATE.mean()
    # 708297	8897a8	a0abb9	b8c0cb	cfd5dc	e7eaee	ffffff
    # ffa98e	ffb8a1	ffc6b4	ffd4c7	ffe2d9	fff1ec
    color2light = '#8897a8'
    color3light = '#ffb8a1'
    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot2grid((1,2),(0,0))
    ax2 = plt.subplot2grid((1,2),(0,1))
    # Rates from manuscript
    ax1.scatter(leastwhite_rate, 1, color=color2, label='Least white')
    ax1.scatter(mostwhite_rate, 1, color=color3, label='Most white')
    ax1.scatter(leastwhitet_rate, 2, color=color2)
    ax1.scatter(mostwhitet_rate, 2, color=color3)
    ax1.scatter(leastwhiteti_rate, 3, color=color2)
    ax1.scatter(mostwhiteti_rate, 3, color=color3)
    # Non-age standardized rates from manuscript
    ax1.scatter(leastwhite.BURDENPMALLRATE.mean(), 1.4, color=color2light)
    ax1.scatter(mostwhite.BURDENPMALLRATE.mean(), 1.4, color=color3light)
    ax1.scatter(leastwhitet_rate_noas, 2.4, color=color2light)
    ax1.scatter(mostwhitet_rate_noas, 2.4, color=color3light)
    ax1.scatter(leastwhiteti_rate_noas, 3.4, color=color2light)
    ax1.scatter(mostwhiteti_rate_noas, 3.4, color=color3light)
    # Same as above but for ethnicity
    ax2.scatter(mosthisp_rate, 1, color=color2, label='Most Hispanic')
    ax2.scatter(leasthisp_rate, 1, color=color3, label='Least Hispanic')
    ax2.scatter(mosthispt_rate, 2, color=color2)
    ax2.scatter(leasthispt_rate, 2, color=color3)
    ax2.scatter(mosthispti_rate, 3, color=color2)
    ax2.scatter(leasthispti_rate, 3, color=color3)
    ax2.scatter(mosthisp.BURDENPMALLRATE.mean(), 1.4, color=color2light)
    ax2.scatter(leasthisp.BURDENPMALLRATE.mean(), 1.4, color=color3light)
    ax2.scatter(mosthispt_rate_noas, 2.4, color=color2light)
    ax2.scatter(leasthispt_rate_noas, 2.4, color=color3light)
    ax2.scatter(mosthispti_rate_noas, 3.4, color=color2light)
    ax2.scatter(leasthispti_rate_noas, 3.4, color=color3light)
    # Add connector lines
    ax1.hlines(y=1, xmin=leastwhite_rate, xmax=mostwhite_rate,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax1.hlines(y=1.4, xmin=leastwhite.BURDENPMALLRATE.mean(),
        xmax=mostwhite.BURDENPMALLRATE.mean(), colors='darkgrey', ls='-', lw=1,
        zorder=0)
    ax1.hlines(y=2, xmin=leastwhitet_rate, xmax=mostwhitet_rate,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax1.hlines(y=2.4, xmin=leastwhitet_rate_noas, xmax=mostwhitet_rate_noas,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax1.hlines(y=3, xmin=leastwhiteti_rate, xmax=mostwhiteti_rate,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax1.hlines(y=3.4, xmin=leastwhiteti_rate_noas, xmax=mostwhiteti_rate_noas,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax2.hlines(y=1, xmin=leasthisp_rate, xmax=mosthisp_rate,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax2.hlines(y=1.4, xmin=leasthisp.BURDENPMALLRATE.mean(),
        xmax=mosthisp.BURDENPMALLRATE.mean(), colors='darkgrey', ls='-', lw=1,
        zorder=0)
    ax2.hlines(y=2, xmin=leasthispt_rate, xmax=mosthispt_rate,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax2.hlines(y=2.4, xmin=leasthispt_rate_noas, xmax=mosthispt_rate_noas,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax2.hlines(y=3, xmin=leasthispti_rate, xmax=mosthispti_rate,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    ax2.hlines(y=3.4, xmin=leasthispti_rate_noas, xmax=mosthispti_rate_noas,
        colors='darkgrey', ls='-', lw=1, zorder=0)
    # Aesthetics
    ax1.set_title('(A) Racial disparities', fontsize=14, loc='left', y=1.07)
    ax2.set_title('(B) Ethnic disparities', fontsize=14, loc='left', y=1.07)
    ax1.legend(ncol=2, frameon=False, bbox_to_anchor=(0.95, -0.18))
    ax2.legend(ncol=2, frameon=False, bbox_to_anchor=(1.01, -0.18))
    for ax in [ax1, ax2]:
        ax.set_xlim([15,85])
        ax.set_xticks(np.arange(15,95,10))
        ax.set_yticks([1,1.4,2,2.4,3,3.4])
        ax.set_yticklabels([])
        ax.set_xlabel('PM$_{\mathregular{2.5}}$-attributable premature deaths ' +
            '[per 100000]')
    ax1.set_yticklabels(['GBD RR with GBD state-level rates',
        'GBD RR with GBD state-level rates\n(no age standardization)',
        'Turner et al. RR with GBD state-level\nrates',
        'Turner et al. RR with GBD state-level\nrates (no age standardization)',   
        'Turner et al. RR with USALEEP\ntract-level rates',
        'Turner et al. RR with USALEEP\ntract-level rates '+\
        '(no age standardization)'])
    for ax in [ax1, ax2]:
        ax.invert_yaxis()
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
    ax2.tick_params(axis='y', which='both', left=False)
    plt.subplots_adjust(left=0.3, bottom=0.2, wspace=0.3, right=0.95)
    plt.savefig(DIR_FIG+'figS5_REVISED.pdf', dpi=1000)
    return 

def figS6(burdents):
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
    import matplotlib.pyplot as plt
    import pandas as pd
    years = np.arange(2010, 2020, 1)
    # Open burdens calculated without TMREL 
    notmrel = []
    for year in np.arange(2010, 2020, 1):
        print(year)
        notmrelty = pd.read_parquet(DIR_HARM+'burdens_%d_TROPOMIv2_noTMREL.gzip'%(year))
        notmrel.append(notmrelty)
    notmrel = pd.concat(notmrel)
    copd, lri, lc, ihd, dm, st, asthma = [], [], [], [], [], [], []
    copdin, lriin, lcin, ihdin, dmin, stin, asthmain = [], [], [], [], [], [], []
    copdnotmrel, lrinotmrel, lcnotmrel, ihdnotmrel = [], [], [], []
    dmnotmrel, stnotmrel, asthmanotmrel = [], [], []
    popped = ['pop_m_lt5','pop_f_lt5','pop_m_5-9','pop_f_5-9', 
        'pop_m_10-14','pop_f_10-14','pop_m_15-17','pop_m_18-19',
        'pop_f_15-17','pop_f_18-19']
    pop25p = ['pop_m_25-29', 'pop_f_25-29', 'pop_m_30-34', 'pop_m_35-39', 
        'pop_m_40-44', 'pop_m_45-49', 'pop_m_50-54', 'pop_m_55-59', 
        'pop_m_60-61', 'pop_m_62-64', 'pop_m_65-66', 'pop_m_67-69', 
        'pop_m_70-74', 'pop_m_75-79', 'pop_m_80-84', 'pop_m_gt85', 
        'pop_f_30-34', 'pop_f_35-39', 'pop_f_40-44', 'pop_f_45-49', 
        'pop_f_50-54', 'pop_f_55-59', 'pop_f_60-61', 'pop_f_62-64', 
        'pop_f_65-66', 'pop_f_67-69', 'pop_f_70-74', 'pop_f_75-79', 
        'pop_f_80-84', 'pop_f_gt85']
    for year in np.arange(2010, 2020, 1):
        vintage = '%d-%d' % (year-4, year)
        burdenty = burdents.loc[burdents['YEAR'] == vintage]
        notmrelty = notmrel.loc[notmrel['YEAR'] == vintage]
        # Burdens for health endpoints
        copd.append(burdenty.BURDENCOPD.sum())
        lri.append(burdenty.BURDENLRI.sum())
        lc.append(burdenty.BURDENLC.sum())
        ihd.append(burdenty.BURDENIHD.sum())
        dm.append(burdenty.BURDENDM.sum())
        st.append(burdenty.BURDENST.sum())
        asthma.append(burdenty.BURDENPA.sum())
        copdnotmrel.append(notmrelty.BURDENCOPD.sum())
        lrinotmrel.append(notmrelty.BURDENLRI.sum())
        lcnotmrel.append(notmrelty.BURDENLC.sum())
        ihdnotmrel.append(notmrelty.BURDENIHD.sum())
        dmnotmrel.append(notmrelty.BURDENDM.sum())
        stnotmrel.append(notmrelty.BURDENST.sum())
        asthmanotmrel.append(notmrelty.BURDENPA.sum())    
        # Incidence
        asthmain.append(((burdenty.BURDENPA/burdenty[popped].sum(
            axis=1))*100000).mean())
        copdin.append(((burdenty.BURDENCOPD/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        lcin.append(((burdenty.BURDENLC/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        ihdin.append(((burdenty.BURDENIHD/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        dmin.append(((burdenty.BURDENDM/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        stin.append(((burdenty.BURDENST/burdenty[pop25p].sum(
            axis=1))*100000).mean())
        lriin.append(((burdenty.BURDENST/burdenty.pop_tot)*100000).mean())
    # Convert lists to arrays
    copd = np.array(copd)
    lri = np.array(lri)
    lc = np.array(lc)
    ihd = np.array(ihd)
    dm = np.array(dm)
    st = np.array(st)
    asthma = np.array(asthma)
    copdnotmrel = np.array(copdnotmrel)
    lrinotmrel = np.array(lrinotmrel)
    lcnotmrel = np.array(lcnotmrel)
    ihdnotmrel = np.array(ihdnotmrel)
    dmnotmrel = np.array(dmnotmrel)
    stnotmrel = np.array(stnotmrel)
    asthmanotmrel = np.array(asthmanotmrel)
    copdin = np.array(copdin)
    lriin = np.array(lriin)
    lcin = np.array(lcin)
    ihdin = np.array(ihdin)
    dmin = np.array(dmin)
    stin = np.array(stin)
    asthmain = np.array(asthmain)
    # Plotting supplementary figure
    fig = plt.figure(figsize=(9, 7))
    axts1 = plt.subplot2grid((2,2),(0,0))
    axts2 = plt.subplot2grid((2,2),(0,1))
    axts3 = plt.subplot2grid((2,2),(1,0))
    axts4 = plt.subplot2grid((2,2),(1,1))
    axts1.set_title('(A) Premature deaths due to PM$_\mathregular{2.5}$', 
        loc='left')
    axts2.set_title('(B) New asthma cases due to NO$_\mathregular{2}$', 
        loc='left')
    axts3.set_title('(C) Premature deaths due to PM$_\mathregular{2.5}$\nwith no TMREL', 
        loc='left')
    axts4.set_title('(D) New asthma cases due to NO$_\mathregular{2}$\nwith no TMREL', 
        loc='left')
    axts1.set_ylabel('[per 100,000 population]')
    axts2.set_ylabel('[per 100,000 pediatric population]')
    for i in np.arange(0, len(np.arange(2010, 2020, 1)), 1):
        copdty = copdin[i]
        lrity = lriin[i]
        lcty = lcin[i]
        ihdty = ihdin[i]
        dmty = dmin[i]
        stty = stin[i]
        axts1.bar(i+2010, ihdty, color=color1, bottom=0, zorder=10)
        axts1.bar(i+2010, stty, color=color2, bottom=ihdty, zorder=10)
        axts1.bar(i+2010, lcty, color=color3, bottom=ihdty+stty, zorder=10)
        axts1.bar(i+2010, copdty, color=color4, bottom=ihdty+stty+lcty,
            zorder=10)
        axts1.bar(i+2010, dmty, color=color5, bottom=ihdty+stty+lcty+copdty,
            zorder=10)
        axts1.bar(i+2010, lrity, color=color6, bottom=ihdty+stty+lcty+copdty+dmty,
            zorder=10)
    axts1.set_ylim([0, 40])
    axts1.set_yticks(np.linspace(0, 40, 6))
    axts2.bar(years, asthmain, color=color7, zorder=10)
    for i in np.arange(0, len(np.arange(2010, 2020, 1)), 1):
        copdty = copdnotmrel[i]
        lrity = lrinotmrel[i]
        lcty = lcnotmrel[i]
        ihdty = ihdnotmrel[i]
        dmty = dmnotmrel[i]
        stty = stnotmrel[i]
        axts3.bar(i+2010, ihdty, color=color1, bottom=0, zorder=10)
        axts3.bar(i+2010, stty, color=color2, bottom=ihdty, zorder=10)
        axts3.bar(i+2010, lcty, color=color3, bottom=ihdty+stty, zorder=10)
        axts3.bar(i+2010, copdty, color=color4, bottom=ihdty+stty+lcty,
            zorder=10)
        axts3.bar(i+2010, dmty, color=color5, bottom=ihdty+stty+lcty+copdty,
            zorder=10)
        axts3.bar(i+2010, lrity, color=color6, bottom=ihdty+stty+lcty+copdty+dmty,
            zorder=10)
    axts1.yaxis.set_label_coords(-0.15, 0.5)
    axts2.yaxis.set_label_coords(-0.15, 0.5)
    axts4.bar(years, asthmanotmrel, color=color7, zorder=10)
    axts3.set_ylim([0, 150000])
    axts3.set_yticks(np.linspace(0, 150000, 6))
    axts4.set_ylim([0, 400000])
    axts4.set_yticks(np.linspace(0, 400000, 6))
    for ax in [axts1, axts2, axts3, axts4]:
        ax.set_xlim([2009.25, 2019.75])
        ax.set_xticks(np.arange(2010, 2020, 1))
        ax.set_xticklabels([])
        ax.spines['bottom'].set_zorder(1000)
    for ax in [axts3, axts4]:
        ax.set_xticklabels(['2010', '', '2012', '', '2014', '', 
            '2016', '', '2018', ''])    
    plt.subplots_adjust(left=0.08, hspace=0.4, wspace=0.35, bottom=0.18, top=0.93, 
        right=0.92)
    pihd = mpatches.Patch(color=color1, label='Ischemic heart disease')
    pst = mpatches.Patch(color=color2, label='Stroke')
    plc = mpatches.Patch(color=color3, label='Lung cancer')
    pcopd = mpatches.Patch(color=color4, label='COPD')
    pdm = mpatches.Patch(color=color5, label='Type 2 diabetes')
    plri = mpatches.Patch(color=color6, label='Lower respiratory infection')
    axts3.legend(handles=[pihd, pst, plc, pcopd, pdm, plri],
        bbox_to_anchor=(1.25, -0.12), ncol=2, frameon=False)
    plt.savefig(DIR_FIG+'figS6_REVISED.pdf', dpi=500)
    return

def figS7(burdents):
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
    import matplotlib.pyplot as plt
    # For absolute concentrations/rates for racial subgroups
    PM25_mostwhite, PM25_leastwhite = [], []
    NO2_mostwhite, NO2_leastwhite = [], []
    # For relative disparities
    PM25_race_relative, NO2_race_relative = [], []
    # Same as above but for ethnic subgroups
    PM25_mosthisp, PM25_leasthisp = [], []
    NO2_mosthisp, NO2_leasthisp = [], []
    PM25_ethnic_relative, NO2_ethnic_relative = [], []
    for year in np.arange(2010, 2020, 1):
        yearst = '%d-%d' % (year-4, year)
        burdenty = burdents.loc[burdents['YEAR'] == yearst].copy(deep=True)
        # Define ethnoracial groups
        burdenty['fracwhite'] = ((burdenty[['race_nh_white', 'race_h_white'
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
        PM25_mosthisp.append(mosthisp.PM25.mean())
        PM25_leasthisp.append(leasthisp.PM25.mean())
        NO2_mosthisp.append(mosthisp.NO2.mean())
        NO2_leasthisp.append(leasthisp.NO2.mean())
        # Relative disparities
        PM25_race_relative.append(leastwhite.PM25.mean()/mostwhite.PM25.mean())
        NO2_race_relative.append(leastwhite.NO2.mean()/mostwhite.NO2.mean())
        PM25_ethnic_relative.append(mosthisp.PM25.mean()/leasthisp.PM25.mean())
        NO2_ethnic_relative.append(mosthisp.NO2.mean()/leasthisp.NO2.mean())
    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax1t = plt.subplot2grid((10, 2), (0, 0), rowspan=3)
    ax1b = plt.subplot2grid((10, 2), (3, 0), rowspan=2)
    ax2t = plt.subplot2grid((10, 2), (5, 0), rowspan=3)
    ax2b = plt.subplot2grid((10, 2), (8, 0), rowspan=2)
    ax3t = plt.subplot2grid((10, 2), (0, 1), rowspan=3)
    ax3b = plt.subplot2grid((10, 2), (3, 1), rowspan=2)
    ax4t = plt.subplot2grid((10, 2), (5, 1), rowspan=3)
    ax4b = plt.subplot2grid((10, 2), (8, 1), rowspan=2)
    years = np.arange(2010, 2020, 1)
    # Racial PM25
    for i, year in enumerate(years):
        ax1t.vlines(x=year, ymin=PM25_mostwhite[i], ymax=PM25_leastwhite[i],
                    colors='darkgrey', ls='-', lw=1)
    ax1t.scatter(years, PM25_mostwhite, color=color3, zorder=10,
                 label='Most white')
    ax1t.scatter(years, PM25_leastwhite, color=color2, zorder=10,
                 label='Least white')
    # Text for first and last years
    ax1t.text(years[0], PM25_mostwhite[0]-0.5, '%.1f' % PM25_mostwhite[0],
              ha='center', va='top', color=color3, fontsize=8)
    ax1t.text(years[0], PM25_leastwhite[0]+0.3, '%.1f' % PM25_leastwhite[0],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax1t.text(years[-1], PM25_mostwhite[-1]-0.5, '%.1f' % PM25_mostwhite[-1],
              ha='center', va='top', color=color3, fontsize=8)
    ax1t.text(years[-1], PM25_leastwhite[-1]+0.3, '%.1f' % PM25_leastwhite[-1],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax1b.plot(years, PM25_race_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(PM25_race_relative):
        if i == 0:
            ax1b.text(years[i], PM25_race_relative[i]*1.02, '%.2f' % txt,
                      ha='center', fontsize=8, clip_on=False)
        else:
            ax1b.text(years[i], PM25_race_relative[i]*1.015, '%.2f' % txt,
                      ha='center', fontsize=8, clip_on=False)
    # Racial NO2
    for i, year in enumerate(years):
        ax2t.vlines(x=year, ymin=NO2_mostwhite[i], ymax=NO2_leastwhite[i],
                    colors='darkgrey', ls='-', lw=1)
    ax2t.scatter(years, NO2_mostwhite, color=color3, zorder=10,
                 label='Most white')
    ax2t.scatter(years, NO2_leastwhite, color=color2, zorder=10,
                 label='Least white')
    ax2t.text(years[0], NO2_mostwhite[0]-2.4, '%.1f' % NO2_mostwhite[0],
              ha='center', va='bottom', color=color3, fontsize=8)
    ax2t.text(years[0], NO2_leastwhite[0]+2.2, '%.1f' % NO2_leastwhite[0],
              ha='center', va='top', color=color2, fontsize=8)
    ax2t.text(years[-1], NO2_mostwhite[-1]-1.2, '%.1f' % NO2_mostwhite[-1],
              ha='center', va='top', color=color3, fontsize=8)
    ax2t.text(years[-1], NO2_leastwhite[-1]+1.2, '%.1f' % NO2_leastwhite[-1],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax2b.plot(years, NO2_race_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(NO2_race_relative):
        ax2b.text(years[i], NO2_race_relative[i]*1.03, '%.2f' % txt,
                  fontsize=8, ha='center', clip_on=False)
    # Ethnic PM25
    for i, year in enumerate(years):
        ax3t.vlines(x=year, ymin=PM25_leasthisp[i], ymax=PM25_mosthisp[i],
                    colors='darkgrey', ls='-', lw=1)
    ax3t.scatter(years, PM25_leasthisp, color=color3, zorder=10,
                 label='Least Hispanic')
    ax3t.scatter(years, PM25_mosthisp, color=color2, zorder=10,
                 label='Most Hispanic')
    ax3t.text(years[0], PM25_leasthisp[0]+0.7, '%.1f' % PM25_leasthisp[0],
              ha='center', va='top', color=color3, fontsize=8)
    ax3t.text(years[0], PM25_mosthisp[0]-0.7, '%.1f' % PM25_mosthisp[0],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax3t.text(years[-1], PM25_leasthisp[-1]-0.5, '%.1f' % PM25_leasthisp[-1],
              ha='center', va='top', color=color3, fontsize=8)
    ax3t.text(years[-1], PM25_mosthisp[-1]+0.3, '%.1f' % PM25_mosthisp[-1],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax3b.plot(years, PM25_ethnic_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(PM25_ethnic_relative):
        ax3b.text(years[i], PM25_ethnic_relative[i]*1.06, '%.2f' % txt, fontsize=8,
                  ha='center', clip_on=False)
    # Ethnic NO2
    for i, year in enumerate(years):
        ax4t.vlines(x=year, ymin=NO2_leasthisp[i], ymax=NO2_mosthisp[i],
                    colors='darkgrey', ls='-', lw=1)
    ax4t.scatter(years, NO2_leasthisp, color=color3, zorder=10,
                 label='Least Hispanic')
    ax4t.scatter(years, NO2_mosthisp, color=color2, zorder=10,
                 label='Most Hispanic')
    ax4t.text(years[0], NO2_leasthisp[0]-2.4, '%.1f' % NO2_leasthisp[0],
              ha='center', va='bottom', color=color3, fontsize=8)
    ax4t.text(years[0], NO2_mosthisp[0]+2.2, '%.1f' % NO2_mosthisp[0],
              ha='center', va='top', color=color2, fontsize=8)
    ax4t.text(years[-1], NO2_leasthisp[-1]-2.2, '%.1f' % NO2_leasthisp[-1],
              ha='center', va='bottom', color=color3, fontsize=8)
    ax4t.text(years[-1], NO2_mosthisp[-1]+2.2, '%.1f' % NO2_mosthisp[-1],
              ha='center', va='top', color=color2, fontsize=8)
    ax4b.plot(years, NO2_ethnic_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(NO2_ethnic_relative):
        #     if i==0:
        #         ax4b.text(years[i], pd_ethnic_relative[i]*1.1, '%.2f'%txt, fontsize=8,
        #             ha='center', clip_on=False)
        #     elif i==8:
        #         ax4b.text(years[i], pd_ethnic_relative[i]*1.05, '%.2f'%txt, fontsize=8,
        #             ha='center', clip_on=False)
        #     else:
        ax4b.text(years[i], NO2_ethnic_relative[i]*1.02, '%.2f' % txt, fontsize=8,
                  ha='center', clip_on=False)
    # Aesthetics
    ax1t.set_title('(A) Racial disparities', fontsize=14, loc='left', y=1.07)
    ax2t.set_title('(C)', fontsize=14, loc='left', y=1.07)
    ax3t.set_title('(B) Ethnic disparities', fontsize=14, loc='left', y=1.07)
    ax4t.set_title('(D)', fontsize=14, loc='left', y=1.07)
    ax1t.set_ylabel('PM$_{\mathregular{2.5}}$ [$\mathregular{\mu}$g m' +
                    '$^{\mathregular{-3}}$]')
    ax1t.get_yaxis().set_label_coords(-0.15, 0.5)
    ax2t.set_ylabel('NO$_{\mathregular{2}}$ [ppbv]')
    ax2t.get_yaxis().set_label_coords(-0.15, 0.5)
    # Axis limits
    for ax in [ax1t, ax3t]:
        ax.set_ylim([6, 11])
        ax.set_yticks(np.linspace(6, 11, 6))
        ax.set_yticklabels([])
    ax1t.set_yticklabels([int(x) for x in np.linspace(6, 11, 6)])
    for ax in [ax2t, ax4t]:
        ax.set_ylim([4, 18])
        ax.set_yticks(np.linspace(4, 18, 3))
        ax.set_yticklabels([])
    ax2t.set_yticklabels([int(x) for x in np.linspace(4, 18, 3)])
    # Relative disparities plots
    ax1b.set_ylim([min(PM25_race_relative)*0.97,
                   max(PM25_race_relative)*1.0])
    ax2b.set_ylim([min(NO2_race_relative)*0.92,
                   max(NO2_race_relative)*1.0])
    ax3b.set_ylim([min(PM25_ethnic_relative)*0.97,
                   max(PM25_ethnic_relative)*1.0])
    ax4b.set_ylim([min(NO2_ethnic_relative)*0.97,
                   max(NO2_ethnic_relative)*1])
    plt.subplots_adjust(wspace=0.2, hspace=15.5,
                        bottom=0.08, top=0.92, right=0.96)
    ax2t.legend(ncol=2, frameon=False, bbox_to_anchor=(0.95, -1.))
    ax4t.legend(ncol=2, frameon=False, bbox_to_anchor=(1.01, -1.))
    for ax in [ax1t, ax2t, ax3t, ax4t]:
        ax.set_xlim([2009.5, 2019.5])
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False)
    for ax in [ax1b, ax2b, ax3b, ax4b]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlim([2009.5, 2019.5])
        # Only draw spine between the y-ticks
        ax.spines.bottom.set_bounds((2010, 2019))
        ax.set_xticks(years)
        ax.set_xticklabels(['2010', '', '', '2013', '', '', '2016', '', '',
                            '2019'])
        ax.set_yticks([])
        # Move relative disparities subplots up
        box = ax.get_position()
        box.y0 = box.y0 + 0.04
        box.y1 = box.y1 + 0.04
        ax.set_position(box)
    plt.savefig(DIR_FIG+'figS7_REVISED.pdf', dpi=1000)
    return

def figS9(burdents):
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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
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
        yearst = '%d-%d' % (year-4, year)
        burdenty = burdents.loc[burdents['YEAR'] == yearst].copy(deep=True)
        # Define ethnoracial groups
        burdenty['fracwhite'] = ((burdenty[['race_nh_white', 'race_h_white'
                                            ]].sum(axis=1))/burdenty['race_tot'])
        burdenty['frachisp'] = (burdenty['race_h']/burdenty['race_tot'])
        # Define population-weighted categories
        pm25black = ((burdenty['PM25']*burdenty[['race_nh_black',
                                                 'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
                                                                                               'race_h_black']].sum(axis=1).sum())
        pm25white = ((burdenty['PM25'] *
                      burdenty[['race_nh_white', 'race_h_white']].sum(axis=1)).sum() /
                     burdenty[['race_nh_white', 'race_h_white']].sum(axis=1).sum())
        no2black = ((burdenty['NO2']*burdenty[['race_nh_black',
                                               'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
                                                                                             'race_h_black']].sum(axis=1).sum())
        no2white = ((burdenty['NO2'] *
                     burdenty[['race_nh_white', 'race_h_white']].sum(axis=1)).sum() /
                    burdenty[['race_nh_white', 'race_h_white']].sum(axis=1).sum())
        pdblack = ((burdenty['BURDENPMALLRATE']*burdenty[['race_nh_black',
                                                          'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
                                                                                                        'race_h_black']].sum(axis=1).sum())
        pdwhite = ((burdenty['BURDENPMALLRATE'] *
                    burdenty[['race_nh_white', 'race_h_white']].sum(axis=1)).sum() /
                   burdenty[['race_nh_white', 'race_h_white']].sum(axis=1).sum())
        asthmablack = ((burdenty['BURDENPARATE']*burdenty[['race_nh_black',
                                                               'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
                                                                                                             'race_h_black']].sum(axis=1).sum())
        asthmawhite = ((burdenty['BURDENPARATE'] *
                        burdenty[['race_nh_white', 'race_h_white']].sum(axis=1)).sum() /
                       burdenty[['race_nh_white', 'race_h_white']].sum(axis=1).sum())
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
        asthmanh = ((burdenty['BURDENPARATE']*burdenty['race_nh']).sum() /
                    burdenty['race_nh'].sum())
        asthmah = ((burdenty['BURDENPARATE']*burdenty['race_h']).sum() /
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
    fig = plt.figure(figsize=(8, 6))
    ax1t = plt.subplot2grid((10, 2), (0, 0), rowspan=3)
    ax1b = plt.subplot2grid((10, 2), (3, 0), rowspan=2)
    ax2t = plt.subplot2grid((10, 2), (5, 0), rowspan=3)
    ax2b = plt.subplot2grid((10, 2), (8, 0), rowspan=2)
    ax3t = plt.subplot2grid((10, 2), (0, 1), rowspan=3)
    ax3b = plt.subplot2grid((10, 2), (3, 1), rowspan=2)
    ax4t = plt.subplot2grid((10, 2), (5, 1), rowspan=3)
    ax4b = plt.subplot2grid((10, 2), (8, 1), rowspan=2)
    years = np.arange(2010, 2020, 1)
    # Racial NO2-attributable pediatric asthma
    for i, year in enumerate(years):
        ax1t.vlines(x=year, ymin=asthma_white[i], ymax=asthma_black[i],
                    colors='darkgrey', ls='-', lw=1)
    ax1t.scatter(years, asthma_white, color=color3, zorder=10,
                 label='White')
    ax1t.scatter(years, asthma_black, color=color2, zorder=10,
                 label='Black')
    # Text for first and last years
    ax1t.text(years[0], asthma_white[0]-30, '%d' % asthma_white[0],
              ha='center', va='top', color=color3, fontsize=8)
    ax1t.text(years[0], asthma_black[0]+25, '%d' % asthma_black[0],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax1t.text(years[-1], asthma_white[-1]-30, '%d' % asthma_white[-1],
              ha='center', va='top', color=color3, fontsize=8)
    ax1t.text(years[-1], asthma_black[-1]+25, '%d' % asthma_black[-1],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax1b.plot(years, asthma_race_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(asthma_race_relative):
        ax1b.text(years[i], asthma_race_relative[i]*1.015, '%.2f' % txt,
                  ha='center', fontsize=8, clip_on=False)
    # Racial PM2.5-attributable premature mortality
    for i, year in enumerate(years):
        ax2t.vlines(x=year, ymin=pd_white[i], ymax=pd_black[i],
                    colors='darkgrey', ls='-', lw=1)
    ax2t.scatter(years, pd_white, color=color3, zorder=10,
                 label='White')
    ax2t.scatter(years, pd_black, color=color2, zorder=10,
                 label='Black')
    ax2t.text(years[0], pd_white[0]-3.5, '%d' % pd_white[0],
              ha='center', va='bottom', color=color3, fontsize=8)
    ax2t.text(years[0], pd_black[0]+3.5, '%d' % pd_black[0],
              ha='center', va='top', color=color2, fontsize=8)
    ax2t.text(years[-1], pd_white[-1]-1.2, '%d' % pd_white[-1],
              ha='center', va='top', color=color3, fontsize=8)
    ax2t.text(years[-1], pd_black[-1]+1, '%d' % pd_black[-1],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax2b.plot(years, pd_race_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(pd_race_relative):
        if i == 3:
            ax2b.text(years[i], pd_race_relative[i]*1.02, '%.2f' % txt,
                      fontsize=8, ha='center', clip_on=False)
        elif i == 6:
            ax2b.text(years[i], pd_race_relative[i]*1.01, '%.2f' % txt,
                      fontsize=8, ha='center', clip_on=False)
        elif i == 7:
            ax2b.text(years[i], pd_race_relative[i]*1.025, '%.2f' % txt,
                      fontsize=8, ha='center', clip_on=False)
        elif i == 8:
            ax2b.text(years[i], pd_race_relative[i]*1.025, '%.2f' % txt,
                      fontsize=8, ha='center', clip_on=False)
        else:
            ax2b.text(years[i], pd_race_relative[i]*1.015, '%.2f' % txt,
                      fontsize=8, ha='center', clip_on=False)
    # Ethnic NO2-attributable pediatric asthma
    for i, year in enumerate(years):
        ax3t.vlines(x=year, ymin=asthma_nonhisp[i], ymax=asthma_hisp[i],
                    colors='darkgrey', ls='-', lw=1)
    ax3t.scatter(years, asthma_nonhisp, color=color3, zorder=10,
                 label='Non-Hispanic')
    ax3t.scatter(years, asthma_hisp, color=color2, zorder=10,
                 label='Hispanic')
    ax3t.text(years[0], asthma_nonhisp[0]-30, '%d' % asthma_nonhisp[0],
              ha='center', va='top', color=color3, fontsize=8)
    ax3t.text(years[0], asthma_hisp[0]+25, '%d' % asthma_hisp[0],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax3t.text(years[-1], asthma_nonhisp[-1]-30, '%d' % asthma_nonhisp[-1],
              ha='center', va='top', color=color3, fontsize=8)
    ax3t.text(years[-1], asthma_hisp[-1]+25, '%d' % asthma_hisp[-1],
              ha='center', va='bottom', color=color2, fontsize=8)
    ax3b.plot(years, asthma_ethnic_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(asthma_ethnic_relative):
        ax3b.text(years[i], asthma_ethnic_relative[i]*1.013, '%.2f' % txt, fontsize=8,
                  ha='center', clip_on=False)
    # Ethnic PM2.5-attributable premature mortality
    for i, year in enumerate(years):
        ax4t.vlines(x=year, ymin=pd_nonhisp[i], ymax=pd_hisp[i],
                    colors='darkgrey', ls='-', lw=1)
    ax4t.scatter(years, pd_nonhisp, color=color3, zorder=10,
                 label='Non-Hispanic')
    ax4t.scatter(years, pd_hisp, color=color2, zorder=10,
                 label='Hispanic')
    ax4t.text(years[0], pd_nonhisp[0]+1, '%d' % pd_nonhisp[0],
              ha='center', va='bottom', color=color3, fontsize=8)
    ax4t.text(years[0], pd_hisp[0]-1.2, '%d' % pd_hisp[0],
              ha='center', va='top', color=color2, fontsize=8)
    ax4t.text(years[-1], pd_nonhisp[-1]+1, '%d' % pd_nonhisp[-1],
              ha='center', va='bottom', color=color3, fontsize=8)
    ax4t.text(years[-1], pd_hisp[-1]-1.2, '%d' % pd_hisp[-1],
              ha='center', va='top', color=color2, fontsize=8)
    ax4b.plot(years, pd_ethnic_relative, color='k', marker='o',
              markerfacecolor='w', markeredgecolor='k', clip_on=False)
    for i, txt in enumerate(pd_ethnic_relative):
        ax4b.text(years[i], pd_ethnic_relative[i]*1.045, '%.2f' % txt, fontsize=8,
                  ha='center', clip_on=False)
    # Aesthetics
    ax1t.set_title('(A) Racial disparities', fontsize=14, loc='left', y=1.07)
    ax2t.set_title('(C)', fontsize=14, loc='left', y=1.07)
    ax3t.set_title('(B) Ethnic disparities', fontsize=14, loc='left', y=1.07)
    ax4t.set_title('(D)', fontsize=14, loc='left', y=1.07)
    ax1t.set_ylabel('New asthma cases due\nto NO$_{\mathregular{2}}$ ' +
        'per 100000')
    ax1t.get_yaxis().set_label_coords(-0.15, 0.5)
    ax2t.set_ylabel('Premature deaths due\nto PM$_{\mathregular{2.5}}$ ' +
        'per 100000')
    ax2t.get_yaxis().set_label_coords(-0.15, 0.5)
    # Axis limits
    for ax in [ax1t, ax3t]:
        ax.set_ylim([100, 400])
        ax.set_yticks(np.arange(100, 400+100, 100))
        ax.set_yticklabels([])
    ax1t.set_yticklabels([int(x) for x in np.arange(100, 400+100, 100)])
    for ax in [ax2t, ax4t]:
        ax.set_ylim([16, 40])
        ax.set_yticks(np.linspace(16, 40, 5))
        ax.set_yticklabels([])
    ax2t.set_yticklabels([int(x) for x in np.linspace(16, 40, 5)])
    # # Relative disparities plots
    ax1b.set_ylim([min(asthma_race_relative)*0.97,
                   max(asthma_race_relative)*1.0])
    ax2b.set_ylim([min(pd_race_relative)*0.985,
                   max(pd_race_relative)*1.0])
    ax3b.set_ylim([min(asthma_ethnic_relative)*0.98,
                   max(asthma_ethnic_relative)*1.0])
    ax4b.set_ylim([min(pd_ethnic_relative)*0.9,
                   max(pd_ethnic_relative)*0.95])
    plt.subplots_adjust(wspace=0.2, hspace=15.5,
                        bottom=0.08, top=0.92, right=0.96)
    ax2t.legend(ncol=2, frameon=False, bbox_to_anchor=(0.83, -1.))
    ax4t.legend(ncol=2, frameon=False, bbox_to_anchor=(1.01, -1.))
    for ax in [ax1t, ax2t, ax3t, ax4t]:
        ax.set_xlim([2009.5, 2019.5])
        for spine in ['top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False)
    for ax in [ax1b, ax2b, ax3b, ax4b]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlim([2009.5, 2019.5])
        # Only draw spine between the y-ticks
        ax.spines.bottom.set_bounds((2010, 2019))
        ax.set_xticks(years)
        ax.set_xticklabels(['2010', '', '', '2013', '', '', '2016', '', '',
                            '2019'])
        ax.set_yticks([])
        # Move relative disparities subplots up
        box = ax.get_position()
        box.y0 = box.y0 + 0.04
        box.y1 = box.y1 + 0.04
        ax.set_position(box)
    plt.savefig(DIR_FIG+'figS9_REVISED.pdf', dpi=1000)
    return

def figS10(burdents):
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
    import matplotlib.pyplot as plt
    # For population-weighted values
    PM25_white, PM25_black = [], []
    NO2_white, NO2_black = [], []
    pd_white, pd_black = [], []
    asthma_white, asthma_black = [], []
    PM25_hisp, PM25_nonhisp = [], []
    NO2_hisp, NO2_nonhisp = [], []
    pd_hisp, pd_nonhisp = [], []
    asthma_hisp, asthma_nonhisp = [], []
    for year in np.arange(2010, 2020, 1):
        yearst = '%d-%d' % (year-4, year)
        burdenty = burdents.loc[burdents['YEAR'] == yearst].copy(deep=True)
        burdenty['fracwhite'] = ((burdenty[['race_nh_white', 'race_h_white'
            ]].sum(axis=1))/burdenty['race_tot'])
        burdenty['frachisp'] = (burdenty['race_h']/burdenty['race_tot'])
        pm25black = ((burdenty['PM25']*burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1).sum())
        pm25white = ((burdenty['PM25']*burdenty[['race_nh_white', 
            'race_h_white']].sum(axis=1)).sum()/burdenty[['race_nh_white',
            'race_h_white']].sum(axis=1).sum())
        no2black = ((burdenty['NO2']*burdenty[['race_nh_black', 'race_h_black']
            ].sum(axis=1)).sum()/burdenty[['race_nh_black','race_h_black']
            ].sum(axis=1).sum())
        no2white = ((burdenty['NO2']*burdenty[['race_nh_white',
            'race_h_white']].sum(axis=1)).sum()/burdenty[['race_nh_white',
            'race_h_white']].sum(axis=1).sum())
        pdblack = ((burdenty['BURDENPMALLRATE']*burdenty[['race_nh_black', 
            'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1).sum())
        pdwhite = ((burdenty['BURDENPMALLRATE']*burdenty[['race_nh_white',
            'race_h_white']].sum(axis=1)).sum()/burdenty[['race_nh_white',
            'race_h_white']].sum(axis=1).sum())
        asthmablack = ((burdenty['BURDENPARATE']*burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1)).sum()/burdenty[['race_nh_black',
            'race_h_black']].sum(axis=1).sum())
        asthmawhite = ((burdenty['BURDENPARATE']*burdenty[['race_nh_white', 
            'race_h_white']].sum(axis=1)).sum()/burdenty[['race_nh_white', 
            'race_h_white']].sum(axis=1).sum())
        pm25nh = ((burdenty['PM25']*burdenty['race_nh']).sum()/
            burdenty['race_nh'].sum())
        pm25h = ((burdenty['PM25']*burdenty['race_h']).sum()/ 
            burdenty['race_h'].sum())
        no2nh = ((burdenty['NO2']*burdenty['race_nh']).sum()/
            burdenty['race_nh'].sum())
        no2h = ((burdenty['NO2']*burdenty['race_h']).sum()/
            burdenty['race_h'].sum())
        pdnh = ((burdenty['BURDENPMALLRATE']*burdenty['race_nh']).sum() /
            burdenty['race_nh'].sum())
        pdh = ((burdenty['BURDENPMALLRATE']*burdenty['race_h']).sum()/
            burdenty['race_h'].sum())
        asthmanh = ((burdenty['BURDENPARATE']*burdenty['race_nh']).sum()/
            burdenty['race_nh'].sum())
        asthmah = ((burdenty['BURDENPARATE']*burdenty['race_h']).sum()/
            burdenty['race_h'].sum())
        PM25_white.append(pm25white)
        PM25_black.append(pm25black)
        NO2_white.append(no2white)
        NO2_black.append(no2black)
        pd_white.append(pdwhite)
        pd_black.append(pdblack)
        asthma_white.append(asthmawhite)
        asthma_black.append(asthmablack)
        PM25_hisp.append(pm25h)
        PM25_nonhisp.append(pm25nh)
        NO2_hisp.append(no2h)
        NO2_nonhisp.append(no2nh)
        pd_hisp.append(pdh)
        pd_nonhisp.append(pdnh)
        asthma_hisp.append(asthmah)
        asthma_nonhisp.append(asthmanh)
    # For absolute concentrations/rates for racial subgroups
    PM25_mostwhite, PM25_leastwhite = [], []
    NO2_mostwhite, NO2_leastwhite = [], []
    pd_mostwhite, pd_leastwhite = [], []
    asthma_mostwhite, asthma_leastwhite = [], []
    PM25_mosthisp, PM25_leasthisp = [], []
    NO2_mosthisp, NO2_leasthisp = [], []
    pd_mosthisp, pd_leasthisp = [], []
    asthma_mosthisp, asthma_leasthisp = [], []
    for year in np.arange(2010, 2020, 1):
        print(year)
        yearst = '%d-%d' % (year-4, year)
        burdenty = burdents.loc[burdents['YEAR'] == yearst].copy(deep=True)
        burdenty['fracwhite'] = ((burdenty[['race_nh_white', 'race_h_white'
            ]].sum(axis=1))/burdenty['race_tot'])
        burdenty['frachisp'] = (burdenty['race_h']/burdenty['race_tot'])
        mostwhite = burdenty.iloc[np.where(burdenty.fracwhite >=
            np.nanpercentile(burdenty.fracwhite, 90))]
        leastwhite = burdenty.iloc[np.where(burdenty.fracwhite <=
            np.nanpercentile(burdenty.fracwhite, 10))]
        mosthisp = burdenty.iloc[np.where(burdenty.frachisp >
            np.nanpercentile(burdenty.frachisp, 90))]
        leasthisp = burdenty.iloc[np.where(burdenty.frachisp <
            np.nanpercentile(burdenty.frachisp, 10))]
        PM25_mostwhite.append(mostwhite.PM25.mean())
        PM25_leastwhite.append(leastwhite.PM25.mean())
        NO2_mostwhite.append(mostwhite.NO2.mean())
        NO2_leastwhite.append(leastwhite.NO2.mean())
        PM25_mosthisp.append(mosthisp.PM25.mean())
        PM25_leasthisp.append(leasthisp.PM25.mean())
        NO2_mosthisp.append(mosthisp.NO2.mean())
        NO2_leasthisp.append(leasthisp.NO2.mean())    
        # Age standardized burdens    
        mostwhitepm, leastwhitepm, mosthisppm, leasthisppm = [], [], [], []
        for pmendpoint in ['IHD', 'DM', 'ST', 'COPD', 'LRI', 'LC']:
            mostwhitepm.append(agestandardize(burdenty, mostwhite, pmendpoint))
            leastwhitepm.append(agestandardize(burdenty, leastwhite, pmendpoint))
            mosthisppm.append(agestandardize(burdenty, mosthisp, pmendpoint))
            leasthisppm.append(agestandardize(burdenty, leasthisp, pmendpoint))        
        mostwhitepm = np.sum(mostwhitepm)
        leastwhitepm = np.sum(leastwhitepm)
        mosthisppm = np.sum(mosthisppm)
        leasthisppm = np.sum(leasthisppm)    
        mostwhiteasthma = agestandardize(burdenty, mostwhite, 'PA')
        leastwhiteasthma = agestandardize(burdenty, leastwhite, 'PA')
        mosthispasthma = agestandardize(burdenty, mosthisp, 'PA')
        leasthispasthma = agestandardize(burdenty, leasthisp, 'PA')
        pd_mostwhite.append(mostwhitepm)
        pd_leastwhite.append(leastwhitepm)
        asthma_mostwhite.append(mostwhiteasthma)
        asthma_leastwhite.append(leastwhiteasthma)
        pd_mosthisp.append(mosthisppm)
        pd_leasthisp.append(leasthisppm)
        asthma_mosthisp.append(mosthispasthma)
        asthma_leasthisp.append(leasthispasthma)
    # Plotting
    years = np.arange(2010,2020,1)
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((4,2),(0,0))
    ax2 = plt.subplot2grid((4,2),(1,0))
    ax3 = plt.subplot2grid((4,2),(2,0))
    ax4 = plt.subplot2grid((4,2),(3,0))
    ax5 = plt.subplot2grid((4,2),(0,1))
    ax6 = plt.subplot2grid((4,2),(1,1))
    ax7 = plt.subplot2grid((4,2),(2,1))
    ax8 = plt.subplot2grid((4,2),(3,1))
    # Absolute NO2 disparities
    ax1.plot(years, np.array(NO2_leastwhite)-np.array(NO2_mostwhite), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False)
    ax1.plot(years, np.array(NO2_black)-np.array(NO2_white), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False)
    ax5.plot(years, np.array(NO2_mosthisp)-np.array(NO2_leasthisp), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False)
    ax5.plot(years, np.array(NO2_hisp)-np.array(NO2_nonhisp), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False)
    # Absolute PM25 disparities
    ax2.plot(years, np.array(PM25_leastwhite)-np.array(PM25_mostwhite), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False)
    ax2.plot(years, np.array(PM25_black)-np.array(PM25_white), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False)
    ax6.plot(years, np.array(PM25_mosthisp)-np.array(PM25_leasthisp), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False)
    ax6.plot(years, np.array(PM25_hisp)-np.array(PM25_nonhisp), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False)
    # Absolute NO2-attributable asthma disparities
    ax3.plot(years, np.array(asthma_leastwhite)-np.array(asthma_mostwhite), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False)
    ax3.plot(years, np.array(asthma_black)-np.array(asthma_white), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False)
    ax7.plot(years, np.array(asthma_mosthisp)-np.array(asthma_leasthisp), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False)
    ax7.plot(years, np.array(asthma_hisp)-np.array(asthma_nonhisp), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False)
    # Absolute NO2-attributable asthma disparities
    ax4.plot(years, np.array(pd_leastwhite)-np.array(pd_mostwhite), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False, label='Top and bottom deciles\n'+\
        '(Least White - Most White)')
    ax4.plot(years, np.array(pd_black)-np.array(pd_white), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False,
        label='Population-weighted\n(Black - White)')
    ax8.plot(years, np.array(pd_mosthisp)-np.array(pd_leasthisp), 
        color='k', marker='o', markerfacecolor='w', markeredgecolor='k', 
        clip_on=False, label='Top and bottom deciles\n'+\
        '(Most Hispanic - Least Hispanic)')
    ax8.plot(years, np.array(pd_hisp)-np.array(pd_nonhisp), color=cscat, 
        marker='o', markerfacecolor='w', markeredgecolor=cscat, clip_on=False, 
        label='Population-weighted\n(Hispanic - Non-Hispanic)')
    # Axes labels  
    ax1.set_title('(A) Racial disparities', fontsize=14, loc='left')
    ax5.set_title('(B) Ethnic disparities', fontsize=14, loc='left')
    ax2.set_title('(C)', fontsize=14, loc='left')
    ax6.set_title('(D)', fontsize=14, loc='left')
    ax3.set_title('(E)', fontsize=14, loc='left')
    ax7.set_title('(F)', fontsize=14, loc='left')
    ax4.set_title('(G)', fontsize=14, loc='left')
    ax8.set_title('(H)', fontsize=14, loc='left') 
    ax1.set_ylabel('NO$_{\mathregular{2}}$ [ppbv]')
    ax2.set_ylabel('PM$_{\mathregular{2.5}}$ [$\mathregular{\mu}$g m$' +
        '^{\mathregular{-3}}$]')
    ax3.set_ylabel('New asthma cases due\nto NO$_{\mathregular{2}}$ ' +
        '[per 100000]')
    ax4.set_ylabel('Premature deaths due\nto PM$_{\mathregular{2.5}}$ ' +
        '[per 100000]')
    for ax in [ax1, ax2]:
        ax.get_yaxis().set_label_coords(-0.2, 0.5)
    for ax in [ax3, ax4]:
        ax.get_yaxis().set_label_coords(-0.15, 0.5)
    # Axes aesthetics 
    for ax in [ax1, ax5]:
        ax.set_ylim([0, 10])
        ax.set_yticks(np.linspace(0, 10, 5))
    ax5.set_yticklabels([])
    for ax in [ax2, ax6]:
        ax.set_ylim([-0.5, 2.5])
        ax.set_yticks(np.linspace(-0.5, 2.5, 5))
    ax6.set_yticklabels([])
    for ax in [ax3, ax7]:
        ax.set_ylim([0, 400])
        ax.set_yticks(np.linspace(0, 400, 5))
    ax7.set_yticklabels([])
    for ax in [ax4, ax8]:
        ax.set_ylim([-10, 10])
        ax.set_yticks(np.linspace(-10, 10, 5))
    ax8.set_yticklabels([])    
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([2009.5, 2019.5])
        ax.spines.bottom.set_bounds((2010, 2019))
        ax.set_xticks(years)
        ax.set_xticklabels(['2010', '', '', '2013', '', '', '2016', '', '',
            '2019'])
    plt.subplots_adjust(hspace=0.7, top=0.95, bottom=0.15)
    ax4.legend(frameon=False, bbox_to_anchor=(0.95, -0.25))
    ax8.legend(frameon=False, bbox_to_anchor=(1, -0.25))
    plt.savefig(DIR_FIG+'figS10_REVISED.pdf', dpi=1000)
    return

# import pandas as pd
# import math
# import time
# from datetime import datetime
# import numpy as np
# from scipy import stats
# import sys
# sys.path.append(DIR)
# # import edf_open, edf_calculate
# # sys.path.append('/Users/ghkerr/GW/tropomi_ej/')
# # import tropomi_census_utils
# sys.path.append('/Users/ghkerr/GW/edf/')
# import pm25no2_constants

# # Load crosswalk to enable subsampling of MSAs
# crosswalk = pd.read_csv(DIR_CROSS+'qcew-county-msa-csa-crosswalk.csv',
#     engine='python', encoding='latin1')
# # Add a leading zero to FIPS codes 0-9
# crosswalk['County Code'] = crosswalk['County Code'].map(lambda x:
#     f'{x:0>5}')
# # Open 2010-2019 harmonized tables and calculate burdens
# burdents = []
# for year in np.arange(2010, 2020, 1):
#     print(year)
#     burden = pd.read_parquet(DIR_HARM+'burdens_%d.gzip'%(year))
#     # Total PM2.5-attributable deaths and new cases of NO2-attributable asthma
#     print('sum(Stroke) = %d'%round(burden.BURDENST.sum()))
#     print('sum(COPD) = %d'%round(burden.BURDENCOPD.sum()))
#     print('sum(Lung cancer) = %d'%round(burden.BURDENLC.sum()))
#     print('sum(Type 2 diabetes) = %d'%round(burden.BURDENDM.sum()))
#     print('sum(Total IHD) = %d'%round(burden.BURDENIHD.sum()))
#     print('sum(Lower respiratory infection) = %d'%round(burden.BURDENLRI.sum()))
#     print('sum(Pediatric asthma) = %d'%round(burden.BURDENPA.sum()))
#     print('Total PM deaths = %d'%round(burden.BURDENST.sum()+
#         burden.BURDENCOPD.sum()+burden.BURDENLC.sum()+burden.BURDENDM.sum()+
#         burden.BURDENIHD.sum()+burden.BURDENLRI.sum()))
#     print('\n')
#     burdents.append(burden)
# burdents = pd.concat(burdents)

# # # # # Subset harmonized tables in MSAs
# burden_msa = []
# geoids = burdents.index.values
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
#     # harm_imsa = harmts.loc[harmts.index.isin(geoids_msa)]
#     # harm_msa.append(harm_imsa)
#     burden_imsa = burdents.loc[burdents.index.isin(geoids_msa)]
#     burden_msa.append(burden_imsa)
# # harm_msa = pd.concat(harm_msa)
# burden_msa = pd.concat(burden_msa)

# pmburden_allmsa, asthmaburden_allmsa = [], []
# pm25_pwm_allmsa, no2_pwm_allmsa = [], []
# pmburdenrate_allmsa, asthmaburdenrate_allmsa = [], []
# lng_allmsa, lat_allmsa, name_allmsa = [], [], []
# burden19 = burdents.loc[burdents.YEAR=='2015-2019']
# geoids = burden19.index.values
# # Loop through MSAs in U.S.
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
#     # Select NO2-attributable burdens for most recent year available (2019)
#     burden_imsa = burden19.loc[burden19.index.isin(geoids_msa)]
#     pm25_pwm_allmsa.append(w_avg(burden_imsa, 'PM25', 'pop_tot'))
#     no2_pwm_allmsa.append(w_avg(burden_imsa, 'NO2', 'pop_tot'))    
#     asthmaburdenrate_allmsa.append(burden_imsa['BURDENPARATE'].mean())
#     asthmaburden_allmsa.append(burden_imsa['BURDENPA'].sum())
#     pmburdenrate_allmsa.append(burden_imsa['BURDENPMALLRATE'].mean())
#     pmburden_allmsa.append(burden_imsa['BURDENPMALL'].sum())
#     lng_allmsa.append(burden_imsa['LNG_CENTROID'].mean())
#     lat_allmsa.append(burden_imsa['LAT_CENTROID'].mean())
#     name_allmsa.append(msa)
# asthmaburdenrate_allmsa = np.array(asthmaburdenrate_allmsa)
# pmburdenrate_allmsa = np.array(pmburdenrate_allmsa)

# Main text figures
fig1()
fig2(burdents)
fig3(lng_allmsa, lat_allmsa, asthmaburdenrate_allmsa, pmburdenrate_allmsa)
fig4(burdents)
fig5(burdents)
fig6(burdents)
# Supplementary figures
figS1()
figS2()
figS3(burdents)
figS4(burdents)
figS5(burdents)
sigS6(burdens)
figS7(burdents)
# Figure S8 is the population-weighted version of Figure 2
figS9(burdents)
figS10(burdents)

""""STATISTICS ABOUT POPULATION AGE STRUCTURE""""
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
# 'pop_f_5-9', 'pop_f_10-14', 'pop_f_15-17', 'pop_f_18-19', 'pop_f_20',
# 'pop_f_21', 'pop_f_22-24']].sum(axis=1)/mosthisp['pop_tot']
# lwp = leasthisp[['pop_m_lt5', 'pop_m_5-9', 'pop_m_10-14',
# 'pop_m_15-17', 'pop_m_18-19', 'pop_m_20', 'pop_m_21', 'pop_m_22-24', 'pop_f_lt5',
# 'pop_f_5-9', 'pop_f_10-14', 'pop_f_15-17', 'pop_f_18-19', 'pop_f_20',
# 'pop_f_21', 'pop_f_22-24']].sum(axis=1)/leasthisp['pop_tot']

""""STATISTICS ABOUT SIZE OF CENSUS TRACTS""""
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
#         area = (record['ALAND']+record['AWATER'])/(1000*1000)
#         geoid = record['GEOID']
#         # if geoid in msa:
#         areas.append(area)

""""STATISTICS ABOUT NUMBER OF STATES INCLUDED IN POPULATION SUBGROUP
    DEFINITIONS""""
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
# year = 2019
# yearst = '%d-%d'%(year-4, year)
# burdenty = burdents.loc[burdents['YEAR']==yearst].copy(deep=True)
# # Define ethnoracial groups
# burdenty['fracwhite'] = ((burdenty[['race_nh_white','race_h_white'
#     ]].sum(axis=1))/burdenty['race_tot'])
# # Define extreme ethnoracial subgroups
# mostwhite = burdenty.iloc[np.where(burdenty.fracwhite >=
#     np.nanpercentile(burdenty.fracwhite, 90))]
# leastwhite = burdenty.iloc[np.where(burdenty.fracwhite <=
#     np.nanpercentile(burdenty.fracwhite, 10))]
# print('For top and bottom deciles of population subgroups (2019')
# print('Number of tracts:')
# print('Most white = %d'%len(mostwhite))
# print('Least white = %d\n'%len(leastwhite))
# print('Number of states represented:')
# print('Most white = %d'%np.unique(mostwhite.STATE).shape[0])
# print('Least white = %d\n'%np.unique(leastwhite.STATE).shape[0])
# mostwhite_urban = np.where(np.in1d(mostwhite.index,
#     harm_urban.index)==True)[0].shape[0]
# leastwhite_urban = np.where(np.in1d(leastwhite.index,
#     harm_urban.index)==True)[0].shape[0]
# print('Fraction urban:')
# print('Most white = %.1f'%(mostwhite_urban/len(mostwhite)))
# print('Least white = %.1f\n'%(leastwhite_urban/len(leastwhite)))