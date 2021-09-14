#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:35:17 2021

@author: ghkerr
"""

DIR = '/Users/ghkerr/GW/edf/'
DIR_HARM = DIR+'anenberg_mohegh_no2/harmonizedtables/'
DIR_GEO = '/Users/ghkerr/GW/data/geography/'
DIR_OUT = '/Users/ghkerr/GW/data/anenberg_mohegh_no2/harmonizedshp/'

import netCDF4 as nc
import pandas as pd
import shapefile
import numpy as np   
import sys
sys.path.append(DIR)
import edf_open, edf_calculate
sys.path.append('/Users/ghkerr/GW/tropomi_ej/')

fips = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', 
    '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
    '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
    '36', '37', '38', '39', '40', '41', '42', '44', '45', '46', '47', 
    '48', '49', '50', '51', '53', '54', '55', '56', '72']
# Calculate burden for 2019
vintage = '2015-2019'
harm = edf_open.load_vintageharmonized(vintage)
harm = edf_calculate.calculate_harmburden(harm)
# Loop through states/territories and create new shapefiles with AF
for statefips in fips: 
    print('Handling %s...'%statefips)
    # Read in our existing shapefile for state of interest
    fname = DIR_GEO+'tigerline/tract_2010/tl_2019_%s_tract/'%statefips
    r = shapefile.Reader(fname+'tl_2019_%s_tract.shp'%statefips)
    # Create a new shapefile in memory
    w = shapefile.Writer(DIR_OUT+'%s/asthma_af2019_%s_v1'%(statefips, statefips))
    
    # Copy over the existing fields
    w.fields = list(r.fields)
    # To help prevent accidental misalignment PyShp has an "auto balance" 
    # feature to make sure when you add either a shape or a record the two 
    # sides of the equation line up. This way if you forget to update an entry 
    # the shapefile will still be valid and handled correctly by most shapefile 
    # software. Autobalancing is NOT turned on by default. To activate it set 
    # the attribute autoBalance to 1 or True:
    w.autoBalance = 1
    
    # Add our new field using the pyshp API; note that each field is a Python 
    # list with the following information:
    # Field name: the name describing the data at this column index.
    # Field type: the type of data at this column index. Types can be:
    #   "C": Characters, text.
    #   "N": Numbers, with or without decimals.
    #   "F": Floats (same as "N").
    #   "L": Logical, for boolean True/False values.
    #   "D": Dates.
    #   "M": Memo, has no meaning within a GIS and is part of the xbase spec 
    #        instead.
    # Field length: the length of the data found at this column index. Older GIS 
    #   software may truncate this length to 8 or 11 characters for "Character" 
    #   fields.
    # Decimal length: the number of decimal places found in "Number" fields.
    w.field("KHREISAF", "F", 8, 3)
    w.field("KHREISAF_UPPER", "F", 8, 3)
    w.field("KHREISAF_LOWER", "F", 8, 3)
    
    # Loop through each record, add a column.  We'll
    # insert our sample data but you could also just
    # insert a blank string or NULL DATA number
    # as a place holder
    for shaperec in r.iterShapeRecords():
        geoid = shaperec.record.GEOID
        # Find AF for tract
        tract = harm.loc[harm.index==geoid]
        khreisaf = tract['AF'].values[0]
        shaperec.record.append(khreisaf)
        khreisafupper = tract['AFUPPER'].values[0]
        shaperec.record.append(khreisafupper)    
        khreisaflower = tract['AFLOWER'].values[0]
        shaperec.record.append(khreisaflower)
        # Add the modified record  and existing shape to the new shapefile; 
        # for additional details see https://pypi.org/project/pyshp/#adding-geometry
        w.record(*shaperec.record)
        w.shape(shaperec.shape)
    w.close()    
        
# # Check to ensure operation worked 
# import matplotlib.pyplot as plt
# for statefips in fips: 
#     mod = shapefile.Reader(DIR_OUT+'%s/asthma_af2019_%s_v1'%(statefips, statefips))
#     tracts_mod = mod.shapes()
#     records_mod = mod.records()
#     lats, lngs, afs, ids = [], [], [], []
#     for i, rec in enumerate(records_mod):
#         lats.append(float(rec.INTPTLAT))
#         lngs.append(float(rec.INTPTLON))
#         afs.append(rec.KHREISAF)
#         ids.append(rec.GEOID)
#     plt.scatter(lngs, lats, c=afs, vmin=0, vmax=0.3, cmap=plt.get_cmap('magma'))
#     plt.colorbar(extend='max')
#     plt.title(harm.loc[harm['STATEA']==int(statefips)]['STATE'][0])
#     plt.show()