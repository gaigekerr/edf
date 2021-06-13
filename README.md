This file provides detailed documentation for the data and methodology used for the EDF project to develop a Mapbox-based visualization of NO<sub>2</sub>-pediatric asthma attributable fraction for the full extent of the United States (#1052-000000-10400-100-00) and the repository contents. 

# Data source

## NO<sub>2</sub>
Surface-level NO<sub>2</sub> data at 1 km x 1 km resolution from [figshare](https://figshare.com/articles/dataset/Global_surface_NO2_concentrations_1990-2020/12968114/4) was trimmed to include the United States and Puerto Rico with the following command using the gdal library: 
```
gdalwarp -ts 14184 6248 -te -179.1479159 17.8022 -60.0023159 71.3936168 -t_srs metafile.wkt -wo SOURCE_EXTRA=10 -r average -co TILED=YES -overwrite -co compress=LZW YYYY_final_1km.tif YYYY_final_1km_usa.tif
```
Note that the well-known text `metadata.wkt` file has the following contents: 
```
GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]
```
These trimmed files are stored `/GWSPH/groups/anenberggrp/ghkerr/data/edf/no2/`.

## Census 
The following variables were downloaded from 5-year ACS data from https://www.nhgis.org on 27 May 2021: 
1. Median Household Income in the Past 12 Months (in YYYY Inflation-Adjusted Dollars)
2. Educational Attainment for the Population 25 Years and Over
3. Hispanic or Latino Origin by Race
4. Tenure by Vehicles Available
5. Nativity in the United States
6. Gini Index of Income Inequality
7. Sex by age
The YYYY corresponds to the final year in the 5-year ACS span. 
TIGER/Line Shapefiles for the 50 U.S. states, District of Columbia, and Puerto Rico downloaded for 
For ACS 2006-2010, 2007-2011, 2008-2012, 2009-2013, 2010-2014, 2011-2015, 2012-2016, 2013-2017, 2014-2018, and 2015-2019: 
https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2019&layergroup=Census+Tracts
For ACS 2005-2009: 
https://www2.census.gov/geo/tiger/TIGER2009/

FIPS-MSA crosswalk (downloaded 7 June 2021): note The table below matches QCEW counties with Metropolitan Statistical Areas (MSAs) and Combined Statistical Areas (CSAs), as defined by the U.S. Census Bureau's February 2013 Core-Based Statistical Areas (CBSA) delineation file (available at https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2013/delineation-files/list1.xls).

SITE: https://www.bls.gov/cew/classifications/areas/county-msa-csa-crosswalk.htm
FILE: https://www.bls.gov/cew/classifications/areas/qcew-county-msa-csa-crosswalk-csv.csv

## Pediatric asthma incidence
Asthma burden/risk: http://ghdx.healthdata.org/gbd-results-tool
Age-specific asthma incidence rates for 2019 with trends since 1990 downloaded on 7 June 2021 for 
Location: 50 US states, District of Columbia, Puerto Rico, and overall US
Age: 1 to 4, 5 to 9, 10 to 14, 15 to 19
Year: 2009-2019
Measure: Incidence
Cause: B.3.3 Asthma
Context: Cause
Sex: Both
Metric: Rate
