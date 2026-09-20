# Wildfire Data Mining & ML Analysis

## Overview

This project tests whether landscape and firefighting-access attributes predict wildfire
escape and final size more accurately than fire-weather attributes, across the entire
occurrence-to-size lifecycle, using a matched 9 vs 9 feature comparison on
768,255 U.S. wildfires (2011-2020), with Portugese and Algerian data as
validation for occurrence and escape, respectively.

## Data

Three publicly available datasets, none included in this repo due to size/licensing:
(USA) FPA FOD-Attributes [10.5281/zenodo.8381129](https://doi.org/10.5281/zenodo.8381129) 
(Portugal) Forest Fires in Montesinho, Portugal: [10.24432/C5D88D](https://doi.org/10.24432/C5D88D)
(Algeria) Algerian Forest Fire [10.24432/C5KW4N](https://doi.org/10.24432/C5KW4N) 


Download the FPA FOD-Attributes annual CSVs (2011-2020) into `data/fpa/` before running
the pipeline below.

## Pipeline
1. **`accuracy_final.py`** — cleans the raw data and leakage and builds `data/prepped.pkl`. Every later script loads
   this cache.
2. **`matched_test_final.py`** — matched 9-vs-9 weather-vs-landscape comparison across
   random/spatial-block/temporal cross-validation.
3. **`regional_final.py`** — repeats the comparison within the three largest Level-II
   ecoregion.
4. **`model_robustness.py`** — repeating the comparison under random forest, ridge
   regression and logistic regression.
5. **`weather_fair_test.py`** — expands the weather set to a 5-day window with climate
   percentiles, to test whether more generous weather feature can close the gap observed.
6. **`make_poster_figures.py`** — generate all figures necessary relating to fire-danger indices, Spearman figures, Simpson Paradox results, etc. 

The 20 Level-II EPA Ecoregions used were: Mississippi Alluvial Plain, Southeastern USA Plains, South Central Semi-Arid Prairies, Western Cordillera, Ozark/Ouachita–Appalachian Forests, Mixed Wood Plains, Mediterranean California, Cold Deserts, Warm Deserts, Temperate Prairies, West-Central Semi-Arid Prairies, Mixed Wood Shield, Upper Gila Mountains, Atlantic Highlands, Marine West Coast Forest, Texas-Louisiana Coastal Plain, Central USA Plains, Tamaulipas-Texas Semi-Arid Plain, Western Sierra Madre Piedmont, Everglades
