# Data sources and access

Raw and intermediate modelling tables are not redistributed in this GitHub repository. Source products are public and can be obtained from the providers below. Collection identifiers and versions match those used in the study.

## AlphaEarth annual embeddings
- Product: Google Satellite Embedding V1 (annual), 64 bands (A00 to A63), 10 m.
- Earth Engine collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Reference: Brown et al. (2025), *AlphaEarth Foundations*.

## LUCAS topsoil (SOC target)
- LUCAS 2018 topsoil survey; target variable: organic carbon (OC, g kg-1).
- Source: European Soil Data Centre (ESDAC), Joint Research Centre.
- Reference: Orgiazzi et al. (2018).
- Point set used for extraction: `projects/seismic-relic-481709-r8/assets/LUCAS`.

## Sentinel-2 surface reflectance
- Bands B2, B3, B4, B8, B11, B12 and NDVI.
- Reference: Drusch et al. (2012).

## Sentinel-1 backscatter
- VV, VH and derived metrics VV/VH and VV-VH.
- Reference: Torres et al. (2012).

## SRTM terrain
- Elevation, slope and aspect.
- Reference: Farr et al. (2007).

## ERA5-Land climate
- SOC benchmark: annual soil moisture, summer 2 m temperature, winter total precipitation.
- Vegetation benchmark: preceding-winter and spring precipitation, spring 2 m temperature, and preceding-winter and spring volumetric soil water.
- Reference: Muñoz-Sabater et al. (2021).

## ESA WorldCover
- ESA WorldCover 10 m 2021 v200, remapped to forest, cropland, grassland and other for vegetation sampling.
- Earth Engine: `ESA/WorldCover/v200`.
- Reference: Zanaga et al. (2022).

## MODIS NDVI
- MOD13A3 monthly NDVI, Collection 6.1 (V061).
- Earth Engine: `MODIS/061/MOD13A3`.
- Reference: Didan (2021).

## Local modelling tables

The code expects locally prepared tables under a non-versioned `data/` directory:

```text
data/SOC_master_aligned.csv
data/veg_stress_pointlevel.csv
```

The SOC benchmark retains 772 LUCAS sites after screening and predictor alignment.

For vegetation stress, 2,000 locations were initially sampled. Twenty-four locations were excluded because ERA5-Land predictors were unavailable for all years at those locations, leaving 1,976 locations with complete climate and NDVI records. The prior-year embedding design removes the first available model year, giving a final balanced panel of 1,976 locations × 7 years (2018-2024) = 13,832 point-years.

Predictors are extracted at the sampling coordinates at their native spatial supports; no predictor group is spatially averaged to the MODIS footprint.

## Licensing

Each source product retains its own license and terms of use. Consult the corresponding provider before redistribution.
