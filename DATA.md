# Data sources and access

No raw data are redistributed in this repository. All inputs are public and can be obtained from the sources below. Collection identifiers and versions match those used in the paper.

## AlphaEarth annual embeddings
- Product: Google Satellite Embedding V1 (annual), 64 bands (A00 to A63), 10 m.
- Earth Engine collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Reference: Brown et al. (2025), AlphaEarth Foundations.

## LUCAS topsoil (soil organic carbon target)
- LUCAS 2018 topsoil survey; target variable: organic carbon (OC, g kg-1).
- Source: European Soil Data Centre (ESDAC), Joint Research Centre.
- Reference: Orgiazzi et al. (2018), European Journal of Soil Science.
- Extraction point set used here is held as an Earth Engine asset:
  `projects/seismic-relic-481709-r8/assets/LUCAS`

## Sentinel-2 surface reflectance
- Bands B2, B3, B4, B8, B11, B12 and NDVI.
- Earth Engine: Sentinel-2 surface reflectance collection.
- Reference: Drusch et al. (2012).

## Sentinel-1 backscatter
- VV, VH and derived ratios (VV/VH, VV-VH).
- Earth Engine: Sentinel-1 GRD collection.
- Reference: Torres et al. (2012).

## SRTM terrain
- Elevation, slope, aspect.
- Reference: Farr et al. (2007).

## ERA5-Land climate
- Soil moisture, 2 m temperature, total precipitation (seasonal summaries).
- Reference: Muñoz-Sabater et al. (2021).

## ESA WorldCover (vegetation sampling strata)
- ESA WorldCover 10 m 2021 v200, remapped to four classes (forest, cropland, grassland, other).
- Earth Engine: `ESA/WorldCover/v200`
- Reference: Zanaga et al. (2022).

## MODIS NDVI (vegetation stress label)
- MOD13A3 monthly NDVI, Collection 6.1 (V061).
- Earth Engine: `MODIS/061/MOD13A3`
- Reference: Didan (2021).

## Notes
- Spatial alignment: predictors are sampled at the analysis points; the soil set retains 772 points after screening, the vegetation set is a 2000-point, eight-year panel (13832 point-years after dropping 2017 for the prior-year embedding).
- Licenses: each product retains its own terms of use. Consult the provider before redistribution.
