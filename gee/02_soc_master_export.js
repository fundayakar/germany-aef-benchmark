/**
 * SOC task: master export (OC + engineered stack + AEF embedding)
 *
 * Samples the 2018 AlphaEarth embedding directly over a user-provided point
 * set that already carries the LUCAS organic-carbon target (Lucas_OC / OC)
 * and the 17 engineered predictors (Sentinel-1/2, SRTM terrain, ERA5-Land
 * seasonal summaries). The export joins everything on a single row per
 * point, with no separate ID-based join required afterwards.
 *
 * Input: a point feature collection uploaded as an Earth Engine asset
 *   (CSV with lon/lat columns, X = lon, Y = lat, CRS EPSG:4326), containing:
 *     POINTID, Lucas_OC, lon, lat,
 *     B2, B3, B4, B8, B11, B12, NDVI,
 *     VV, VH, VV_div_VH, VV_minus_VH,
 *     aspect, elev, slope, sm_annual, t2m_summer, tp_winter
 *
 * Output: SOC_master.csv
 *   columns: POINTID, Lucas_OC, lon, lat, <17 engineered features>, A00..A63
 *
 * Run in the Earth Engine Code Editor (code.earthengine.google.com).
 * Edit POINTS_ASSET to point at your uploaded asset.
 */

// ---- 1. USER INPUTS ----
var POINTS_ASSET = 'projects/seismic-relic-481709-r8/assets/LUCAS_OC_features'; // <- your asset path
var YEAR         = 2018; // LUCAS 2018 sampling year
var SCALE        = 10;   // AEF native resolution (m)
var OUT_NAME     = 'SOC_master';

// ---- 2. LOAD POINTS AND EMBEDDING ----
var pts = ee.FeatureCollection(POINTS_ASSET);

var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
  .filterBounds(pts)
  .mosaic();

// ---- 3. SAMPLE EMBEDDING AT POINTS (existing properties are preserved) ----
var master = emb.reduceRegions({
  collection: pts,
  reducer: ee.Reducer.first(),
  scale: SCALE,
  tileScale: 4 // increase to 8 or 16 if you hit "computed value too large"
});

// ---- 4. EXPORT (your fields first, then A00..A63) ----
var bands = [];
for (var i = 0; i < 64; i++) { bands.push('A' + (i < 10 ? '0' + i : i)); }
var feats = ['POINTID', 'Lucas_OC', 'lon', 'lat',
              'B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI',
              'VV', 'VH', 'VV_div_VH', 'VV_minus_VH',
              'aspect', 'elev', 'slope', 'sm_annual', 't2m_summer', 'tp_winter'];
var selectors = feats.concat(bands);

Export.table.toDrive({
  collection: master,
  description: OUT_NAME,
  folder: 'GEE_exports',
  fileNamePrefix: OUT_NAME,
  fileFormat: 'CSV',
  selectors: selectors
});
