/**
 * Vegetation stress task: point-level dataset construction (Germany, 2017-2024)
 *
 * Draws stratified random points across Germany by ESA WorldCover class
 * (forest, cropland, grassland, other), then for each point and each year
 * 2017-2024 samples:
 *   - growing-season mean NDVI (MODIS MOD13A3, Apr-Sep), used downstream to
 *     compute a per-point standardized anomaly stress label in Python,
 *   - antecedent ERA5-Land climate summaries (preceding winter and spring:
 *     total precipitation, mean 2 m temperature, mean volumetric soil
 *     water), and
 *   - the AlphaEarth annual embedding (64 bands, A00..A63) for that year.
 *
 * The stress label itself is NOT computed here; it is derived in Python as
 * a per-location z-score of growing-season NDVI across 2017-2024 (see
 * src/ for the labeling step). This keeps the anomaly definition and
 * threshold under explicit control.
 *
 * Output: veg_stress_pointlevel.csv (~16,000 rows = 2000 points x 8 years)
 *   columns: id, lon, lat, lc, year, gs_ndvi,
 *            precip_winter, precip_spring, temp_spring, sm_winter, sm_spring,
 *            A00..A63
 *
 * Run in the Earth Engine Code Editor (code.earthengine.google.com).
 */

// ---- 0. SETTINGS ----
var PER_CLASS = 500;            // points per land-cover class (x4 classes)
var SEED      = 42;
var GS_START  = 4, GS_END = 10; // growing-season NDVI window: April-September (half-open [4,10))

var germany = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Germany')).geometry();

// ---- 1. STRATIFIED POINTS BY LAND COVER (ESA WorldCover -> 4 classes) ----
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map');
var lc = wc.remap(
  [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
  [ 1,  4,  3,  2,  4,  4,  4,  4,  4,  4,   4]
).rename('lc');
// 1 = forest, 2 = cropland, 3 = grassland, 4 = other

var raw = lc.stratifiedSample({
  numPoints: PER_CLASS, classBand: 'lc', region: germany,
  scale: 100, seed: SEED, geometries: true
});

// attach a stable id and lon/lat
var rawList = raw.toList(raw.size());
var pts = ee.FeatureCollection(
  ee.List.sequence(0, raw.size().subtract(1)).map(function (i) {
    i = ee.Number(i);
    var f = ee.Feature(rawList.get(i));
    var c = f.geometry().coordinates();
    return f.set({ id: i, lon: c.get(0), lat: c.get(1) });
  })
);
print('Total points:', pts.size());

// ---- 2. PER-YEAR FEATURES AND SAMPLING ----
var era   = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');
var modis = ee.ImageCollection('MODIS/061/MOD13A3');
var aef   = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

var years = ee.List.sequence(2017, 2024);

var allFc = ee.FeatureCollection(years.map(function (y) {
  y = ee.Number(y);

  var gs_ndvi = modis
    .filterDate(ee.Date.fromYMD(y, GS_START, 1), ee.Date.fromYMD(y, GS_END, 1))
    .select('NDVI').mean().multiply(0.0001).rename('gs_ndvi');

  var winter = era.filterDate(ee.Date.fromYMD(y.subtract(1), 12, 1), ee.Date.fromYMD(y, 3, 1));
  var spring = era.filterDate(ee.Date.fromYMD(y, 3, 1),               ee.Date.fromYMD(y, 6, 1));

  var precip_winter = winter.select('total_precipitation_sum').sum().multiply(1000).rename('precip_winter');
  var precip_spring = spring.select('total_precipitation_sum').sum().multiply(1000).rename('precip_spring');
  var temp_spring   = spring.select('temperature_2m').mean().subtract(273.15).rename('temp_spring');
  var sm_winter     = winter.select('volumetric_soil_water_layer_1').mean().rename('sm_winter');
  var sm_spring     = spring.select('volumetric_soil_water_layer_1').mean().rename('sm_spring');

  var clim = gs_ndvi.addBands([precip_winter, precip_spring, temp_spring, sm_winter, sm_spring]);

  var emb = aef.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y.add(1), 1, 1))
               .filterBounds(germany).mosaic();

  // sample coarse climate/NDVI (1 km) first, then embedding (10 m);
  // chaining preserves the point geometry through both reductions
  var s1 = clim.reduceRegions({ collection: pts, reducer: ee.Reducer.first(), scale: 1000, tileScale: 4 });
  var s2 = emb.reduceRegions({ collection: s1,  reducer: ee.Reducer.first(), scale: 10,   tileScale: 4 });

  return s2.map(function (f) { return f.set('year', y); });
})).flatten();

// ---- 3. EXPORT ----
var bands = [];
for (var i = 0; i < 64; i++) { bands.push('A' + (i < 10 ? '0' + i : i)); }
var base = ['id', 'lon', 'lat', 'lc', 'year', 'gs_ndvi',
            'precip_winter', 'precip_spring', 'temp_spring', 'sm_winter', 'sm_spring'];
var selectors = base.concat(bands);

Export.table.toDrive({
  collection: allFc,
  description: 'veg_stress_pointlevel',
  folder: 'GEE_exports',
  fileNamePrefix: 'veg_stress_pointlevel',
  fileFormat: 'CSV',
  selectors: selectors
});
