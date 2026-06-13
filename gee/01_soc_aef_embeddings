/**
 * AlphaEarth (Google Satellite Embedding V1) -> LUCAS points (SOC task)
 *
 * Samples the 2018 annual AlphaEarth embedding (64 bands, A00..A63) at the
 * LUCAS 2018 topsoil survey points and exports lon/lat alongside the
 * embedding so the result can be joined to the engineered feature table
 * (see gee/02_soc_engineered_stack.js) without relying on a shared ID.
 *
 * Output: AEF_embeddings_LUCAS_2018_coords.csv
 *   columns: A00..A63, POINTID, lon, lat
 *
 * Run in the Earth Engine Code Editor (code.earthengine.google.com).
 * Edit LUCAS_ASSET to point at your own LUCAS feature collection.
 */

// ---- 1. USER INPUTS ----
var LUCAS_ASSET = 'projects/seismic-relic-481709-r8/assets/LUCAS'; // <- your asset path
var YEAR        = 2018;   // LUCAS 2018 sampling year
var SCALE       = 10;     // AEF native resolution (m)
var OUT_NAME    = 'AEF_embeddings_LUCAS_2018_coords';

// ---- 2. LOAD POINTS AND EMBEDDINGS ----
var LUCAS = ee.FeatureCollection(LUCAS_ASSET);

var emb = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate(YEAR + '-01-01', (YEAR + 1) + '-01-01')
  .filterBounds(LUCAS)
  .mosaic();   // single 64-band image over the AOI

// ---- 3. SAMPLE AT POINTS, ATTACH COORDINATES ----
var sampled = emb.reduceRegions({
  collection: LUCAS,
  reducer: ee.Reducer.first(),
  scale: SCALE,
  tileScale: 4   // increase to 8 or 16 if you hit "computed value too large"
}).map(function (f) {
  var c = f.geometry().coordinates();
  return f.set({ lon: c.get(0), lat: c.get(1) });
});

// ---- 4. EXPORT ----
var bandList = [];
for (var i = 0; i < 64; i++) { bandList.push('A' + (i < 10 ? '0' + i : i)); }
var selectors = bandList.concat(['POINTID', 'lon', 'lat']);

Export.table.toDrive({
  collection: sampled,
  description: OUT_NAME,
  folder: 'GEE_exports',
  fileNamePrefix: OUT_NAME,
  fileFormat: 'CSV',
  selectors: selectors
});
