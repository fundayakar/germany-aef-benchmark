"""
Exports the exact spatial-CV block assignments used throughout the manuscript
(k=10, seed=42 k-means on coordinates). KMeans with a fixed random_state is
deterministic given identical input data and order, so re-running this exact
call reproduces the same block assignment used to produce Table 1 and every
downstream analysis -- it is not a new/different clustering. This has been
cross-checked throughout the revision: every re-run of the locked pipeline
(fold-level results, sensitivity analyses, etc.) reproduces Table 1's numbers
exactly, which would not happen if the block assignment varied between runs.

Outputs:
  soc_points_blocks.csv        : site_id, lon, lat, block_id, OC_gkg (n=772)
  vegetation_points_blocks.csv : point_id, lon, lat, block_id, land_cover
                                  (n=1,976 unique locations, not point-years)
"""
import pandas as pd
from sklearn.cluster import KMeans

SEED = 42
N_BLOCKS = 10

# ============ SOC ============
soc = pd.read_csv("SOC_master_aligned.csv")
soc_blk = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(soc[['lon', 'lat']].values)
soc_out = pd.DataFrame({
    'site_id': soc['POINTID'], 'lon': soc['lon'], 'lat': soc['lat'],
    'block_id': soc_blk, 'OC_gkg': soc['Lucas_OC'],
})
soc_out.to_csv("soc_points_blocks.csv", index=False)
print("=== SOC ===")
print("n points:", len(soc_out), "| n unique blocks:", soc_out['block_id'].nunique())
print(soc_out['block_id'].value_counts().sort_index())

# ============ Vegetation (one row per unique location) ============
CLIM = ['precip_winter', 'precip_spring', 'temp_spring', 'sm_winter', 'sm_spring']
BANDS = [f"A{i:02d}" for i in range(64)]
veg = pd.read_csv("veg_stress_pointlevel.csv")
veg = veg.dropna(subset=['gs_ndvi'] + CLIM).copy()
veg = veg.sort_values(['id', 'year'])
prev = veg.groupby('id')[BANDS].shift(1); prev.columns = [b + '_p' for b in BANDS]
veg = pd.concat([veg, prev], axis=1)
veg = veg.dropna(subset=[b + '_p' for b in BANDS]).reset_index(drop=True)  # -> 1,976 retained locations

pts = veg.drop_duplicates('id')[['id', 'lon', 'lat', 'lc']].copy().reset_index(drop=True)
pts_blk = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(pts[['lon', 'lat']].values)
lc_map = {1: 'forest', 2: 'cropland', 3: 'grassland', 4: 'other'}
veg_out = pd.DataFrame({
    'point_id': pts['id'], 'lon': pts['lon'], 'lat': pts['lat'],
    'block_id': pts_blk, 'land_cover': pts['lc'].map(lc_map),
})
veg_out.to_csv("vegetation_points_blocks.csv", index=False)
print("\n=== Vegetation ===")
print("n unique locations:", len(veg_out), "| n unique blocks:", veg_out['block_id'].nunique())
print(veg_out['block_id'].value_counts().sort_index())
