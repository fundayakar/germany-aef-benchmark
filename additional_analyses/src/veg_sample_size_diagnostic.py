"""
Diagnostic: why the vegetation-stress panel has 1,976 unique locations
(13,832 point-years) rather than the 2,000 originally sampled.

Finding: 24 of the 2,000 sampled locations are missing all five ERA5-Land
climate variables in ALL EIGHT of their years (24 x 8 = 192 rows, exactly
accounting for the reduction). AlphaEarth embeddings have zero missing values
across the full 16,000-row export; MODIS NDVI is missing in only 8 rows, all
of which are a subset of the same 192 climate-missing rows. The 24 affected
locations are geographically concentrated in Germany's northernmost coastal
strip (53.3-54.9 deg N, North Sea to Baltic coastline), consistent with (but
not independently confirmed against) ERA5-Land's known land-sea masking near
coastlines.
"""
import pandas as pd

VEG_PATH = "veg_stress_pointlevel.csv"
CLIM = ['precip_winter', 'precip_spring', 'temp_spring', 'sm_winter', 'sm_spring']
BANDS = [f"A{i:02d}" for i in range(64)]

df = pd.read_csv(VEG_PATH)
print(f"Raw export: {df.shape[0]} rows, {df['id'].nunique()} unique locations, "
      f"{df['year'].nunique()} years -> expected {df['id'].nunique()*df['year'].nunique()} rows if complete")
assert (df.groupby('id').size() == 8).all(), "unexpected: not every location has 8 raw rows"

print(f"\nMissing gs_ndvi: {df['gs_ndvi'].isna().sum()} rows")
for c in CLIM:
    print(f"Missing {c}: {df[c].isna().sum()} rows")
print(f"Missing any AlphaEarth band (A00 as proxy): {df['A00'].isna().sum()} rows")

missing_clim = df[df[CLIM].isna().any(axis=1)]
print(f"\nRows missing >=1 climate variable: {len(missing_clim)}")
print(f"Unique locations affected: {missing_clim['id'].nunique()}")

per_id_missing = missing_clim.groupby('id').size()
print("\nDistribution of (# missing years) per affected location:")
print(per_id_missing.value_counts().sort_index())
dropped_ids = per_id_missing[per_id_missing == 8].index.tolist()
print(f"\nLocations missing climate in ALL 8 years (fully dropped): {len(dropped_ids)}")

# geography of the dropped locations
sub = df[df['id'].isin(dropped_ids)].drop_duplicates('id')[['id', 'lon', 'lat', 'lc']]
print("\nLand-cover class distribution among dropped locations "
      "(1=forest, 2=cropland, 3=grassland, 4=other):")
print(sub['lc'].value_counts().sort_index())
print(f"\nLatitude range of dropped locations: {sub['lat'].min():.3f} - {sub['lat'].max():.3f}")
allpts = df.drop_duplicates('id')
print(f"Latitude range of all 2,000 sampled locations: {allpts['lat'].min():.3f} - {allpts['lat'].max():.3f}")

# confirm the final panel: 1,976 locations x 7 years, fully balanced
step2 = df.dropna(subset=['gs_ndvi'] + CLIM).copy().sort_values(['id', 'year'])
prev = step2.groupby('id')[BANDS].shift(1)
prev.columns = [b + '_p' for b in BANDS]
step2 = pd.concat([step2, prev], axis=1)
final = step2.dropna(subset=[b + '_p' for b in BANDS]).reset_index(drop=True)
print(f"\nFinal panel: {final.shape[0]} point-years, {final['id'].nunique()} locations")
years_per_id = final.groupby('id')['year'].apply(lambda x: tuple(sorted(x.astype(int))))
print(f"Distinct year-sets across all retained locations: {years_per_id.nunique()} "
      f"(1 means fully balanced panel)")
print(f"Year-set: {years_per_id.unique()}")
