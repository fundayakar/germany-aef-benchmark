"""
Vegetation stress task: label construction and antecedent embedding shift.

Input:  veg_stress_pointlevel.csv (from gee/03_vegetation_pointlevel.js)
        one row per point-year (2000 points x 8 years, 2017-2024), with
        columns: id, lon, lat, lc, year, gs_ndvi,
                 precip_winter, precip_spring, temp_spring, sm_winter, sm_spring,
                 A00..A63 (that year's AlphaEarth embedding)

Steps:
  1. Drop rows with missing growing-season NDVI or antecedent climate.
  2. Stress label: for each point, standardize growing-season NDVI across
     its own 2017-2024 years (z-score). A point-year is labelled stressed
     (1) when z < -1, unstressed (0) otherwise.
  3. Antecedent embedding: for each point, shift the AlphaEarth embedding
     bands forward by one year (each year's predictors use the PREVIOUS
     year's embedding, suffix "_prev"). This avoids circularity, since the
     same-year embedding would encode the growing-season surface state from
     which the NDVI-based label is derived. This drops the first year (2017)
     per point, leaving 2018-2024 (7 years x 2000 points = 13832 point-years
     after the NDVI/climate completeness check).

Output: veg_labeled.csv
  columns: id, lon, lat, lc, year, gs_ndvi,
           precip_winter, precip_spring, temp_spring, sm_winter, sm_spring,
           A00..A63 (same-year, unused as predictors but kept for reference),
           ndvi_mean, ndvi_std, z, stress (0/1),
           A00_prev..A63_prev (antecedent embedding predictors)
"""

import pandas as pd

AEF_BANDS = [f"A{i:02d}" for i in range(64)]
CLIMATE_COLS = ["precip_winter", "precip_spring", "temp_spring", "sm_winter", "sm_spring"]

INPUT_CSV = "veg_stress_pointlevel.csv"
OUTPUT_CSV = "veg_labeled.csv"

STRESS_Z_THRESHOLD = -1.0  # z < threshold -> stress = 1

df = pd.read_csv(INPUT_CSV)

# ---- 1. Drop rows with missing NDVI or antecedent climate ----
df = df.dropna(subset=["gs_ndvi"] + CLIMATE_COLS).copy()

# ---- 2. Stress label: per-point standardized NDVI anomaly ----
g = df.groupby("id")["gs_ndvi"]
df["ndvi_mean"] = g.transform("mean")
df["ndvi_std"] = g.transform("std")
df["z"] = (df["gs_ndvi"] - df["ndvi_mean"]) / df["ndvi_std"]
df["stress"] = (df["z"] < STRESS_Z_THRESHOLD).astype(int)

print(f"Rows after cleaning: {len(df)}")
print(f"Stress prevalence: {df['stress'].mean():.3f} "
      f"(n stressed = {int(df['stress'].sum())})")

# ---- 3. Antecedent embedding: previous-year AlphaEarth vector per point ----
df = df.sort_values(["id", "year"])
prev = df.groupby("id")[AEF_BANDS].shift(1)
prev.columns = [f"{b}_prev" for b in AEF_BANDS]
df = pd.concat([df, prev], axis=1)

prev_bands = [f"{b}_prev" for b in AEF_BANDS]
df = df.dropna(subset=prev_bands).copy()  # drops each point's first year (2017)

print(f"Rows with antecedent embedding: {len(df)} "
      f"| years: {sorted(df['year'].unique().astype(int).tolist())}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {OUTPUT_CSV}")
