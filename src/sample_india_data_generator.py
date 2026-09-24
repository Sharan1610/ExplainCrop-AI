"""
ExplainCrop-AI / CropMind AI: India Sample Raw Multimodal Data Generator
Generates realistic sample files across all required raw formats:
  - Satellite: GeoTIFF (.tif)
  - Climate: NetCDF (.nc)
  - Soil: GeoTIFF (.tif) & CSV (.csv)
  - Yield: Excel (.xlsx), CSV (.csv), JSON (.json)
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import numpy as np
import pandas as pd

import tifffile
import netCDF4 as nc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SAT_DIR = os.path.join(RAW_DIR, "satellite")
CLI_DIR = os.path.join(RAW_DIR, "climate")
SOIL_DIR = os.path.join(RAW_DIR, "soil")
YLD_DIR = os.path.join(RAW_DIR, "yield")

for d in [SAT_DIR, CLI_DIR, SOIL_DIR, YLD_DIR]:
    os.makedirs(d, exist_ok=True)


def generate_satellite_geotiffs():
    """Generates synthetic 4-band Sentinel-2 GeoTIFFs (B2, B3, B4, B8) for Indian regions."""
    np.random.seed(42)
    H, W = 64, 64  # 64x64 spatial field patch (~640m x 640m at 10m res)

    # 1. Punjab Wheat Belt (High NIR, High NDVI)
    blue = np.random.uniform(0.02, 0.05, (H, W)).astype(np.float32)
    green = np.random.uniform(0.04, 0.08, (H, W)).astype(np.float32)
    red = np.random.uniform(0.03, 0.06, (H, W)).astype(np.float32)
    nir = np.random.uniform(0.35, 0.55, (H, W)).astype(np.float32)
    punjab_s2 = np.stack([blue, green, red, nir], axis=0)
    tifffile.imwrite(os.path.join(SAT_DIR, "sentinel2_punjab_wheat_belt.tif"), punjab_s2)

    # 2. Cauvery Delta Rice Paddy (High Humidity/Water, Strong Green/NIR)
    blue_r = np.random.uniform(0.04, 0.08, (H, W)).astype(np.float32)
    green_r = np.random.uniform(0.06, 0.12, (H, W)).astype(np.float32)
    red_r = np.random.uniform(0.05, 0.09, (H, W)).astype(np.float32)
    nir_r = np.random.uniform(0.40, 0.65, (H, W)).astype(np.float32)
    cauvery_s2 = np.stack([blue_r, green_r, red_r, nir_r], axis=0)
    tifffile.imwrite(os.path.join(SAT_DIR, "sentinel2_cauvery_rice_paddy.tif"), cauvery_s2)

    print("[SUCCESS] Created Satellite GeoTIFF files in data/raw/satellite/")


def generate_climate_netcdf():
    """Generates synthetic NetCDF climate time series for Indian Kharif/Rabi seasons."""
    nc_path = os.path.join(CLI_DIR, "imd_gridded_monsoon_climate.nc")

    days = 120  # Season days
    lat_len = 10
    lon_len = 10

    ds = nc.Dataset(nc_path, "w", format="NETCDF4")
    ds.createDimension("time", days)
    ds.createDimension("lat", lat_len)
    ds.createDimension("lon", lon_len)

    # Coordinates
    lat_var = ds.createVariable("lat", "f4", ("lat",))
    lon_var = ds.createVariable("lon", "f4", ("lon",))
    lat_var[:] = np.linspace(10.0, 32.0, lat_len)
    lon_var[:] = np.linspace(72.0, 88.0, lon_len)

    # Variables: rainfall (mm), temperature (°C), relative humidity (%)
    rain_var = ds.createVariable("rainfall", "f4", ("time", "lat", "lon"))
    temp_var = ds.createVariable("temperature", "f4", ("time", "lat", "lon"))
    rh_var = ds.createVariable("relative_humidity", "f4", ("time", "lat", "lon"))

    np.random.seed(101)
    rain_var[:] = np.random.gamma(shape=2.0, scale=1.2, size=(days, lat_len, lon_len)).astype(np.float32)
    temp_var[:] = np.random.normal(loc=26.5, scale=3.5, size=(days, lat_len, lon_len)).astype(np.float32)
    rh_var[:] = np.random.uniform(60.0, 92.0, size=(days, lat_len, lon_len)).astype(np.float32)

    ds.close()
    print("[SUCCESS] Created Climate NetCDF file in data/raw/climate/")


def generate_soil_data():
    """Generates SoilGrids GeoTIFF and Soil Health Card CSV."""
    np.random.seed(88)
    H, W = 64, 64

    # 1. Soil pH GeoTIFF (e.g. 6.5 - 7.5 range)
    ph_grid = np.random.uniform(6.2, 7.8, (H, W)).astype(np.float32)
    tifffile.imwrite(os.path.join(SOIL_DIR, "soilgrids_india_ph_topsoil.tif"), ph_grid)

    # 2. Soil Health Card District CSV
    shc_data = {
        "District": ["Ludhiana", "Thanjavur", "Nashik", "Indore", "Guntur", "Varanasi", "Kurnool"],
        "State": ["Punjab", "Tamil Nadu", "Maharashtra", "Madhya Pradesh", "Andhra Pradesh", "Uttar Pradesh", "Andhra Pradesh"],
        "Nitrogen_kg_ha": [92.5, 88.0, 45.0, 55.0, 78.0, 82.0, 50.0],
        "Phosphorus_kg_ha": [55.0, 42.0, 68.0, 50.0, 48.0, 52.0, 44.0],
        "Potassium_kg_ha": [45.0, 40.0, 78.0, 52.0, 38.0, 46.0, 35.0],
        "Soil_pH": [7.2, 6.6, 7.6, 7.4, 6.9, 7.1, 7.8],
        "Organic_Carbon_pct": [0.72, 0.68, 0.54, 0.60, 0.65, 0.70, 0.50],
    }
    df_shc = pd.DataFrame(shc_data)
    df_shc.to_csv(os.path.join(SOIL_DIR, "india_soil_health_card_districts.csv"), index=False)

    print("[SUCCESS] Created Soil GeoTIFF and CSV in data/raw/soil/")


def generate_yield_records():
    """Generates ICRISAT / DES APY format files in Excel, CSV, and JSON."""
    yield_data = {
        "State": ["Punjab", "Tamil Nadu", "Maharashtra", "Madhya Pradesh", "Karnataka", "West Bengal", "Gujarat"],
        "District": ["Ludhiana", "Thanjavur", "Nashik", "Indore", "Dharwad", "Burdwan", "Anand"],
        "Season": ["Rabi", "Kharif", "Kharif", "Rabi", "Kharif", "Kharif", "Kharif"],
        "Crop": ["Wheat", "Rice", "Cotton", "Chickpea", "Maize", "Jute", "Banana"],
        "Area_ha": [125000, 95000, 82000, 64000, 71000, 55000, 24000],
        "Production_tonnes": [625000, 380000, 164000, 115200, 248500, 192500, 1200000],
        "Yield_kg_per_ha": [5000.0, 4000.0, 2000.0, 1800.0, 3500.0, 3500.0, 50000.0],
    }
    df_yield = pd.DataFrame(yield_data)

    # 1. Excel (.xlsx)
    df_yield.to_excel(os.path.join(YLD_DIR, "icrisat_district_crop_yield.xlsx"), index=False)

    # 2. CSV (.csv)
    df_yield.to_csv(os.path.join(YLD_DIR, "ministry_agri_apy_yield.csv"), index=False)

    # 3. JSON (.json)
    df_yield.to_json(os.path.join(YLD_DIR, "india_crop_yield_benchmarks.json"), orient="records", indent=2)

    print("[SUCCESS] Created Yield files in Excel, CSV, and JSON in data/raw/yield/")


if __name__ == "__main__":
    generate_satellite_geotiffs()
    generate_climate_netcdf()
    generate_soil_data()
    generate_yield_records()
    print("\n[COMPLETE] All multimodal raw India sample datasets generated successfully!")
