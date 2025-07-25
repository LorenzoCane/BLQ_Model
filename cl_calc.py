"""
    Lift Coefficient calibration
    
    this script calibrates the lift coefficient (C_L) for a given aircraft and engine by fitting
    modeled takeoff distances to manufacturer data. Computes takeoff distances using
    the optimized C_L, estimates uncertainty bounds, and saves results with metadata
    to a Parquet file for further analysis.

    It depends on the OpenAP library and project-specific modules.

    Author: Lorenzo Cane - DBL E&E Area Consultant
    Last modified: 20/06/2025
"""


import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys
from config_loader import load_config
from performance_functions import cl_finder, calc_todr
from aircraft_utils import get_aircraft_data, get_thrust, get_to_speed
from constants import *
from openap import prop #aircraft and engine-related data
sys.path.insert(0,'./utils')
from unit_converter import ComplexUnitConverter as conv

#***************************************************************************
#import from configuration file config.yml & create dirs
config = load_config()

cl_path = config['Dir']['cl_dir']
aircraft_name = config["Aircraft"]["aircr_name"]
engine_name = config["Aircraft"]["aircr_engine"]

aircraft_info = get_aircraft_data(aircraft_name, engine_name)

wing_area = aircraft_info["wing_area"]
cd0 = aircraft_info["cd0"]
k = aircraft_info["k"]
mu = aircraft_info["mu"]
mass_max = aircraft_info["mass_max"]

asc_m = conv.convert(ASC, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(CLIMB_ANGLE_DEG, 'deg', 'rad')) # m

rho_isa = ISA_PR / (R_SPEC * conv.convert(ISA_TEMP, 'celsius', 'kelvin'))

#------------------------------------------------------------------------------
#Load aircraft masses and TO manuf results
manuf_file = cl_path + '/TODR_MTOM_manuf' + f"/{aircraft_name}_{engine_name}.txt"

if os.path.exists(manuf_file):
    manuf_data = np.loadtxt(manuf_file, comments="#", delimiter=",")
    aircraft_mass = manuf_data[:, 0]
    to_manuf_value = manuf_data[:, 1]
    to_err = np.ones(len(to_manuf_value))
else :
    raise ValueError (f"Manufacturer data for aircraft: {aircraft_name} - {engine_name} not found in {cl_path + 'TODR_MTOM_manuf'}")

#dim check
if (len(aircraft_mass) != len(to_manuf_value)):
    raise ValueError (f"Dimension error: aircraft masses dim = {len(aircraft_mass)} != to manufacturer value = {len(to_manuf_value)}")

#***************************************************************************
#Find best C_l given the manufacturer data
cl_values = []
cl_rsmd = []

print('-------------------------------------------------')
cl_val, err_cl = cl_finder(aircraft_name, engine_name, aircraft_mass, to_manuf_value,
                             rho_isa, cd0, k, wing_area, airborne_dist, safe_margin_coeff=MARGIN_COEFF, mu = mu,
                               dv0=0.01, theta = 0.0, cl_min=1.0, cl_max=2.0, cl_step=0.01)

cl_values.append(cl_val)
cl_rsmd.append(err_cl)
print(np.min(cl_values))

#final results
cl_best = cl_val
err_cl_best = 0.01

print(f"C_l finding process results: C_l = {cl_best} +- {err_cl_best}")

#***************************************************************************
# Compute model predictions using best-fit CL
model_to_dist = np.array([calc_todr(m=i, rho=rho_isa, cl=cl_best, cd0=cd0, k=k, w_area=wing_area,
        airborne_d=airborne_dist, aircraft_name=aircraft_name, engine_name=engine_name,
        alt_ft=0.0, head_wind=0.0, margin_coeff=MARGIN_COEFF, mu=mu, theta=0.0, dv0=0.01,
        v_max=200.0,complete_data=False) for i in aircraft_mass
])

# Compute % difference between model and manufacturer values
perc_diff = (model_to_dist - to_manuf_value) / to_manuf_value * 100.0

# Uncertainty band (upper/lower bounds from CL uncertainty)
model_upper = np.array([
    calc_todr(
        m=m,
        rho=rho_isa,
        cl=cl_best - err_cl_best,
        cd0=cd0,
        k=k,
        w_area=wing_area,
        airborne_d=airborne_dist,
        aircraft_name=aircraft_name,
        engine_name=engine_name,
        alt_ft=0.0,
        head_wind=0.0,
        margin_coeff=MARGIN_COEFF,
        mu=mu,
        theta=0.0,
        dv0=0.01,
        v_max=200.0,
        complete_data=False
    ) for m in aircraft_mass
])

model_lower = np.array([
    calc_todr(
        m=m,
        rho=rho_isa,
        cl=cl_best + err_cl_best,
        cd0=cd0,
        k=k,
        w_area=wing_area,
        airborne_d=airborne_dist,
        aircraft_name=aircraft_name,
        engine_name=engine_name,
        alt_ft=0.0,
        head_wind=0.0,
        margin_coeff=MARGIN_COEFF,
        mu=mu,
        theta=0.0,
        dv0=0.01,
        v_max=200.0,
        complete_data=False
    ) for m in aircraft_mass
])

#Compute errors
model_err_upper = abs(model_upper - model_to_dist)
model_err_lower = abs(model_to_dist - model_lower)
'''
print('-------------------------------------------------')
print('Perc. difference between Manufacturer and model values:' )
print((perc_diff))
print(f'Mean abs perc. difference: {np.mean(abs(perc_diff)):.3f} %')
print('-------------------------------------------------')
'''

#***************************************************************************
# Create a DataFrame with all the relevant data
df = pd.DataFrame({
    "mass_kg": aircraft_mass,
    "mass_tonnes": aircraft_mass / 1000.,
    "todr_manufacturer": to_manuf_value,
    "todr_model": model_to_dist,
    "todr_model_err_upper": model_err_upper,
    "todr_model_err_lower": model_err_lower,
    "todr_manufacturer_err": to_err,
})

# Add global values as metadata
metadata_dict = {
    "cl_best": cl_best,
    "cl_err": err_cl_best,
    "rho_isa": rho_isa,
    "cd0": cd0,
    "k": k,
    "wing_area": wing_area,
    "airborne_dist": airborne_dist,
    "safe_margin_coef": MARGIN_COEFF,
    "mu": mu,
}

# Convert to Arrow Table (no index)
table = pa.Table.from_pandas(df, preserve_index=False)

# Add metadata (must be encoded as bytes)
encoded_meta = {str(k): str(v).encode("utf-8") for k, v in metadata_dict.items()}
existing_meta = table.schema.metadata or {}
merged_meta = {**existing_meta, **encoded_meta}
table = table.replace_schema_metadata(merged_meta)

# Save to parquet
parquet_path = os.path.join(cl_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")

#parquet_path = os.path.join(output_path, "cl_TODR_data_vel_break.parquet")

pq.write_table(table, parquet_path)
print(f"C_L DataFrame with metadata written to {parquet_path}")
