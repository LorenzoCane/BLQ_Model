'''
DEEP BLUE SRL.
Lorenzo Cane - Energy & Environment Area Consultant

Last Modified : 12/06/2025
'''

#Import libraries and dependencies


import sys
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv
from file_utils import install_requirements
install_requirements(requirements_file='requirements.txt')
print('---------------------------------------------------------------------------')
from constants import *

import numpy as np
import os
import yaml
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning
import subprocess
from multiprocessing import Pool, cpu_count
from itertools import product

from performanc_functions import take_off
from airport_utils import *
from aircraft_utils import *
from atm_functions import compute_air_density

#***************************************************************************
#Phases
CL_FINDER = False

#***************************************************************************
#import from configuration file config.yml & create dirs
config_file = 'config.yml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)


# Accessing different sections
cl_path = config['Dir']['cl_dir']
results_dir = config['Dir']['results_dir']
img_dir = config['Dir']['img_dir']
output_dir =  config['Dir']['output_dir']

aircraft_name = config['Aircraft']['aircr_name']
engine_name = config['Aircraft']['aircr_engine']

airport_code = config['Airport']['airport_code']

climate_model = config['Climate']['model']
climate_months = config['Climate']['months']

passenger_mass = config['Mass']['passenger_mass']

# Grid parameters
grid = config['Grid']
temp_arr = np.arange(grid['temp_min'], grid['temp_max'] + 0.1, grid['temp_step'])
pres_arr = np.arange(grid['pres_min'], grid['pres_max'] + 1, grid['pres_step'])
hum_arr  = np.arange(grid['hum_min'],  grid['hum_max'] + 0.1, grid['hum_step'])
wind_arr = np.arange(grid['wind_min'], grid['wind_max'] + 0.1, grid['wind_step'])

#airborne dist
asc_m = conv.convert(ASC, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(CLIMB_ANGLE_DEG, 'deg', 'rad')) # m

#import airport and aircraft data
aircraft_info = get_aircraft_data(aircraft_name, engine_name)

wing_area = aircraft_info["wing_area"]
cd0 = aircraft_info["cd0"]
k = aircraft_info["k"]
mu = aircraft_info["mu"]
mass_max = aircraft_info["mass_max"]

airport_l_m = airport_get_lenght(airport_code)
airport_elev =  airport_get_elev(airport_code)

print(f'Configuration successfully loaded from {config_file}')

#dirs
os.makedirs(results_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
#***************************************************************************
# #Perform C_L evaluation if needed or request
cl_parquet_path = os.path.join(cl_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")

if any([CL_FINDER, not os.path.exists(cl_parquet_path)]):
    print('==========================================================')
    print("Parquet file not found. Running CL evaluation script...")
    subprocess.run(["python", "cl_calc.py"], stdout=subprocess.DEVNULL)  
    print('==========================================================')
   
# Load the Parquet file with C_L value
table = pq.read_table(cl_parquet_path)
# Extract and decode metadata
metadata = table.schema.metadata
if metadata:
    decoded_meta = {k.decode(): v.decode() for k, v in metadata.items()}
    cl_best = float(decoded_meta["cl_best"])
    cl_err = float(decoded_meta["cl_err"])
    print(f"C_l best: {cl_best}")
    print(f"C_l error: {cl_err}")
else:
    cl_best = 1.00
    print(f"No metadata found in the file.\n Default value C_l ={cl_best} will be used.")


#***************************************************************************
# GRID CALCULATION

#Create grid
param_grid = list(product(temp_arr, pres_arr, hum_arr, wind_arr))

#Define worker function
def worker(params):
    temp, h_wind, press, hum = params
    #compute air density
    rho = compute_air_density(temp, press, hum) #kg m^-3
    #compute TODR
    todr = take_off(m=mass_max, cl=cl_best, rho=rho, cd0=cd0, k=k, w_area=wing_area,
        airborne_d=airborne_dist, aircraft_name=aircraft_name, engine_name=engine_name,
        alt_ft=airport_elev, head_wind=h_wind )

    return {
        "Temperature": temp,
        "Headwind": h_wind,
        "Pressure": press,
        "Humidity": hum,
        "AirDensity": rho,
        "TODR": todr
    }

#Parallel execution

if __name__ == '__main__':
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(worker, param_grid)

        #Save results as parquet
        df = pd.DataFrame(results)
        df.to_parquet(f"{aircraft_name}_{engine_name}_perf_grid.parquet", index=False)

        #Check 
        print(df.head())