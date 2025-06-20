"""
    Main script for aircraft performance evaluation at a specific airport (BLQ).

    This script evaluates the Take-Off Distance Required (TODR) over a grid of atmospheric
    parameters (temperature, pressure, humidity, headwind) using a parallelized computation.
    Results are saved in a Parquet file for post-processing.

    Workflow:
    ---------
    1. Load configuration from YAML file.
    2. Import and check aircraft and airport parameters.
    3. Load (or compute) optimal lift coefficient (C_L).
    4. Create a grid of parameter combinations (2 varying, 2 fixed).
    5. Evaluate TODR and air density over the grid using multiprocessing.
    6. Save the output as a Parquet file. (opt plots)

    Author: Lorenzo Cane - DBL E&E Area Consultant
    Last Modified: 20/06/2025
"""


#Import libraries and dependencies
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv
from file_utils import install_requirements
install_requirements(requirements_file='requirements.txt')
print('---------------------------------------------------------------------------')
from constants import *

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
from tqdm import tqdm
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning
import subprocess
from multiprocessing import Pool, cpu_count
from itertools import product

from performance_functions import take_off, calc_todr
from airport_utils import *
from aircraft_utils import *
from atm_functions import compute_air_density

#***************************************************************************
#Phases
CL_FINDER = False #C_L finder sub process can be impose
PLOTS = False     #Make plot or not in main (plot subprecess)

#***************************************************************************
#Fixed values dict (used if the parameter is fixed)
#Fixed parameters are selected in config file, fixed value are taken from constant.py file
fixed_values = {
    "Temperature" : ISA_TEMP,
    "Headwind" : FIX_HW,
    "Humidity" : FIX_HUM, 
    "Pressure" : ISA_PR
}
allowed_keys =  fixed_values.keys()

#***************************************************************************
# Plotting, general appereance
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 110
plt.style.use('custom_style.mplstyle')
cyclec = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
cyclec = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
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

#Validation of ranges
assert grid['temp_min'] < grid['temp_max'], "Temperature range invalid!"
assert grid['pres_min'] < grid['pres_max'], "Pressure range invalid!"
assert grid['hum_min'] < grid['hum_max'], "Relative humidity range invalid!"
assert grid['wind_min'] < grid['wind_max'], "Headwind range invalid!"

temp_arr = np.arange(grid['temp_min'], grid['temp_max'] + 0.1, grid['temp_step'])
pres_arr = np.arange(grid['pres_min'], grid['pres_max'] + 1, grid['pres_step'])
hum_arr  = np.arange(grid['hum_min'],  grid['hum_max'] + 0.1, grid['hum_step'])
wind_arr = np.arange(grid['wind_min'], grid['wind_max'] + 0.1, grid['wind_step'])
#Dict of all parameters arrays
param_arrays = {"Temperature": temp_arr,
                "Pressure": pres_arr,
                "Humidity": hum_arr,
                "Headwind": wind_arr
            }
#fixed and varying params
fixed_params = grid['fixed_params']
varying_params = [key for key in allowed_keys if key not in fixed_params]

#Check that all params are allowed
if not all(key in allowed_keys for key in fixed_params):
    invalid_keys = [key for key in fixed_params if key not in allowed_keys]
    raise NameError(f"Invalid keys found in fixed_params: {invalid_keys}")
if not len(fixed_params) == 2:
    raise ValueError(f'fixed_params len must be 2, {len(fixed_params)} found.')

temp_dim = len(temp_arr)
press_dim = len(pres_arr)
hum_dim = len(hum_arr)
wind_dim = len(wind_arr)

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

#Ensure dirs existance
os.makedirs(results_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
#***************************************************************************
#1) C_L evaluation if needed or request
cl_parquet_path = os.path.join(cl_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")

if any([CL_FINDER, not os.path.exists(cl_parquet_path)]):
    print('==========================================================')
    print("Parquet file not found. Running C_L evaluation script...")
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
#2) PARAMETER GRID CALCULATION

#Create grid with the varying params (avoiding nested for loops)
param_grid = list(product(param_arrays[varying_params[0]], param_arrays[varying_params[1]]))
len_grid = len(param_grid)

selected_fix_val = {key: fixed_values[key] for key in fixed_params }

print(sep)
print("Parameters grid successfully created.")
print('================================================================')
print(f'Varying params: {varying_params[0]} and {varying_params[1]}')
print(f'Fixed params: {selected_fix_val}')
print(f'Dimension of the grid : {len_grid}')
print('================================================================')


#Define worker function for parallel computation
def worker(params):
    '''
        Compute air density and aircraft performance give a combination of two varying parameters,

        Parameters:
        ----------
        params : list, float
                Value of the varying parameters
        
        Return:
        ---------
        dict : Dictionary of input parameters, air density, and computed TODR.
    '''
    p1_val, p2_val = params
    param_dict = {
        varying_params[0]: p1_val,
        varying_params[1]: p2_val,
        **selected_fix_val
    }

    # Compute air density (Buch  moist air equation)
    rho = compute_air_density(
        temp_c=param_dict["Temperature"],
        press_pa=param_dict["Pressure"],
        rh_perc=param_dict["Humidity"]
    )

    try:
        # Compute TODR
        todr = calc_todr(m=mass_max, rho=rho, cl=cl_best, cd0=cd0, k=k, w_area=wing_area,
                     airborne_d=airborne_dist, aircraft_name=aircraft_name, engine_name=engine_name,
                     alt_ft=airport_elev, head_wind=param_dict["Headwind"])
    
    except Exception as e:
        print(f"[Warning] Failed at {param_dict} → {e}")
        todr = float("nan")  # or np.nan
    
    return {
        varying_params[0]: p1_val,
        varying_params[1]: p2_val,
        **selected_fix_val,
        "AirDensity": rho,
        "TODR": todr
    }

#***************************************************************************
#3) Parallel execution

if __name__ == '__main__':
    # Use multiprocessing to evaluate performance grid in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(worker, param_grid),
                            total=len(param_grid),
                            desc=f"Computing Grid: {varying_params[0]} vs {varying_params[1]}",
                            leave=True
                            )
                        )

    #collect all the partial results in a DateFrame and save it as a parquet file
    df = pd.DataFrame(results)
    output_name = f"todr_grid_{aircraft_name}_{engine_name}_{varying_params[0].lower()}_{varying_params[1].lower()}.parquet"
    output_path = os.path.join(output_dir, output_name)
    df.to_parquet(output_path, index=False) #no index
    
    #Print dataFrame lines for quick check
    print(sep)
    print(f'Computation ends. Performance results saved in {output_path}')
    print('Quick check:')
    print(df.head())

