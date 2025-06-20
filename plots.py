"""
    plots.py

    This module loads take-off distance required (TODR) data from a grid simulation 
    and generates a heatmap of TODR values across two varying atmospheric parameters.

    It uses a precomputed TODR `.parquet` file containing results of TODR calculations

    The script enforces that two parameters are fixed (e.g., Temperature and Pressure) 
    and the other two are varied to produce a 2D heatmap.

    Dependencies project-specific utilities: unit_converter, file_utils, constants

    Author: Lorenzo Cane - DBL E&E Area Consultant
    Last modified: 20/06/2025
"""
#***************************************************************************
#Import libraries and dependencies
import sys
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv
from file_utils import install_requirements
install_requirements(requirements_file='requirements.txt')
print('---------------------------------------------------------------------------')
from constants import *

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning


#***************************************************************************
#Fixed values dict (use if ph quantinty is fixed)
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
#Dict of all the arrays
param_arrays = {"Temperature": temp_arr,
                "Pressure": pres_arr,
                "Humidity": hum_arr,
                "Headwind": wind_arr
            }
#fixed and varying params
fixed_params = grid['fixed_params']
varying_params = [key for key in allowed_keys if key not in fixed_params]

#Check that all params are allowed keys
if not all(key in allowed_keys for key in fixed_params):
    invalid_keys = [key for key in fixed_params if key not in allowed_keys]
    raise NameError(f"Invalid keys found in fixed_params: {invalid_keys}")
if not len(fixed_params) == 2:
    raise ValueError(f'fixed_params len must be 2, {len(fixed_params)} found.')

#***************************************************************************
#Laod TODR results
otuput_file = f"todr_grid_{aircraft_name}_{engine_name}_{varying_params[0].lower()}_{varying_params[1].lower()}.parquet"
df = pd.read_parquet(os.path.join(output_dir, otuput_file))

#***************************************************************************
#Pivot table
heatmap_data = df.pivot(index= varying_params[1], columns= varying_params[0], values='TODR')


# ----------------------------------------------------------
# Helper function to create axis labels
def create_labels(quantity_name):
    """Returns a label with unit for the given physical quantity."""

    unit_dict = {
        "Temperature" : " [˚C]",
        "Pressure" : " [Pa]",
        "Headwind" : " [m/s]",
        "Humidity" : " [%]"
    }

    label = quantity_name + unit_dict[quantity_name]

    return label


# ----------------------------------------------------------
# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=False, cmap='summer')
plt.title("TODR table")
plt.xlabel(create_labels(varying_params[0]))
plt.ylabel(create_labels(varying_params[1]))
plt.tight_layout()

# Save to PDF
plt.savefig(os.path.join(img_dir,f"todr_grid_{aircraft_name}_{engine_name}_{varying_params[0].lower()}_{varying_params[1].lower()}.pdf"))
plt.show()
plt.close()


'''
!!!
REMINDER: main style configuration are in custom_style.mplstyle file
!!!
'''