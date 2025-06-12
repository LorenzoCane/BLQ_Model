'''
DEEP BLUE SRL.
Lorenzo Cane - Energy & Environment Area Consultant

Last Modified : 12/06/2025
'''

#Import libraries and dependencies
import subprocess
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

#***************************************************************************
#Phases
CL_FINDER = True

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

#airborne dist
asc_m = conv.convert(ASC, 'ft', 'm') # m
airborne_dist = asc_m / np.tan(conv.convert(CLIMB_ANGLE_DEG, 'deg', 'rad')) # m

print(f'Configuration successfully loaded from {config_file}')

#dirs
os.makedirs(results_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
#***************************************************************************
# #Perform C_L evaluation if needed or request
cl_parquet_path = os.path.join(cl_path, f"cl_{aircraft_name}_{engine_name}_TODR_data.parquet")

if CL_FINDER or not os.path.exists(cl_parquet_path):
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
