import sys 
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from performance_functions import calc_todr, take_off
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


test = calc_todr(78000, 1.22, 1.43, cd0=cd0, k=k, w_area=wing_area, airborne_d=airborne_dist, aircraft_name=aircraft_name,
                 engine_name=engine_name, alt_ft=airport_elev, head_wind=0.0)

test2 =  take_off(78000, 1.22, 1.43, cd0=cd0, k=k, w_area=wing_area, airborne_d=airborne_dist, aircraft_name=aircraft_name,
                 engine_name=engine_name, alt_ft=airport_elev, head_wind=0.0)
print(test)
print(test2)