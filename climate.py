'''
'''

#***************************************************************************
#Import libraries and dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv
from tools import create_labels
import yaml
import warnings
warnings.filterwarnings('ignore') #to exclude sns warning
from functools import reduce
from matplotlib import cm 

from constants import *

#***************************************************************************
#Read config file
config_file = 'config.yml'

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

#----------------------------------------------------------------------------
#Input dir and files
input_dir = config['Dir']['climate_dir']

wind_file = 'wr_CERRA.csv'
tmax_file = 'tmax_CERRA.csv'
tmean_file = 'tmean_CERRA.csv'

wind_path = os.path.join(input_dir, wind_file)
tmax_path = os.path.join(input_dir, tmax_file)
tmean_path = os.path.join(input_dir, tmean_file)

path_array = [wind_path, tmax_path, tmean_path]

#Check files
for path in path_array:
    if not os.path.exists(path):
        raise ValueError (f'File {path} not found')

#Output dirs
results_dir = config['Dir']['output_dir']
img_dir = config['Dir']['img_dir']

os.makedirs(results_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)
#----------------------------------------------------------------------------
# Grid parameters
grid = config['Grid']

#Validation of ranges
assert grid['temp_min'] < grid['temp_max'], "Temperature range invalid!"
#assert grid['pres_min'] < grid['pres_max'], "Pressure range invalid!"
#assert grid['hum_min'] < grid['hum_max'], "Relative humidity range invalid!"
assert grid['wind_min'] < grid['wind_max'], "Headwind range invalid!"

temp_arr = np.arange(grid['temp_min'], grid['temp_max'] + 0.1, grid['temp_step'])
#pres_arr = np.arange(grid['pres_min'], grid['pres_max'] + 1, grid['pres_step'])
#hum_arr  = np.arange(grid['hum_min'],  grid['hum_max'] + 0.1, grid['hum_step'])
wind_arr = np.arange(grid['wind_min'], grid['wind_max'] + 0.1, grid['wind_step'])

#***************************************************************************
#Import data
wind_df =  pd.read_csv(wind_path)
tmax_df = pd.read_csv(tmax_path)
tmean_df = pd.read_csv(tmean_path)

#Dataframes array
df_array = [wind_df, tmax_df, tmean_df]

print('Climate data imported')
print(sep)

'''
for df in df_array:
    print(f'{df} Check')
    df.head()
print(sep)
'''
#***************************************************************************
#Create and save a merged dataset
for i in range(len(df_array)):
    df_array[i]['time'] = pd.to_datetime(df_array[i]['time'])

# Merge all datasets on 'time'
merged = reduce(lambda left, right: pd.merge(left, right, on='time', how='outer'), df_array)
#Sort by time
merged = merged.sort_values('time').reset_index(drop=True)
#Clean for empty values
merged = merged.dropna()

#Save to csv
merged_csv = 'historical_merged.csv'
merged_path = os.path.join(input_dir, merged_csv)
merged.to_csv(merged_path)

print('Datasets merged:')
print(merged.head(5))
print(f'Merged dataset saved as {merged_path}')
print(sep)
#***************************************************************************
#Data analysis and plot

#Max and min values
txt_report = 'max_min_climate.txt'
output_txt = os.path.join(results_dir, txt_report)

with open(output_txt, 'w') as f:
    for df in df_array:
        var = df.columns.values[1] #variable name
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        #print(df.dtypes)
        #print(df.head())
        print(f'Summary table for {var} data:')

        #Build summary
        summary = pd.DataFrame({
            'min'   : df.min(),         #min value
            'max'   : df.max(),         #max value 
            'first' : df.index.min(), #first date
            'last'  : df.index.max(), #last date
        })

        print(summary)
        print(sep)

        f.write(f"Summary table for {var} data:\n")
        f.write(summary.to_string())  
        f.write("\n" + "-"*40 + "\n\n")  # Separator between summaries

print(f'Summaries written in {output_txt}')
print(sep)
    
#---------------------------------------------------------------------------- 
#Plot and 1D hist
for i, df in enumerate(df_array):
    var = df.columns.values[1]  # variable name
    print(var)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    values = df[var]
    
    #Timeseries
    img_name = f"time_series_{var}.pdf"
    img_path = os.path.join(img_dir, img_name)

    plt.figure(figsize=(10, 4))
    plt.plot(values.index, values, label=var, color='blue')
    plt.title(f"Time Series of {var}")
    plt.xlabel("Time")
    plt.ylabel(create_labels(var))
    #plt.grid(True)
    plt.tight_layout()
    plt.savefig(img_path) 
    plt.close()
    print(f'Timeseries for variable {var} saved as {img_path}')
    
    # histogram
    bins_arr = temp_arr
    if var == "wds": #impose differnt bin edges only for wind data
        bins_arr = wind_arr
    img_name = f"1Dhist_{var}.pdf"
    img_path = os.path.join(img_dir, img_name)
    
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins_arr, color='orange', edgecolor='black')
    plt.title(f"Histogram of {var}")
    plt.xlabel(create_labels(var))
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(img_path)  # Save to file
    plt.close()
    print(f'1D histogram for variable {var} saved as {img_path}')

    print(sep)

#---------------------------------------------------------------------------- 
#2D Hists

#t_max vs wind
img_name = f"2Dhist_tmax_wind.pdf"
img_path = os.path.join(img_dir, img_name)

plt.figure(figsize=(6,5))
plt.hist2d(merged['mx2t24'], merged['wds'], bins=[temp_arr, wind_arr], cmap='viridis')
plt.colorbar(label='Counts')
plt.xlabel('t_max [˚C]')
plt.ylabel('Wind [m/s]')
plt.title('2D Histogram: t_max vs Wind')
plt.tight_layout()
plt.savefig(img_path)
plt.close()

#t_mean vs wind
img_name = f"2Dhist_tmean_wind.pdf"
img_path = os.path.join(img_dir, img_name)
plt.figure(figsize=(6,5))
plt.hist2d(merged['t2m'], merged['wds'], bins=[temp_arr, wind_arr], cmap='viridis')
plt.colorbar(label='Counts')
plt.xlabel('t_mean [˚C]')
plt.ylabel('Wind [m/s]')
plt.title('2D Histogram: t_mean vs Wind')
plt.tight_layout()
plt.savefig(img_path)
plt.close()

print(f'2D histograms saved in {img_dir}')
print(sep)

#---------------------------------------------------------------------------- 
#3D bar graphs

#t_max vs wind
img_name = f"3D_bar_plot_tmax_wind.pdf"
img_path = os.path.join(img_dir, img_name)

hist, xedges, yedges = np.histogram2d(merged['mx2t24'], merged['wds'], bins=[temp_arr, wind_arr])

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij") #meshgrid
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = (xedges[1] - xedges[0]) * 0.75  # width of bars
dz = hist.ravel()  # height of bars

# Colormap based on height
max_height = np.max(dz)
colors = cm.get_cmap('viridis')(dz / max_height)  # normalize dz for colormap

fig = plt.figure(figsize=(8,6))
    
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.75, shade=True)
ax.set_xlabel('t_max [˚C]')
ax.set_ylabel('Wind [m/s]')
ax.set_zlabel('Counts')

plt.tight_layout()
plt.savefig(img_path)
plt.close()

#t_mean vs wind
img_name = f"3D_bar_plot_tmean_wind.pdf"
img_path = os.path.join(img_dir, img_name)

hist, xedges, yedges = np.histogram2d(merged['t2m'], merged['wds'], bins=[temp_arr, wind_arr])

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij") #meshgrid
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = (xedges[1] - xedges[0]) * 0.75  # width of bars
dz = hist.ravel()  # height of bars

# Colormap based on height
max_height = np.max(dz)
colors = cm.get_cmap('viridis')(dz / max_height)  # normalize dz for colormap

fig = plt.figure(figsize=(8,6))
    
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.75, shade=True)
ax.set_xlabel('t_mean [˚C]')
ax.set_ylabel('Wind [m/s]')
ax.set_zlabel('Counts')

plt.tight_layout()
plt.savefig(img_path)
plt.close()

print(f'3D bar plots saved in {img_dir}')