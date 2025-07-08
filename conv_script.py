'''
        Conversion script

'''
import os
import numpy as np
import sys
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv


input_dir = './raw'
output_dir = './input/cl_data/TODR_MTOM_manuf'

input_file = 'B777_pound_ft.txt'
output_file = 'B77W_GE90-115B.txt'

input_path =  os.path.join(input_dir, input_file)
output_path = os.path.join(output_dir, output_file)

data_raw = np.loadtxt(input_path, delimiter=',')

mass = data_raw[:,0]
todr = data_raw[:,1]

mass_kg = np.round(conv.convert(mass, 'lb', 'kg'), 0)
todr_m = np.round(conv.convert(todr, 'ft', 'm'), 0)

with open(output_path, 'w') as f:
    #header    
    f.write('#mass_kg, todr_m\n')
    #rows
    for m,t in zip(mass_kg, todr_m):
        line = str(m) + ', ' + str(t) +'\n'
        f.write(line)   
