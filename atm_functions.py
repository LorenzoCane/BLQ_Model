import numpy as np
import sys
from constants import *
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv

def satur_vapor_pres(temp_c):
    '''
     Calculate saturated vapour pressure of water (https://en.wikipedia.org/wiki/Vapour_pressure_of_water), using
     A. Buck equation (https://en.wikipedia.org/wiki/Arden_Buck_equation).
     
     Parameters:
     -----------
     temp_c : float
              Air temperature in degree Celsius

     Returns:
     --------
     float : saturated vapour pressure at the given temperature, in kPa
    '''
    #Select the coefficients based on temperature as suggested by the author
    coeff_list = (0.61121, 18.678, 234.5, 257.14) if temp_c >= 0 else (0.61115, 23.036, 333.7, 279.82)
   
    a = (coeff_list[1] - temp_c / coeff_list[2]) 
    b = temp_c / (coeff_list[3] + temp_c)
    svp = coeff_list[0] * np.exp(a * b) 

    return svp 

def compute_air_density(temp_c, press_pa, rh_perc):
    '''
     Compute air density at given level of temperature, pressure, andrelative humidity

     Parameters:
     -----------
     temp_c : float
              Air temperature in degree Celsius
     press_pa : float
              Air pressure in Pa
     rh_perc : float
              Relative humidity of air , as percentage

     Returns:
     --------
     float : air density [kg m^-3]
    ''' 

    temp_k = conv.convert(temp_c, 'celsius', 'kelvin') #K
    if any([rh_perc < 0.0, rh_perc > 100.0]):
        raise ValueError (f'Value {rh_perc} is out of range for relative humidity percentage')
    
    rh = rh_perc / 100.0 #relative humidity
    p_sat = satur_vapor_pres(temp_c) #kPa 
    
    p_v_kpa = rh * p_sat #kPa vapor pressure
    p_v_pa = p_v_kpa * 1000 #Pa
    p_d = press_pa - p_v_pa #Pa dry air pressure

    rho = (p_d / (R_SPEC * temp_k)) + (p_v_pa / (R_V * temp_k))

    return rho


