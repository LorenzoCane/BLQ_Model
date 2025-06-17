import sys 
sys.path.insert(0, './utils')
from unit_converter import ComplexUnitConverter as conv
from tools import rmsd
import numpy as np

from aircraft_utils import get_thrust
from constants import *
from openap.thrust import Thrust #thrust calc


def take_off(m, rho, cl, cd0, k, w_area, airborne_d, aircraft_name, engine_name, 
             alt_ft= 0.0, head_wind = 0.0, margin_coeff=1.15, 
             mu=0.017, theta=0., lift_frac=1.0, v_to=74.5, vel_break = False, return_velocity=False, dv0 = 0.01, dv_decay = 'const'):
    '''
     Simulates the ground roll phase of an aircraft's take-off run and estimates 
     the total take-off distance required (TODR), including the airborne distance.

     Parameters:
     -----------
     m : float
         Aircraft mass in kilograms.
     thrust : float
         Engine thrust in newtons.
     rho : float
         Air density in kg/m^3.
     cl : float
         Lift coefficient (assumed constant).
     cd0 : float
         Zero-lift drag coefficient.
     k : float
         Induced drag factor.
     w_area : float
         Wing area in m^2.
     airborne_d : float
         Airborne distance in meters (after lift-off).
     aircraft_name : str
         Name of the aircraft selected.
     engine_name : str 
         Specific engine used by the selected aircraft.
     alt_ft : float
         Altitude of the airport in ft ASL. 
     head_wind: float, optional 
         head winf velocity in m/s
     margin_coeff : float, optional
         Safety margin multiplier applied to the total distance. Default is 1.15.
     mu : float, optional
         Rolling friction coefficient. Default is 0.017.
     theta : float, optional
         Runway slope angle in degrees (positive for uphill). Default is 0°.
     lift_frac : float, optional
         Fraction of weight that must be supported by lift to initiate rotation. Default is 1.0.
     v_to : float, optional
         Target take-off velocity in m/s. Used only if `vel_break=True`. Default is 74.5 m/s.
     vel_break : bool, optional
         If True, terminates ground roll when velocity reaches `v_to`, regardless of lift. Default is False.
     return_velocity : bool, optional
         If True, returns a tuple (take-off distance, lift-off velocity). Default is False.
     dv0 : float, optional
        Base velocity increment (m/s) for simulation time steps. Default is 0.01 m/s.
     dv_decay : str, optional
         Strategy for reducing dv as velocity increases. Options: 
         'const' (constant dv), 'exp' (exponential decay), 
         'exp+' (stronger exponential), 'inv' (inverse). Default is 'const'.

     Returns:
     --------
     float or tuple
         If return_velocity is False: total take-off distance in meters.
         If return_velocity is True: (total take-off distance in meters, lift-off velocity in m/s).
    '''
    ground_speed = 0.0  # m/s
    vel = ground_speed + head_wind
    d = 0.0    # m

    thr = Thrust(ac= aircraft_name, eng= engine_name)
    thrust = thr.takeoff(tas = conv.convert(vel, 'ms', 'kts'), alt=alt_ft) #N conv
    
    theta = conv.convert(theta, 'deg', 'rad')
    cd = cd0 + k * cl* cl
    weight = m * 9.80665
   
    #print(f'weight = {weight} N')
    while True:
        # Current state
        D = 0.5 * rho * vel* vel * w_area * cd #parabolic "classic" drag
  
        L = 0.5 * rho * vel* vel * w_area * cl
        #print ('init:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        if L >= weight * lift_frac:
            break

        dv=dv0
        a_current = (thrust - D - mu * (weight * np.cos(theta) - L) - weight * np.sin(theta)) / m
        
        # Advance velocity
        ground_speed += dv
        vel = ground_speed + head_wind

        thrust = thr.takeoff(tas = conv.convert(vel, 'ms', 'kts'), alt=alt_ft) #N conv

        # Next state
        D = 0.5 * rho * (vel**2) * w_area * cd
        L = 0.5 * rho * (vel**2) * w_area * cl
        a_next = (thrust - D - mu * (weight * np.cos(theta) - L) - weight * np.sin(theta)) / m
        #print ('end:')
        #print(f'V = {vel} m/s')
        #print(f'L = {L}')
        #print(f'D = {D}')
        a_mean = 0.5 * (a_next + a_current)
        if a_mean <= 0.0:
            break  # Prevent division by zero or deceleration

        v_mean = ground_speed - (0.5 * dv)
        dx = v_mean * dv / a_mean
        d += dx

    final_distance = (d + airborne_d) * margin_coeff
    return final_distance


def calc_todr(m, rho, cl, cd0, k, w_area, airborne_d, aircraft_name, engine_name, 
             alt_ft= 0.0, head_wind = 0.0, margin_coeff=1.15, 
             mu=0.017, theta=0., dv0 = 0.01, v_max = 200.0, complete_data = False):
    
    theta = conv.convert(theta, 'deg', 'rad')
    cd = cd0 + k * cl* cl
    weight = m * G
    
    #speeds used in the calc
    ground_speeds_ms = np.arange(0.0, v_max, step=dv0) #m/s
    #TAS 
    tas_ms = np.array([x + head_wind for x in ground_speeds_ms]) #m/s
    #create a dict to store performance related values
    data_dict = {}

    #Populate dict with thrust, drag , weight , acceleration values
    data_dict["thr"] = get_thrust(aircraft_name=aircraft_name, engine_name=engine_name, speeds_ms=tas_ms, alt_ft=alt_ft)
    data_dict["D"] = 0.5 * rho * tas_ms**2 * w_area * cd
    data_dict["L"] = 0.5 * rho * tas_ms**2 * w_area * cl
    data_dict["W"] = weight
    data_dict["a"] = (data_dict["thr"] -  data_dict["D"] - mu * (weight * np.cos(theta) - data_dict['L']) - weight * np.sin(theta)) / m
    
    #store useful data
    data_dict['deltav'] = dv0
    data_dict["Vtas"] = tas_ms

    #Min TAS when L > W
    data_dict["index"] = (np.where(data_dict['L'] > data_dict['W'])[0][0])

    #Calc Delta x for todr integral calc
    
    deltax = []

    for i in np.arange(1, len(ground_speeds_ms[0 : (data_dict["index"]+3)])):
        deltax.append(
            (ground_speeds_ms[i] + ground_speeds_ms[i-1]) * dv0 / (data_dict['a'][i] + data_dict['a'][i-1]) # 2.0 division between means have been erased       
            )
    data_dict["deltax"] = deltax

    #Calc ground segment distance
    data_dict["x"] = np.cumsum(deltax)[data_dict['index']] #m

    #TODR as ground segment + airborne distance * security margin coefficient
    data_dict['TODR'] = ( data_dict['x'] + airborne_d ) * margin_coeff

    return data_dict if complete_data else data_dict['TODR']


def cl_finder(aircraft_name, engine_name, aircraft_mass, to_manuf_value, rho_isa, 
              cd_0, k_p, wing_area, airborne_dist, safe_margin_coeff,
              mu, dv0=0.01, theta=0., cl_min=1.0, cl_max=2.0, cl_step=0.01):
    '''
    Grid search to find the optimal lift coefficient C_L that minimizes the RMSD
    between model-based TODR (using calc_todr) and manufacturer values.

    Parameters
    ----------
    aircraft_name : str
        Name of the aircraft model.
    engine_name : str
        Name of the engine model.
    aircraft_mass : array-like
        Aircraft mass values in kg.
    to_manuf_value : array-like
        Manufacturer take-off distances in meters.
    rho_isa : float
        Air density in kg/m³.
    cd_0 : float
        Zero-lift drag coefficient.
    k_p : float
        Induced drag factor.
    wing_area : float
        Wing area in m².
    airborne_dist : float
        Airborne segment in meters.
    safe_margin_coeff : float
        Safety margin multiplier.
    mu : float
        Friction coefficient.
    dv0 : float, optional
        Speed resolution for TODR integration.
    theta : float, optional
        Runway slope (degrees).
    cl_min, cl_max : float
        Bounds of the C_L grid.
    cl_step : float
        Grid step.

    Returns
    -------
    best_cl_guess : float
        Best estimate for the lift coefficient.
    dummy_error : float
        Placeholder for uncertainty (currently 0.1).
    '''

    cl_candidates = np.arange(cl_min, cl_max + cl_step, cl_step)
    predicted_todr = []

    for cl_val in cl_candidates:
        todr_list = []
        for m, to_ref in zip(aircraft_mass, to_manuf_value):
            try:
                todr = calc_todr(
                    m=m,
                    rho=rho_isa,
                    cl=cl_val,
                    cd0=cd_0,
                    k=k_p,
                    w_area=wing_area,
                    airborne_d=airborne_dist,
                    aircraft_name=aircraft_name,
                    engine_name=engine_name,
                    alt_ft=0.0,
                    head_wind=0.0,
                    margin_coeff=safe_margin_coeff,
                    mu=mu,
                    theta=theta,
                    dv0=dv0,
                    v_max=200.0,
                    complete_data=False
                )
                todr_list.append(todr)
            except Exception as e:
                print(f"Error at cl={cl_val:.3f}, m={m:.0f}: {e}")
                todr_list.append(np.nan)

        predicted_todr.append(todr_list)

    # Compute RMSD for each CL candidate
    predicted_todr = np.array(predicted_todr)
    diffs = predicted_todr - np.array(to_manuf_value)
    rmsd_vals = np.sqrt(np.nanmean(diffs**2, axis=1))

    best_idx = np.nanargmin(rmsd_vals)
    best_cl_guess = cl_candidates[best_idx]

    print(f"Best C_L: {best_cl_guess:.3f} | RMSD: {rmsd_vals[best_idx]:.2f}")

    return best_cl_guess, 0.1



def mtom(runway_length, initial_mass, alt_ft, aircraft_name, engine_name, rho, cl, cd0, k, wing_area, airborne_dist, safety_coef, mu, path_angle):
    '''
     Estimates the Maximum Take-Off Mass (MTOM) such that the total take-off
     distance required (TODR) does not exceed the given runway length.
 
     The function uses a coarse-to-fine grid search strategy by progressively
     decreasing the aircraft mass until TODR fits within the runway, then refining
     the estimate with smaller mass steps.
 
     Parameters
     ----------
     runway_length : float
         Maximum available take-off distance (meters).
     initial_mass : float
         Starting guess for aircraft mass (kg).
     thrust : float
         Engine thrust (N).
     rho : float
         Air density (kg/m^3).
     cl : float
         Lift coefficient.
     cd0 : float
         Zero-lift drag coefficient.
     k : float
         Induced drag coefficient.
     wing_area : float
         Wing surface area (m²).
     airborne_dist : float
         Distance required after lift-off (m).
     safety_coef : float
         Safety margin applied to TODR (e.g., 1.15).
     mu : float
         Friction coefficient during ground roll.
     path_angle : float
         Runway slope angle (degrees).
 
     Returns
     -------
     mass : float
         Estimated MTOM (kg) that ensures TODR ≤ runway_length.
    '''

    mass = initial_mass
    iter = 0
    if take_off(mass, rho, cl, cd0, k, wing_area, airborne_dist,aircraft_name, engine_name, alt_ft=alt_ft,
                 margin_coeff=safety_coef, mu=mu, 
                theta=path_angle, return_velocity=False) < runway_length:
        return mass  # already feasible

    for step in [1e3, 1e2, 1e1, 1]:  # reduce mass by 1000, 100, 10, 1
        while True:
            todr = take_off(mass, rho, cl, cd0, k, wing_area, airborne_dist,aircraft_name, engine_name, alt_ft=alt_ft,
                 margin_coeff=safety_coef, mu=mu, theta=path_angle, return_velocity=False) 
            if todr < runway_length:
                mass += step  # step back up to refine
                break
            mass -= step  # keep reducing
            iter +=1
            #print(f'Iter #{iter+1}: MTOM = {mass}, TODR = {todr}, l_runway = {runway_length}')

    return mass


def mtom_binary(runway_length, initial_mass, alt_ft, aircraft_name, engine_name, rho, cl, cd0, k,
                wing_area, airborne_dist, safety_coef, mu, path_angle,
                min_mass=61000, tol=1.0, iter_max=1.0e3, verbose=False):
    '''
     Uses binary search to estimate the Maximum Take-Off Mass (MTOM)
     such that TODR ≤ runway_length.
 
     Parameters
     ----------
     runway_length : float
         Available take-off distance (meters).
     initial_mass : float
         Upper bound for MTOM search (kg).
     thrust : float
         Engine thrust (N).
     rho : float
         Air density (kg/m^3).
     cl : float
         Lift coefficient.
     cd0 : float
         Zero-lift drag coefficient.
     k : float
         Induced drag coefficient.
     wing_area : float
         Wing surface area (m²).
     airborne_dist : float
         Distance required after lift-off (m).
     safety_coef : float
         Safety margin applied to TODR.
     mu : float
         Ground friction coefficient.
     path_angle : float
         Slope of runway (degrees).
     min_mass : float, optional
         Lower bound for MTOM search (default is 61000 kg).
     tol : float, optional
         Convergence tolerance on mass (default is 1 kg).
     iter_max : int, optional
         Maximum number of iterations (default is 1e5).
     verbose : bool, optional
         If True, print debug info.
 
     Returns
     -------
     float
         Estimated MTOM (kg), conservative (lower-bound) estimate.
    '''
        
    low = min_mass
    high = initial_mass
    iteration = 0

    while (high - low > tol) and iteration < iter_max:
        mid = (low + high) / 2
        todr = take_off(mid, rho, cl, cd0, k, wing_area, airborne_dist,aircraft_name, engine_name, alt_ft=alt_ft,
                 margin_coeff=safety_coef, mu=mu, theta=path_angle, return_velocity=False) 

        if verbose:
            print(f"[Iter {iteration}] Mass: {mid:.2f} kg, TODR: {todr:.2f} m")

        if todr < runway_length:
            low = mid
        else:
            high = mid

        iteration += 1

    return low  # Conservative estimat
