"""
    Airport Utilities Module

    This module provides functions to retrieve geographic and physical information
    about airports using their ICAO codes. It relies on the `airportsdata` package
    for static airport metadata (latitude, longitude, elevation, and name), and 
    a custom CSV file for runway lengths.

    Expected:
    - Configuration file accessible via `load_config()` that specifies input directory.
    - A CSV file named 'airport_runways.csv' with columns ['ICAO', 'Runway_Length_m'].

    Functions:
    - airport_get_lat:     Return airport latitude.
    - airport_get_log:     Return airport longitude.
    - airport_get_elev:    Return airport elevation.
    - airport_get_name:    Return airport name.
    - airport_get_lenght:  Return airport runway length (in meters).

    Author: Lorenzo Cane - DBL E&E Area Consultant
    Last modified: 20/06/2025

"""

import airportsdata #airport data
import os
import pandas as pd
from config_loader import load_config
from constants import *

config = load_config()
input_dir = config['Dir']['input_dir']
airports = airportsdata.load()

def airport_get_lat(airport_code):
    """
        Get the latitude of an airport given its ICAO code.

        Parameters:
        ----------
        airport_code: str
                    The ICAO airport code (e.g., "LIMC").

        Returns:
        ---------
        float: Latitude of the airport in decimal degrees.
    """
    sel_airport = airports[airport_code]
    return sel_airport['lat']

def airport_get_log(airport_code):
    """
        Get the longitude of an airport given its ICAO code.

        Parameters:
        ----------
        airport_code : str
            The ICAO airport code (e.g., "LIMC").

        Returns:
        -------
        float: Longitude of the airport in decimal degrees.
    """
    sel_airport = airports[airport_code]
    return sel_airport['lon']

def airport_get_elev(airport_code):
    """
        Get the elevation of an airport given its ICAO code.

        Parameters:
        ----------
        airport_code : str
            The ICAO airport code (e.g., "LIMC").

        Returns:
        -------
        float: Elevation of the airport in feet above sea level.
    """
    sel_airport = airports[airport_code]
    return sel_airport['elevation']

def airport_get_name(airport_code):
    """
        Get the full name of an airport given its ICAO code.

        Parameters:
        ----------
        airport_code : str
            The ICAO airport code (e.g., "LIMC").

        Returns:
        -------
        str: Full name of the airport.
    """
    sel_airport = airports[airport_code]
    return sel_airport['name']

def airport_get_lenght(airport_code):
    """
        Get the runway length for an airport from a CSV file.

        Parameters:
        ----------
        airport_code : str
                    The ICAO airport code (e.g., "LIMC").

        Returns:
        -------
        float: Runway length in meters.

        Raises:
        ------
        ValueError: If the airport_code is not found in the CSV or the length is invalid (negative).
        FileNotFoundError: If the CSV file does not exist at the configured input directory.
    """
    csv_path = os.path.join(input_dir, "airport_runways.csv")
    df_airport = pd.read_csv(csv_path)
    try:
        airport_length_m = df_airport.loc[
            df_airport['ICAO'] == airport_code,
            'Runway_Length_m'
        ].values[0]
    except IndexError:
        raise ValueError(
            f"ICAO code '{airport_code}' not found in '{csv_path}'."
        )

    if airport_length_m < 0:
        raise ValueError(
            f"Runway length for ICAO code '{airport_code}' is invalid: {airport_length_m}."
        )
    return airport_length_m