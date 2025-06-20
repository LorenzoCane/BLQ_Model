"""
    Configuration loader

    This module provides function to load and deal with the configuaration file.

    Author: Lorenzo Cane - DBL E&E Area Consultant
    Last modified: 20/06/2025
"""
import yaml

def load_config(path='config.yml'):
    """
        Load configuration from a YAML file.

        Parameters:
        ----------
        path : str, optional
             Path to the YAML configuration file (default is 'config.yml').

        Returns:
        -------
        dict: Configuration data parsed from the YAML file.
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)
