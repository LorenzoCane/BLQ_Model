"""
    Utility functions for environment setup and directory management.

    Functions:
    ----------
    - install_requirements(requirements_file): Installs Python dependencies from a requirements file.
    - dir_existance_checker(dir_path): Ensures the existence of a directory, creating it if necessary.
"""

import subprocess
import threading
import time
import sys
from tqdm import tqdm
import os

def install_requirements(requirements_file="requirements.txt"):
    """
        Install Python dependencies from a requirements file in a background thread.
        While installing, a progress bar is shown for user feedback.

        Parameters:
        ----------
        requirements_file : str
            Path to the requirements file. Defaults to "requirements.txt".

        Behavior:
        ---------
        - Runs pip install command with quiet output.
        - Handles errors gracefully (file not found, pip errors, unexpected issues).
        - Displays a simple progress bar using tqdm during installation.

        Returns:
        --------
        None
    """
    print(f'Installing requirements (see {requirements_file})')
    def pip_install():
        try:
            # Suppress output with -q (quiet)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            status[0] = "done"
        except subprocess.CalledProcessError:
            status[0] = "error"
        except FileNotFoundError:
            status[0] = "file not found"
        except Exception as e:
            status[0] = f"unexpected error: {e}"

    status = ["installing"]
    thread = threading.Thread(target=pip_install)
    thread.start()

    # Show progress bar while installation is ongoing
    with tqdm(total=1, bar_format="{l_bar}{bar}| {elapsed}", position=0) as pbar:
        while thread.is_alive():
            time.sleep(0.1)
            pbar.update(0)  # Keep bar active
        pbar.update(1)

    # Final message
    if status[0] == "done":
        print("Requirements installed successfully.")
    elif status[0] == "error":
        print(" An error occurred during installation.")
    elif status[0] == "file not found":
        print(f"Requirements file '{requirements_file}' not found.")
    else:
        print(f"{status[0]}")
