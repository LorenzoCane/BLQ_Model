import sys 
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from atm_functions import satur_vapor_pres, compute_air_density

buch_dir = os.path.join(current, 'buch_eval')
os.makedirs(buch_dir, exist_ok=True)
out_file = os.path.join(buch_dir, 'buch_lide_confr.csv')
out_img =  os.path.join(buch_dir, 'buch_lide_confr.pdf')

temp_c = [0, 20, 35, 50 , 75, 100] # ˚C
true_P = np.array([0.6113, 2.3388, 5.6267, 12.344, 38.563, 101.32]) #Lide Table

#evaluate svp using Buch formula
Buch_eval = np.array([satur_vapor_pres(t) for t in temp_c]) #kPa

perc_diff = (Buch_eval - true_P) / true_P * 100
#Save results as .csv file
df = pd.DataFrame({
    "temp_c" : temp_c,
    "lide_P" : true_P,
    "buch_eval" : Buch_eval,
    "perc_diff" : perc_diff
})

df.set_index("temp_c", inplace=True)
df.to_csv(out_file)

#Plots results
plt.figure()

#plt.scatter(temp_c, true_P, color = 'orange', label = 'Lide results')
#plt.scatter(temp_c, Buch_eval, color = 'blue', label = 'Buch eval.')
plt.scatter(temp_c, perc_diff, label = 'Difference between Lide and Buch')

plt.xlabel('Temperature [˚C]')
#plt.ylabel('SVP [kPa]')
plt.ylabel('Perc. Diff [%]')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(out_img)
#plt.show()
plt.close()

#--------------------------------------------------------------------------
#test air density calulator

test_rho = compute_air_density(100, 101325, 50)
print(test_rho)