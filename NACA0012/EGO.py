

#%---------- imports ----------
from smt.applications import EGO
from smt.utils.design_space import DesignSpace
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
import subprocess
import numpy as np
import os
import importlib
import optAirfoil.aeroData

#%---------- Setup Design Variables ----------
shape = np.zeros(8)    #~ shape of the airfoil
patchV = np.zeros(2)   #~ angle of attack
patchV[0] = 10         #~ far field velocity (a fixed value)

#%---------- Setup EGO Problem ----------
criterion = "EI"                           #~ criterion for next evaluation point determination -> EGO algorithm
n_iter = 20                                #~ num iterations to optimize function
xlimits = np.array([[-0.05, 0.05]] * 9)    #~ DV bounds
xlimits[-1] = [0 , 5]                      #~ last DV is AOA which has different bounds, so here we adjust the bounds
xlimits[-2] = [-0.01 , 0.01]
xlimits[-3] = [-0.01 , 0.01]
design_space = DesignSpace(xlimits)        #~ Generate the design space given the domain on x
n_doe = 19                                 #~ number of sampling points
random_state = 45                          #~ seed value to reproduce results
path = os.getcwd()                         #~ path to cwd for cleaning and running dafoam

#%---------- Objective & Constraint Functions Definition ----------
def obj_val(x):

    length = int(x.shape[0])
    drag = np.zeros(length)

    for i in range(length):

        #~ set FFD points
        shape[:] = x[i , :-1]

        #~ set AOA
        patchV[-1] = x[i , -1]

        #~ write current results to file
        with open('optAirfoil/designVars.py' , 'w') as f:
            f.write(f"shape = {shape.tolist()}\n")
            f.write(f"patchV = {patchV.tolist()}\n")

        #~ run simulation
        subprocess.run([path + "/run", "arguments"], shell=True)

        #~ update CD and CL values
        importlib.reload(optAirfoil.aeroData)
        from optAirfoil.aeroData import CD, CL

        #~ get drag
        drag[i] = CD[0]

        #~ add penalty for CL constraint (CL = 0.5)
        #~ here we use a quadratic penalty method: penalty_weight * constraint_violation^2
        constraint_violation = abs(CL[0] - 0.5)
        penalty_weight = 10.0 
        drag[i] += penalty_weight * constraint_violation**2

        #~ clean case before running
        subprocess.run([path + "/clean", "arguments"], shell=True)

    return drag.reshape((-1 , 1))

#%---------- Run EGO Algorithm ----------
#~ pass parameters to EGO algorithm
ego = EGO(
    n_iter=n_iter,
    criterion=criterion,
    n_doe=n_doe,
    surrogate=KRG(design_space=design_space, print_global=False),
    random_state=random_state,
)

#~ run EGO algorithm
x_opt, y_opt, _, _, _ = ego.optimize(fun=obj_val)

#%---------- Return Results To The User ----------
#~ run primal on opt config for post processing
shape = x_opt[:-1]
patchV[-1] = x_opt[-1]
with open('optAirfoil/designVars.py' , 'w') as f:
    f.write(f"shape = {shape.tolist()}\n")
    f.write(f"patchV = {patchV.tolist()}\n")

subprocess.run([path + "/clean", "arguments"], shell=True)
subprocess.run([path + "/run", "arguments"], shell=True)
