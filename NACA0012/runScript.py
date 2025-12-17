#!/usr/bin/env python
"""
DAFoam run script for the NACA0012 airfoil at low-speed for SBO
"""

# =============================================================================
# Imports
# =============================================================================
import os
import argparse
import numpy as np
import json
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder, OptFuncs
from mphys.scenario_aerodynamic import ScenarioAerodynamic
from pygeo.mphys import OM_DVGEOCOMP
from optAirfoil.designVars import shape as new_shape
from optAirfoil.designVars import patchV as new_patchV

# =============================================================================
# Input Parameters
# =============================================================================
# initial values
U0 = 10.0
p0 = 0.0
nuTilda0 = 4.5e-5
CL_target = 0.5
aoa0 = 5.13918623195176
A0 = 0.1
rho0 = 1.0

# Input parameters for DAFoam
daOptions = {
    "designSurfaces"      : ["wing"],
    "solverName"          : "DASimpleFoam",
    "primalMinResTol"     : 1.0e-12,
    "primalMinResTolDiff" : 1.0e12,

    "primalBC": {
        "U0": {"variable": "U", "patches": ["inout"], "value": [U0, 0.0, 0.0]},
        "p0": {"variable": "p", "patches": ["inout"], "value": [p0]},
        "nuTilda0": {"variable": "nuTilda", "patches": ["inout"], "value": [nuTilda0]},
        "useWallFunction": True,
    },

    "function": {
        "CD": {
            "type"                   : "force",
            "source"                 : "patchToFace",
            "patches"                : ["wing"],
            "directionMode"          : "parallelToFlow",
            "patchVelocityInputName" : "patchV",
            "scale"                  : 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
        "CL": {
            "type"                   : "force",
            "source"                 : "patchToFace",
            "patches"                : ["wing"],
            "directionMode"          : "normalToFlow",
            "patchVelocityInputName" : "patchV",
            "scale"                  : 1.0 / (0.5 * U0 * U0 * A0 * rho0),
        },
    },

    "normalizeStates": {
        "U"       : U0,
        "p"       : U0 * U0 / 2.0,
        "nuTilda" : nuTilda0 * 10.0,
        "phi"     :  1.0,
    },

    "inputInfo": {
        "aero_vol_coords" : {"type": "volCoord", "components": ["solver", "function"]},

        "patchV": {
            "type"       : "patchVelocity",
            "patches"    : ["inout"],
            "flowAxis"   : "x",
            "normalAxis" : "y",
            "components" : ["solver", "function"],
        },
    },
}

# Mesh deformation setup
meshOptions = {
    "gridFile"       : os.getcwd(),
    "fileType"       : "OpenFOAM",
    "symmetryPlanes" : [[[0.0, 0.0, 0.0] , [0.0, 0.0, 1.0]] , [[0.0, 0.0, 0.1] , [0.0, 0.0, 1.0]]],
}


# Top class to setup the optimization problem
class Top(Multipoint):
    def setup(self):

        # create the builder to initialize the DASolvers
        dafoam_builder = DAFoamBuilder(daOptions, meshOptions, scenario="aerodynamic")
        dafoam_builder.initialize(self.comm)

        # add the design variable component to keep the top level design variables
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # add the mesh component
        self.add_subsystem("mesh", dafoam_builder.get_mesh_coordinate_subsystem())

        # add the geometry component (FFD)
        self.add_subsystem("geometry", OM_DVGEOCOMP(file="FFD/wingFFD.xyz", type="ffd"))

        # add a scenario (flow condition) for optimization, we pass the builder
        # to the scenario to actually run the flow and adjoint
        self.mphys_add_scenario("scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder))

        # need to manually connect the x_aero0 between the mesh and geometry components
        # here x_aero0 means the surface coordinates of structurally undeformed mesh
        self.connect("mesh.x_aero0", "geometry.x_aero_in")
        # need to manually connect the x_aero0 between the geometry component and the scenario1
        # scenario group
        self.connect("geometry.x_aero0", "scenario1.x_aero")

    def configure(self):

        # get the surface coordinates from the mesh component
        points = self.mesh.mphys_get_surface_mesh()

        # add pointset to the geometry component
        self.geometry.nom_add_discipline_coords("aero", points)

        # use the shape function to define shape variables for 2D airfoil
        pts = self.geometry.DVGeo.getLocalIndex(0)
        dir_y = np.array([0.0, 1.0, 0.0])
        shapes = []
        for i in range(1, pts.shape[0] - 1):
            for j in range(pts.shape[1]):
                # k=0 and k=1 move together to ensure symmetry
                shapes.append({pts[i, j, 0]: dir_y, pts[i, j, 1]: dir_y})
        # LE/TE shape, the j=0 and j=1 move in opposite directions so that
        # the LE/TE are fixed
        for i in [0, pts.shape[0] - 1]:
            shapes.append({pts[i, 0, 0]: dir_y, pts[i, 0, 1]: dir_y, pts[i, 1, 0]: -dir_y, pts[i, 1, 1]: -dir_y})
        self.geometry.nom_addShapeFunctionDV(dvName="shape", shapes=shapes)

        # add the design variables to the dvs component's output
        self.dvs.add_output("shape", val=np.array(new_shape))
        self.dvs.add_output("patchV", val=np.array(new_patchV))

        # manually connect the dvs output to the geometry and scenario1
        self.connect("patchV", "scenario1.patchV")
        self.connect("shape", "geometry.shape")

# OpenMDAO setup
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")

# run simulation
prob.run_model()

# write current results to file
with open('optAirfoil/aeroData.py' , 'w') as f:
    f.write(f"CD = {prob.get_val('scenario1.aero_post.CD')}\n")
    f.write(f"CL = {prob.get_val('scenario1.aero_post.CL')}\n")
