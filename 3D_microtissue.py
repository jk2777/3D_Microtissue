'''
------------------------------------------------------------------------------
3D Microtissue Code (Developed by Jaemin Kim (jk2777@cornell.edu))
------------------------------------------------------------------------------
A model for 3D deformation and reconstruction of contractile microtissues
J Kim, E Mailand, I Ang, MS Sakar, N Bouklas
Soft Matter 2021
'''
# ------------------------------------------------------------------------------
# Import Modules
# ------------------------------------------------------------------------------
from __future__ import division
from fenics import *
from mshr import *
from ufl import cofac

import shutil
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Set Parameters for PETSc SNES solver
# ------------------------------------------------------------------------------

# set some dolfin specific parameters
parameters["form_compiler"]["representation"]="uflacs"
parameters["form_compiler"]["optimize"]=True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=4

# Define the solver parameters
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",
                                          'absolute_tolerance':1e-6,
                                          'relative_tolerance':1e-6,
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

# ------------------------------------------------------------------------------
# Set User-defined Parameters
# ------------------------------------------------------------------------------

# Define user parameters
user_par = Parameters("user")
user_par.add("K", 7.0)                 # Bulk modulus
user_par.add("G", 8.0)                 # Shear modulus
user_par.add("beta_bulk", 6.00)        # Normalized active bulk contraction
user_par.add("beta_surf", 0.06)        # Normalzied active surf tension
user_par.add("gamma_min",0.0)          # Ramping inital
user_par.add("gamma_max",1.0)          # Ramping final
user_par.add("gamma_nsteps",100)       # Ramping steps

# Add user parameters in the global parameter set
user_par.parse()
parameters.add(user_par)
user_par = parameters["user"]

# Parse from command line if given
K = user_par["K"]
G = user_par["G"]
beta_bulk = user_par["beta_bulk"]
beta_surf = user_par["beta_surf"]
gamma_min = user_par["gamma_min"]
gamma_max = user_par["gamma_max"]
gamma_nsteps = user_par["gamma_nsteps"]

# Material properties
E = 9*K*G/(3*K+G)                      # Elastic modulus
nu = (3*K-2*G)/(6*K+2*G)               # Poisson's ratio
beta_bulk = beta_bulk*E                # Active bulk contraction
beta_surf = beta_surf*E                # Active bulk contraction

# Ramping parameter
gamma = Expression("t",t=0.00,degree=1) # 't' is not actual time, but steps.

# File name
par1 = "K_%.1f" % (K)
par2 = "_G_%.1f" % (G)
par3 = "_nu_%.2f" % (nu)
par4 = "_beta_bulk_%.2f" % (beta_bulk)
par5 = "_beta_surf_%.2f" % (beta_surf)
par6 = "_gamma_nsteps_%.1d" % (gamma_nsteps)

# Directory for saving parameters
save_dir = par1+par2+par3+par4+par5+par6+"/"

# Save the paramters
File(save_dir + "/parameters.xml") << user_par

# ------------------------------------------------------------------------------
# Mesh (developed by Gmsh)
# ------------------------------------------------------------------------------
# Open mesh files
mesh = Mesh("3D_no_cut.xml")
subdomains = MeshFunction("size_t", mesh, "3D_no_cut_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "3D_no_cut_facet_region.xml")

# Defines the surface domain
file_results = XDMFFile(save_dir + "/" + "boundaries.xdmf")
file_results.write(boundaries)
ds1 = Measure("ds")(domain=mesh, subdomain_data=boundaries, subdomain_id=1)
ds2 = Measure("ds")(domain=mesh, subdomain_data=boundaries, subdomain_id=2)
ds3 = Measure("ds")(domain=mesh, subdomain_data=boundaries, subdomain_id=3)
ds4 = Measure("ds")(domain=mesh, subdomain_data=boundaries, subdomain_id=4)

# Defines the volume domain
dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

# Define finite element function spaces
V = VectorFunctionSpace(mesh, 'Lagrange', 1)
u = Function(V)

# Define trial and test function
du = TrialFunction(V)                 # Trial fucntion
v = TestFunction(V)                   # Test function

# ------------------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------------------
# Center point BC (constraining rigid body motion)
class pin_Point(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.00, DOLFIN_EPS) \
           and near(x[1], 0.00, DOLFIN_EPS) \
           and near(x[2], 0.00, DOLFIN_EPS)
pin_Point = pin_Point()
pin_Point_bc_z = DirichletBC(V.sub(2), Constant(0.0), pin_Point, method='pointwise')

# The Dirichlet BCs (Roller BCs around the four posts)
Side_Cylinder1_bc_x = DirichletBC(V.sub(0), Constant(0.0), boundaries, 1)
Side_Cylinder1_bc_y = DirichletBC(V.sub(1), Constant(0.0), boundaries, 1)
Side_Cylinder2_bc_x = DirichletBC(V.sub(0), Constant(0.0), boundaries, 2)
Side_Cylinder2_bc_y = DirichletBC(V.sub(1), Constant(0.0), boundaries, 2)
Side_Cylinder3_bc_x = DirichletBC(V.sub(0), Constant(0.0), boundaries, 3)
Side_Cylinder3_bc_y = DirichletBC(V.sub(1), Constant(0.0), boundaries, 3)
Side_Cylinder4_bc_x = DirichletBC(V.sub(0), Constant(0.0), boundaries, 4)
Side_Cylinder4_bc_y = DirichletBC(V.sub(1), Constant(0.0), boundaries, 4)

# Combined boundary conditions
bcs = [pin_Point_bc_z,
       Side_Cylinder1_bc_x, Side_Cylinder1_bc_y, \
       Side_Cylinder2_bc_x, Side_Cylinder2_bc_y, \
       Side_Cylinder3_bc_x, Side_Cylinder3_bc_y, \
       Side_Cylinder4_bc_x, Side_Cylinder4_bc_y]

# ------------------------------------------------------------------------------
# Kinematics
# ------------------------------------------------------------------------------
# Forces
B  = Constant((0.0, 0.0, 0.0))            # Body force per unit volume
T  = Constant((0.0, 0.0, 0.0))            # Traction force on the boundary

# Bulk Kinematics
d = u.geometric_dimension()               # Dimension
I = Identity(d)                           # Identity matrix (G1*G1+G2*G2+G3*G3)
F = I + grad(u)                           # Deformation gradient
C = F.T*F                                 # Right Cauchy-Green tensor
J = det(F)                                # Volume ratio
Ic = tr(C)                                # Invariant of bulk deformation
Ic_bar = J**(-2/3)*Ic                     # Modified invariant (isochoric)

# Surface kinematics
N_surf = FacetNormal(mesh)                 # Normal vector on surface in the reference configuration (rank1)
I_surf = I - outer(N_surf,N_surf)            # Surface identity matrix (G1*G1+G2*G2)
F_surf = dot(F,I_surf)                      # Surface deformation gradient
C_surf = F_surf.T*F_surf                     # Surface right Cauchy-Green tensor
J_surf = inner(dot(cofac(F),N_surf),dot(cofac(F),N_surf)) # Surface ratio (det(F_surf) is wrong!!)
Ic_surf = tr(C_surf)                        # Invariant of surface deformation

# ------------------------------------------------------------------------------
# Variational formulation
# ------------------------------------------------------------------------------
# Free energy density function in bulk
Psi = K/2*(J-1)**2 + G/2*(Ic_bar-3) + gamma*beta_bulk*J
bulk_energy = Psi*dx

# Free energy density function on surface
Psi_surf = gamma*beta_surf*J_surf # ds means surface integral
surface_energy = Psi_surf*ds

# Total potential energy
potential_energy = bulk_energy + surface_energy

# Setup the variational problem
F = derivative(potential_energy,u,v)
dF = derivative(F,u,du)
problem = NonlinearVariationalProblem(F, u, bcs, J=dF)

# ------------------------------------------------------------------------------
# Set up the solver (Newton solver)
# ------------------------------------------------------------------------------
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)
info(solver.parameters, False)
(iter, converged) = solver.solve()

# ------------------------------------------------------------------------------
# Solve the problem
# ------------------------------------------------------------------------------
# Directory for saving results
file_results = XDMFFile(save_dir + "/" +par1+par2+par3+par4+par5+par6+".xdmf")

# Steps
gamma_list = np.linspace(gamma_min, gamma_max, gamma_nsteps)

# Intialization
step = 0

# Time-Stepping
for t in gamma_list:
	# Update
    step += 1
    gamma.t = t

    # Monitoring the progress
    print("step:", step)
    print( gamma.t)

	# Solve the nonlinear problem (using Newton solver)
    solver.solve()

	# Solutions
    u.rename("Displacement", "u")

    # Parameters will share the same mesh
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.parameters["rewrite_function_mesh"] = False

    # Save solution to file (readable by Paraview or Visit)
    file_results.write(u,t)
