# soft-hydraulic-ale-fsi

Arbitrary Lagrangian&ndash;Eulerian fluid&ndash;structure interaction (FSI) solvers for **soft hydraulics** problems &mdash; pressure-driven flow sin compliant microchannels where the fluid and elastic solid are two-way coupled.

Built on [FEniCSx / DOLFINx](https://github.com/fenics/dolfinx) and customized specifically for 2D problems, using quasi-direct coupling for unsteady problems and a monolithic approach for steady problems.

## Purpose

Simulate and analyze the deformation of a soft elastic wall driven by viscous (and inertial) flow, targeting elastoinertial regimes relevant to microfluidics and soft robotics. The code supports both steady and transient problems, Newtonian and shear-thinning (Carreau) fluids, and benchmarking against analytical solutions.

## Repository contents

| File / folder | Description |
| --- | --- |
| `ALE-FSIx_2D.ipynb` | **Main transient solver** — time-dependent ALE-FSI simulation in DOLFINx |
| `ALE-FSIx_2D_steady.ipynb` | **Steady solver** — monolithic steady ALE-FSI formulation |
| `build_gmsh_x.py` | Mesh generation helper: two-subdomain (fluid + solid) rectangle mesh via gmsh, returns tagged DOLFINx mesh |
| `strip_widgets.py` | Utility to strip notebook widget metadata before committing |
| `dolfin-2019/` | Legacy solvers based on the original FEniCS (DOLFIN 2019) |
| `theory_steady/` | Analytical theory notebooks for steady FSI (including shear-thinning models) |
| `theory_oscillatory/` | Analytical theory notebooks for oscillatory/streaming FSI in channels and tubes |
| `not-ready-for-github/` | Work-in-progress: adaptive time-stepping, FEniCSx conversion scripts, sample output |

## Key features

- Monolithic ALE-FSI discretization (fluid momentum + continuity + solid elasticity solved simultaneously)
- Pressure-inlet / pressure-outlet boundary conditions (Neumann)
- Carreau viscosity model for shear-thinning fluids
- 2D-restricted neo-Hookean solid with isochoric–volumetric splitting for strong compression
- Velocity-based damping in the solid for robust convergence to steady state
- Gmsh-based meshing with tagged subdomains and boundary facets
- Analytical steady-state benchmarks and Matplotlib post-processing

## Dependencies

- [FEniCSx / DOLFINx](https://github.com/fenics/dolfinx) (next-generation FEniCS)
- [Gmsh](https://gmsh.info) Python API
- PETSc / petsc4py, MPI / mpi4py
- NumPy, SciPy, Matplotlib

## Credits

Largely developed (ca. Fall 2025–Spring 2026) and maintained by [Ivan C. Christov](http://christov.tmnt-lab.org), Purdue University, with assistance from GitHub Copilot and Claude.

Initial unsteady code forked from an [earlier version](https://github.com/Radeu/Radeu-FSI-in-2D-Deformable-Channel-with-Oscillatory-Pressure-BC) based on David Kamensky's [fitted-fsi-example](https://github.com/david-kamensky/mae-207-fea-for-coupled-problems/tree/master/fsi).
