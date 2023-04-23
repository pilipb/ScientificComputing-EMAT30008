# ScientificComputing-EMAT30008
Scientific Computing Module
Phil Blecher - xq19351
Jan - May 2023

This module looks at implementing advanced numerical methods for the solution of real-world problems and creating production-standard code.

## Contents
1. solvers.py
- contains one step integration methods including:
    - euler_step
    - heun_step
    - rk4_step

- contains an error checker for the above methods

2. solve_to.py
- contains a function that integrates a differential equation to a specified time
    - solve_to

3. shooting.py
- Discretisation class: contains a class to discretise a boundary value problem
    - shooting_setup: a function to create a solvable F(u) = 0 problem for limit cycles by discretising using shooting method
    - linear: a function to linearly discretise an algebraic equation (x = x) 

- shooting_solve: a function to solve the F(u) = 0 problem using scipy.optimize.root

4. continuation.py
- Continuation class: a class to perform parameter continuation using a method:
    - nat_continuation: a function to perform natural parameter continuation
        this uses the discretisation above

    - **in development** pal_continuation: a function to perform parameter continuation using pseudo-arclength continuation

5. bvp_solver.py
- ODE class: a class to store an ODE equation, boundary conditions and initial conditions
- Solver class: a class to solve an ODE using a method and store the solution:
    - scipy_solve: a function to solve an ODE using scipy.optimize.root
    - numpy_solve: a function to solve an ODE using numpy.linalg.solve
    - tdma_solve: a function to solve an ODE using the Thomas algorithm (from the helpers.py file)

6. pde_solver.py
- PDE class: a class to store a PDE equation, boundary conditions and initial conditions
- Solver class: a class to solve a PDE using a method and store the solution:
    - solve_ivp: a function to solve a PDE using scipy.integrate.solve_ivp
    - implicit_euler_solve: a function to solve a PDE using the implicit euler method
    - crank_nicolson_solve: a function to solve a PDE using the crank-nicolson method
    - imex_euler_solve: a function to solve a PDE using the implicit-explicit euler method
    - custom_solve: a function to solve a PDE using the explicit methods from solver.py

7. helpers.py
Contains helper functions:
- boundary: this function returns the boundary condition matrix and vector for a given boundary condition type
- tdma: this function solves a tridiagonal matrix using the Thomas algorithm

8. report.ipynb
Final report for the module

9. test_equations.py
Example equations to test the solvers and used in the report

10. unit_test.py
Unit tests for the solvers - run using pytest and ci.yaml

11. README.md

12. requirements.txt

## Installation

To install the module, clone the repository and run the following command in the terminal:
```bash
pip install -r requirements.txt
```

## Overview

The module is split into;
- solvers.py
- solve_to.py
- shooting.py
- continuation.py
- bvp_solver.py
- pde_solver.py
- helpers.py
- report.ipynb
- test_equations.py
- unit_test.py










 
