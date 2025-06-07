# Exact Simplex Algorithm for Linear Programming

This Python package provides an implementation of the simplex algorithm for solving linear programming problems. It uses Python's `fractions.Fraction` type throughout its calculations to ensure exact rational arithmetic, avoiding floating-point inaccuracies.

## Table of Contents
- [What is Linear Programming?](#what-is-linear-programming)
- [The Simplex Algorithm](#the-simplex-algorithm)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Defining a Problem](#defining-a-problem)
  - [Solving the Problem](#solving-the-problem)
  - [Interpreting the Solution](#interpreting-the-solution)
  - [Verbose Output](#verbose-output)
- [Running Tests](#running-tests)
- [Limitations](#limitations)
- [Contributing](#contributing)

## What is Linear Programming?
Linear Programming (LP) is a mathematical method for determining the best outcome (such as maximum profit or lowest cost) in a given mathematical model whose requirements are represented by linear relationships. LP problems involve an objective function (what to maximize or minimize) subject to a set of linear equality or inequality constraints.

## The Simplex Algorithm
The simplex algorithm, developed by George Dantzig, is a widely used algorithm for solving linear programming problems. It operates by iteratively moving from one feasible vertex of the polytope defined by the constraints to an adjacent vertex that improves the objective function value, until an optimal solution is reached or the problem is determined to be unbounded or infeasible.

This implementation focuses on the standard simplex method for problems in the form:
Maximize: \( P = c^T x \)
Subject to: \( Ax \leq b \) and \( x \geq 0 \)
It can also handle problems requiring artificial variables for an initial feasible solution by using the two-phase simplex method.

## Features
- Solves linear programming problems (maximization).
- Uses `fractions.Fraction` for exact arithmetic, avoiding floating-point errors.
- Detects optimal solutions and unbounded problems.
- Handles problems requiring the two-phase simplex method (e.g., those with '>=' constraints).
- Provides a clear way to input LP problems and retrieve solutions.
- Includes unit tests for reliability.

## Installation
Currently, this package is set up as a local library. To use it, ensure the `simplex_algorithm` directory is in your Python path.

If this were to be packaged for distribution (e.g., on PyPI), you would typically install it using pip:
```bash
pip install exact_simplex # Hypothetical package name
```
For now, you can clone the repository and use the code directly.

## Usage

Here's how to use the `SimplexSolver` to solve an LP problem:

### Defining a Problem

First, import the necessary components and define your problem using coefficients for the objective function (`c`), the constraint matrix (`A`), and the right-hand side values of the constraints (`b`). All numerical values should be `fractions.Fraction` objects for exact arithmetic.

```python
from fractions import Fraction
# initialize_tableau is an internal function; SimplexSolver is the public interface.
from simplex_algorithm.simplex import SimplexSolver

# Example Problem:
# Maximize P = 3x1 + 2x2
# Subject to:
#   x1 + x2 <= 10
#   2x1 + x2 <= 15
#   x1, x2 >= 0

# Coefficients of the objective function (c)
c = [Fraction(3), Fraction(2)]

# Constraint matrix (A)
A = [
    [Fraction(1), Fraction(1)],  # Constraint 1: x1 + x2
    [Fraction(2), Fraction(1)]   # Constraint 2: 2x1 + x2
]

# Right-hand side values of constraints (b)
b = [Fraction(10), Fraction(15)] # Corresponding to <= 10 and <= 15
```

For problems with many variables where most coefficients are zero, a sparse input format can be more convenient. See the "Using Sparse Input Format" section below for details.

### Using Sparse Input Format

For problems where the objective function `c` or constraint matrix `A` are sparse (i.e., contain many zero coefficients), you can provide them as dictionaries. This can be more readable and potentially more efficient for tableau initialization if the problem is very large and very sparse.

-   **`c` (objective function coefficients):** A dictionary where keys are 0-indexed integer variable numbers and values are `Fraction` coefficients. Variables not included in the dictionary are assumed to have a coefficient of zero.
    Example: `c_sparse = {0: Fraction(3), 2: Fraction(2)}` represents \(P = 3x_1 + 0x_2 + 2x_3\).
-   **`A` (constraint matrix):** A list of dictionaries. Each dictionary represents a constraint row, with keys as 0-indexed integer variable numbers and values as `Fraction` coefficients.
    Example: `A_sparse = [{0: Fraction(1), 1: Fraction(1)}, {0: Fraction(2), 2: Fraction(1)}]`.
-   **`b` (RHS values):** Remains a list of `Fraction` objects, corresponding to each constraint row in `A_sparse`.

To use this format, pass `sparse_input=True` when creating the `SimplexSolver` instance:

```python
from fractions import Fraction
from simplex_algorithm.simplex import SimplexSolver

# Example Sparse Problem:
# Maximize P = 3x1 + 0x2 + 2x3  (Note: x2 has coefficient 0)
# Subject to:
#   1x1 + 1x2 + 0x3 <= 10
#   2x1 + 0x2 + 1x3 <= 15
#   x1, x2, x3 >= 0

c_sparse = {0: Fraction(3), 2: Fraction(2)}  # x2's objective coefficient is 0
A_sparse = [
    {0: Fraction(1), 1: Fraction(1)},      # Constraint 1: 1x1 + 1x2 (+ 0x3)
    {0: Fraction(2), 2: Fraction(1)}       # Constraint 2: 2x1 (+ 0x2) + 1x3
]
b_rhs = [Fraction(10), Fraction(15)]

# Initialize SimplexSolver with sparse_input=True
# The solver will determine the total number of decision variables based on the maximum index found in c_sparse and A_sparse.
# In this example, max index is 2, so it assumes 3 decision variables (x1, x2, x3, indexed 0, 1, 2).
solver = SimplexSolver(c_sparse, A_sparse, b_rhs, sparse_input=True)

# Then proceed as usual:
# status = solver.solve()
# if status == "optimal":
#     solution = solver.get_solution()
#     print(solution)
```

### Solving the Problem
Create a `SimplexSolver` instance with your problem definition (using either the standard list format or the sparse dictionary format as described above). Then, call the `solve()` method:

```python
# Using the standard problem definition from the first example:
c_dense = [Fraction(3), Fraction(2)]
A_dense = [
    [Fraction(1), Fraction(1)],
    [Fraction(2), Fraction(1)]
]
b_dense = [Fraction(10), Fraction(15)]

# Create a SimplexSolver instance
solver = SimplexSolver(c_dense, A_dense, b_dense)
# For sparse input, you would use:
# solver = SimplexSolver(c_sparse, A_sparse, b_rhs, sparse_input=True)

# Solve the problem
status = solver.solve()
# You can also use solver.solve(verbose=True) for step-by-step output

# Example using the sparse definition from above:
# c_sparse = {0: Fraction(3), 2: Fraction(2)}
# A_sparse = [
#     {0: Fraction(1), 1: Fraction(1)},
#     {0: Fraction(2), 2: Fraction(1)}
# ]
# b_rhs = [Fraction(10), Fraction(15)]
# solver_sparse = SimplexSolver(c_sparse, A_sparse, b_rhs, sparse_input=True)
# status_sparse = solver_sparse.solve()
# print(f"Sparse solver status: {status_sparse}")
# if status_sparse == "optimal":
#     solution_sparse = solver_sparse.get_solution()
#     print(solution_sparse)

```

### Interpreting the Solution
The `solve()` method returns a status string (`"optimal"`, `"unbounded"`, etc.). If an optimal solution is found, you can retrieve it using the `get_solution()` method.

```python
print(f"Solver status: {status}")

if status == "optimal":
    solution = solver.get_solution()
    print("\nOptimal Solution:")
    for var_name, value in solution.items():
        if var_name == 'objective_value':
            print(f"Maximum Objective Value (P): {value}")
        else:
            print(f"{var_name}: {value}")
    # Expected output for the example:
    # Maximum Objective Value (P): 25
    # x1: 5
    # x2: 5

elif status == "unbounded":
    print("The problem is unbounded.")

# Final tableau can also be inspected:
# print("\nFinal Tableau:")
# print(solver._format_tableau())
```

### Verbose Output
For debugging or educational purposes, you can enable verbose output during the solving process:
```python
status = solver.solve(verbose=True)
```
This will print the tableau at each iteration, showing the pivot selections and transformations.

## Running Tests
The package includes unit tests to ensure correctness. To run the tests, navigate to the root directory of the project and run:

```bash
python -m unittest discover -s tests
```
Or, more simply, if `tests/test_simplex.py` is the only test file:
```bash
python -m unittest tests.test_simplex
```

## Limitations
- **Standard Form Focus**: While the two-phase simplex method allows handling various constraint types, the core implementation is primarily built around the standard form (Ax <= b, x >= 0) for maximization.
- **No Anti-Cycling Rules**: Does not implement anti-cycling rules (e.g., Bland's rule). While cycling is rare in practice, it's a theoretical possibility for degenerate problems.
- **Minimization**: To solve minimization problems (minimize Z = c'x), convert it to a maximization problem (maximize P = -Z = -c'x).

## Contributing
Contributions are welcome! If you find issues or have suggestions for improvements, please open an issue or submit a pull request.
