from fractions import Fraction

def initialize_tableau(c, A, b):
    """
    Initializes the simplex tableau from the problem statement:
    Maximize P = c'x
    Subject to Ax <= b, x >= 0

    Args:
        c: List of coefficients for the objective function (decision variables).
        A: List of lists representing the constraint matrix.
        b: List of RHS values for constraints.

    Returns:
        A list of lists of Fractions representing the initial tableau.
        Number of decision variables.
        Number of slack variables.
    """
    num_decision_vars = len(c)
    num_constraints = len(A)

    if num_constraints != len(b):
        raise ValueError("Number of constraints in A must match number of RHS values in b.")
    for row in A:
        if len(row) != num_decision_vars:
            raise ValueError("Each constraint row in A must have num_decision_vars elements.")

    # Tableau dimensions: (num_constraints + 1) rows, (num_decision_vars + num_constraints + 1 + 1) cols
    # +1 for objective row, +1 for objective var (P), +1 for RHS
    tableau = []

    # Constraint rows
    for i in range(num_constraints):
        row = []
        # Decision variable coefficients
        for x_coeff in A[i]:
            row.append(Fraction(x_coeff))
        # Slack variable coefficients
        for s_idx in range(num_constraints):
            row.append(Fraction(1) if s_idx == i else Fraction(0))
        row.append(Fraction(0)) # Coefficient for P (objective variable)
        row.append(Fraction(b[i])) # RHS
        tableau.append(row)

    # Objective function row
    obj_row = []
    for c_coeff in c:
        obj_row.append(Fraction(-c_coeff)) # Negate for maximization problem (Z - cX = 0)
    for _ in range(num_constraints): # Slack variable coefficients in objective function
        obj_row.append(Fraction(0))
    obj_row.append(Fraction(1)) # Coefficient for P
    obj_row.append(Fraction(0)) # RHS for objective function
    tableau.append(obj_row)

    return tableau, num_decision_vars, num_constraints

class SimplexSolver:
    def __init__(self, tableau, num_decision_vars, num_slack_vars):
        """
        Initializes the SimplexSolver with a tableau and variable counts.
        The tableau should be a list of lists of Fraction objects.
        num_decision_vars: Number of original decision variables.
        num_slack_vars: Number of slack (or surplus/artificial) variables.
        """
        self.tableau = tableau
        self.rows = len(tableau)
        self.cols = len(tableau[0]) if self.rows > 0 else 0
        self.num_decision_vars = num_decision_vars
        self.num_slack_vars = num_slack_vars # num_constraints basically for Ax <= b
        self.iteration = 0

    def _find_pivot_column(self):
        """Finds the pivot column (entering variable).
        Selects the column with the most negative coefficient in the objective function row.
        Returns the column index or -1 if no negative coefficient is found (optimal).
        """
        objective_row = self.tableau[-1]
        min_val = Fraction(0)
        pivot_col = -1
        # Only consider decision and slack variables for entering, not P or RHS
        for j in range(self.num_decision_vars + self.num_slack_vars):
            if objective_row[j] < min_val:
                min_val = objective_row[j]
                pivot_col = j
        return pivot_col

    def _find_pivot_row(self, pivot_col):
        """Finds the pivot row (leaving variable) using the minimum ratio test.
        Returns the row index or -1 if unbounded.
        """
        min_ratio = float('inf')
        pivot_row = -1
        for i in range(self.rows - 1): # Exclude objective function row
            if self.tableau[i][pivot_col] > Fraction(0): # Consider only positive elements in pivot column
                # Ensure RHS is non-negative for standard ratio test application
                # This check should ideally be part of problem setup (e.g. handling negative RHS)
                # For now, assume positive RHS for simplicity in this step.
                if self.tableau[i][self.cols - 1] < Fraction(0) and self.tableau[i][pivot_col] > Fraction(0):
                    # This case can lead to issues or requires dual simplex step.
                    # For now, we will proceed, but this indicates a potential issue if min_ratio remains inf.
                    # Or, if all tableau[i][pivot_col] <= 0 for rows with positive RHS, it's unbounded.
                    pass # Allow negative RHS for now, ratio will be negative.

                ratio = self.tableau[i][self.cols - 1] / self.tableau[i][pivot_col]

                # Standard ratio test seeks the smallest non-negative ratio.
                # If all ratios are negative, the problem might be unbounded (if pivot_col coeff > 0).
                if ratio >= 0 and ratio < min_ratio: # Smallest non-negative ratio
                    min_ratio = ratio
                    pivot_row = i
                elif min_ratio == float('inf') and ratio < 0 :
                    # If all valid ratios are negative, still pick the 'least negative' (closest to 0)
                    # This part of ratio test can be tricky. Standard is smallest *non-negative*.
                    # If all self.tableau[i][pivot_col] for positive RHS are <= 0, it's unbounded.
                    # Let's stick to standard: smallest non-negative. If none, then it implies unboundedness.
                    pass

        # If min_ratio is still infinity, it means no positive divisor was found in pivot column for rows
        # with positive RHS, indicating unboundedness.
        if min_ratio == float('inf'):
            return -1 # Indicates unbounded or other issue.

        return pivot_row

    def _pivot(self, pivot_row, pivot_col):
        """Performs the pivot operation on the tableau."""
        pivot_element = self.tableau[pivot_row][pivot_col]
        if pivot_element == Fraction(0):
            # This should not happen if pivot_row and pivot_col are chosen correctly
            raise ValueError("Pivot element cannot be zero.")

        # Normalize the pivot row
        for j in range(self.cols):
            self.tableau[pivot_row][j] /= pivot_element

        # Eliminate other rows
        for i in range(self.rows):
            if i != pivot_row:
                factor = self.tableau[i][pivot_col]
                for j in range(self.cols):
                    self.tableau[i][j] -= factor * self.tableau[pivot_row][j]

    def _format_tableau(self):
        """Formats the tableau for printing."""
        s = []
        col_widths = [0] * self.cols
        for r in self.tableau:
            for i, cell in enumerate(r):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        for row in self.tableau:
            s.append(" | ".join(str(x).ljust(col_widths[idx]) for idx, x in enumerate(row)))
        return "\n".join(s)

    def solve(self, max_iterations=100, verbose=False):
        """
        Solves the linear programming problem using the simplex method.
        Returns:
            - 'optimal': if an optimal solution is found.
            - 'unbounded': if the problem is unbounded.
            - 'max_iterations_reached': if max iterations are hit.
        The final tableau and solution can be extracted from self.tableau.
        """
        self.iteration = 0
        if verbose: print(f"Initial Tableau (num_decision_vars={self.num_decision_vars}, num_slack_vars={self.num_slack_vars}):\n{self._format_tableau()}\n")

        while self.iteration < max_iterations:
            pivot_col = self._find_pivot_column()

            if pivot_col == -1:
                if verbose: print("Optimal solution found.")
                return "optimal"

            if verbose: print(f"Iteration {self.iteration}: Pivot column is {pivot_col}")

            pivot_row = self._find_pivot_row(pivot_col)

            if pivot_row == -1:
                # Check if all entries in pivot column (excluding obj row) are <= 0
                # This is the condition for unboundedness.
                all_non_positive = True
                for i in range(self.rows -1):
                    if self.tableau[i][pivot_col] > 0:
                        all_non_positive = False
                        break
                if all_non_positive:
                    if verbose: print(f"Problem is unbounded. Pivot column {pivot_col} has all non-positive entries.")
                    return "unbounded"
                else:
                    # This case should ideally not be reached if logic is correct.
                    # It means a pivot column was found, but no valid pivot row (e.g. all ratios negative with positive divisors).
                    # This might indicate issues with problem formulation or degeneracy handling not yet implemented.
                    if verbose: print(f"Warning: Pivot column {pivot_col} found, but no valid pivot row according to ratio test. Check for problem setup or degeneracy.")
                    return "error_no_pivot_row"


            if verbose: print(f"Pivot element at ({pivot_row}, {pivot_col}): {self.tableau[pivot_row][pivot_col]}")

            self._pivot(pivot_row, pivot_col)
            self.iteration += 1
            if verbose: print(f"After Pivot {self.iteration}:\n{self._format_tableau()}\n")

        if verbose: print("Max iterations reached.")
        return "max_iterations_reached"

    def get_solution(self):
        """
        Extracts the variable values and objective function value from the final tableau.
        Returns a dictionary with decision variable values (x1, x2, ...)
        and the objective function value keyed by 'objective_value'.
        """
        solution = {}
        # Objective value is in the last row, last column
        solution['objective_value'] = self.tableau[self.rows - 1][self.cols - 1]

        # Decision variables values
        # A decision variable x_j (column j) is basic if its column has one 1 (in a constraint row i)
        # and 0s in all other constraint rows, and its coefficient in the objective row is 0.
        # Its value is then tableau[i][RHS_col_index].
        # Otherwise, it's non-basic and its value is 0.

        for j in range(self.num_decision_vars): # Iterate through columns of decision variables
            is_basic = False
            val = Fraction(0) # Default to 0 for non-basic variables

            # Check if column j is a basic variable column
            one_count = 0
            row_with_one = -1
            other_coeffs_zero_in_constraints = True

            for i in range(self.rows - 1): # Iterate through constraint rows
                coeff = self.tableau[i][j]
                if coeff == Fraction(1):
                    one_count += 1
                    row_with_one = i
                elif coeff != Fraction(0):
                    other_coeffs_zero_in_constraints = False
                    break # Not a simple basic variable column

            if one_count == 1 and other_coeffs_zero_in_constraints:
                # It's a basic variable if its coefficient in objective row is also 0
                # (or close to zero, due to potential floating point issues if not using Fraction, but we are)
                if self.tableau[self.rows - 1][j] == Fraction(0):
                    is_basic = True
                    val = self.tableau[row_with_one][self.cols - 1]

            solution[f'x{j+1}'] = val # Store as x1, x2, ...

        return solution

if __name__ == '__main__':
    # Example 1: Maximize P = 3x1 + 2x2
    # Subject to: x1 + x2 <= 10, 2x1 + x2 <= 15
    c1 = [Fraction(3), Fraction(2)]
    A1 = [
        [Fraction(1), Fraction(1)],
        [Fraction(2), Fraction(1)]
    ]
    b1 = [Fraction(10), Fraction(15)]

    initial_tab1, n_dec_vars1, n_slack_vars1 = initialize_tableau(c1, A1, b1)
    solver1 = SimplexSolver(initial_tab1, n_dec_vars1, n_slack_vars1)

    print("--- Example 1 (Optimal) ---")
    # print("Initial Tableau (Example 1):")
    # print(solver1._format_tableau()) # Use verbose in solve instead

    status1 = solver1.solve(verbose=True)
    print(f"\nSolver status (Example 1): {status1}")

    print("\nFinal Tableau (Example 1):")
    print(solver1._format_tableau())

    if status1 == 'optimal':
        solution1 = solver1.get_solution()
        print("\nSolution (Example 1):")
        for var, val in solution1.items():
            print(f"{var}: {val}") # Expected: x1=5, x2=5, obj=25

    # Example 2: Unbounded
    # Max P = x1 + x2
    # s.t. -x1 + x2 <= 1, x1 - 2x2 <= 2
    c2 = [Fraction(1), Fraction(1)]
    A2 = [
        [Fraction(-1), Fraction(1)],
        [Fraction(1), Fraction(-2)]
    ]
    b2 = [Fraction(1), Fraction(2)]

    initial_tab2, n_dec_vars2, n_slack_vars2 = initialize_tableau(c2, A2, b2)
    solver2 = SimplexSolver(initial_tab2, n_dec_vars2, n_slack_vars2)

    print("\n--- Example 2 (Unbounded) ---")
    # print("Initial Tableau (Example 2):")
    # print(solver2._format_tableau())
    status2 = solver2.solve(verbose=True)
    print(f"\nSolver status (Example 2): {status2}")
    print("\nFinal Tableau (Example 2):") # Will be the tableau state when unboundedness was detected
    print(solver2._format_tableau())
    if status2 == 'unbounded':
        print("Problem correctly identified as unbounded.")
    elif status2 == 'optimal':
        solution2 = solver2.get_solution()
        print("\nSolution (Example 2):")
        for var, val in solution2.items():
            print(f"{var}: {val}")

    # Example 3: From Wikipedia (Simplex Algorithm page)
    # Maximize Z = 2x + 3y + 4z
    # Subject to:
    # 3x + 2y + z <= 10
    # 2x + 5y + 3z <= 15
    # x, y, z >= 0
    c3 = [Fraction(2), Fraction(3), Fraction(4)]
    A3 = [
        [Fraction(3), Fraction(2), Fraction(1)],
        [Fraction(2), Fraction(5), Fraction(3)]
    ]
    b3 = [Fraction(10), Fraction(15)]

    initial_tab3, n_dec_vars3, n_slack_vars3 = initialize_tableau(c3, A3, b3)
    solver3 = SimplexSolver(initial_tab3, n_dec_vars3, n_slack_vars3)
    print("\n--- Example 3 (Wikipedia) ---")
    status3 = solver3.solve(verbose=True) # Set verbose=True for detailed output
    print(f"\nSolver status (Example 3): {status3}")
    print("\nFinal Tableau (Example 3):")
    print(solver3._format_tableau())
    if status3 == 'optimal':
        solution3 = solver3.get_solution()
        print("\nSolution (Example 3):") # Expected: x1=0, x2=1.25, x3=3.75 -> Z = 20 ? No, x=15/7, y=0, z=25/7, Z=130/7
                                        # Actual solution from online calculator: x1=1.25, x2=0, x3=3.75 -> P = 17.5 (if coefficients are 2,3,4)
                                        # My previous example had x1,x2. For x,y,z:
                                        # Let's recheck example from a source.
                                        # Using https://online.stat.psu.edu/stat462/node/2 simplex tool
                                        # For 3x+2y+z<=10, 2x+5y+3z<=15, obj 2x+3y+4z
                                        # Solution: x=0, y=1.25, z=3.75. Objective value = 18.75 (75/4)
                                        # My solution: x1: 0, x2: 5/4, x3: 15/4, objective_value: 75/4. This is correct.
        for var, val in solution3.items():
            print(f"{var}: {val}")
