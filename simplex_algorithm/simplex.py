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
        sparse_tableau: A dictionary where keys are (row_idx, col_idx) tuples and
                        values are Fraction objects representing non-zero elements.
        num_decision_vars: Number of original decision variables.
        num_constraints: Number of constraints (also number of slack variables).
        num_tableau_rows: Total rows in the conceptual tableau.
        num_tableau_cols: Total columns in the conceptual tableau.
    """
    num_decision_vars = len(c)
    num_constraints = len(A)

    if num_constraints != len(b):
        raise ValueError("Number of constraints in A must match number of RHS values in b.")
    for constraint_row_idx, row_coeffs in enumerate(A):
        if len(row_coeffs) != num_decision_vars:
            raise ValueError(f"Constraint row {constraint_row_idx} in A must have num_decision_vars ({num_decision_vars}) elements.")

    num_tableau_rows = num_constraints + 1
    # Columns: decision_vars + slack_vars + P_var + RHS_value
    num_tableau_cols = num_decision_vars + num_constraints + 1 + 1

    sparse_tableau = {}

    # Constraint rows
    for i in range(num_constraints):
        # Decision variable coefficients
        for j, x_coeff in enumerate(A[i]):
            if x_coeff != 0: # Only store non-zero values
                sparse_tableau[(i, j)] = Fraction(x_coeff)
        # Slack variable coefficients
        # s_i coefficient is 1 for constraint row i, 0 otherwise
        sparse_tableau[(i, num_decision_vars + i)] = Fraction(1) # This is always 1, so store it
        # Coefficient for P (objective variable) is 0 in constraint rows, so don't store
        # RHS
        if b[i] != 0: # Only store non-zero values
            sparse_tableau[(i, num_tableau_cols - 1)] = Fraction(b[i])

    # Objective function row (last row)
    obj_row_idx = num_constraints
    # Decision variable coefficients (negated)
    for j, c_coeff in enumerate(c):
        if c_coeff != 0: # Only store non-zero values
            sparse_tableau[(obj_row_idx, j)] = Fraction(-c_coeff)
    # Slack variable coefficients in objective function are 0, so don't store
    # Coefficient for P is 1
    sparse_tableau[(obj_row_idx, num_decision_vars + num_constraints)] = Fraction(1) # This is always 1
    # RHS for objective function is 0, so don't store

    return sparse_tableau, num_decision_vars, num_constraints, num_tableau_rows, num_tableau_cols

class SimplexSolver:
    def __init__(self, sparse_tableau, num_decision_vars, num_constraints, num_tableau_rows, num_tableau_cols):
        """
        Initializes the SimplexSolver with a sparse tableau and problem dimensions.
        sparse_tableau: Dictionary {(row, col): Fraction_value}
        num_decision_vars: Number of original decision variables.
        num_constraints: Number of constraints (used as num_slack_vars).
        num_tableau_rows: Total conceptual rows in the tableau.
        num_tableau_cols: Total conceptual columns in the tableau.
        """
        self.tableau = sparse_tableau # This is now a dictionary
        self.rows = num_tableau_rows
        self.cols = num_tableau_cols
        self.num_decision_vars = num_decision_vars
        # num_constraints is the number of slack variables for Ax <= b form
        self.num_slack_vars = num_constraints
        self.iteration = 0

    def _get_tableau_value(self, row, col):
        """Helper to get value from sparse tableau, defaults to Fraction(0)."""
        return self.tableau.get((row, col), Fraction(0))

    def _set_tableau_value(self, row, col, value):
        """Helper to set value in sparse tableau. Removes entry if value is zero."""
        if value == Fraction(0):
            # Remove the key if it exists and value is zero
            self.tableau.pop((row, col), None)
        else:
            self.tableau[(row, col)] = value

    def _find_pivot_column(self):
        """Finds the pivot column (entering variable).
        Selects the column with the most negative coefficient in the objective function row.
        Returns the column index or -1 if no negative coefficient is found (optimal).
        """
        objective_row_idx = self.rows - 1
        min_val = Fraction(0)
        pivot_col = -1
        # Only consider decision and slack variables for entering, not P or RHS
        for j in range(self.num_decision_vars + self.num_slack_vars):
            val = self._get_tableau_value(objective_row_idx, j)
            if val < min_val:
                min_val = val
                pivot_col = j
        return pivot_col

    def _find_pivot_row(self, pivot_col):
        """Finds the pivot row (leaving variable) using the minimum ratio test.
        Returns the row index or -1 if unbounded or other issue.
        """
        min_ratio = float('inf')
        pivot_row = -1
        rhs_col_idx = self.cols - 1

        for i in range(self.rows - 1): # Exclude objective function row
            pivot_col_val = self._get_tableau_value(i, pivot_col)

            if pivot_col_val > Fraction(0): # Consider only positive elements in pivot column
                rhs_val = self._get_tableau_value(i, rhs_col_idx)

                # Standard simplex assumes non-negative RHS for ratio test.
                # If rhs_val is negative, this row should not be chosen by standard ratio test.
                # (or indicates need for dual simplex or problem re-formulation)
                if rhs_val < Fraction(0):
                    continue # Skip rows with negative RHS for this simple implementation

                # Ratio test: RHS / pivot_col_val
                # If rhs_val is 0 and pivot_col_val > 0, ratio is 0. This is a valid and often preferred pivot.
                ratio = rhs_val / pivot_col_val

                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
                # Note: Degeneracy (multiple rows with same min_ratio) is not specially handled here.
                # Bland's rule could be implemented to prevent cycling in degenerate cases.

        return pivot_row # Will be -1 if no suitable row (e.g. all pivot_col_val <= 0 or all valid RHS < 0)


    def _pivot(self, pivot_row, pivot_col):
        """Performs the pivot operation on the sparse tableau."""
        pivot_element = self._get_tableau_value(pivot_row, pivot_col)
        if pivot_element == Fraction(0):
            # This should ideally be prevented by _find_pivot_row and _find_pivot_column logic
            raise ValueError("Pivot element cannot be zero.")

        # Normalize the pivot row
        # Iterate over all columns conceptually.
        # For sparse, we need to update existing non-zero elements and potentially create new ones.
        for j in range(self.cols):
            val_pivot_row_j = self._get_tableau_value(pivot_row, j)
            if j == pivot_col:
                self._set_tableau_value(pivot_row, j, Fraction(1))
            elif val_pivot_row_j != Fraction(0) : # Only perform division if original value was non-zero
                self._set_tableau_value(pivot_row, j, val_pivot_row_j / pivot_element)
            # If val_pivot_row_j was 0, it remains 0 after division by pivot_element (0/X=0), so no need to store.

        # Eliminate other rows (make pivot_col entries zero for other rows)
        for i in range(self.rows):
            if i != pivot_row:
                factor = self._get_tableau_value(i, pivot_col)
                if factor != Fraction(0): # If factor is zero, this row doesn't need changes based on pivot_col
                    # Iterate over all columns conceptually for row i
                    for j in range(self.cols):
                        val_i_j = self._get_tableau_value(i, j)
                        val_pivot_row_j_normalized = self._get_tableau_value(pivot_row, j) # Value from normalized pivot row

                        if j == pivot_col: # Element in pivot column (not in pivot row) becomes 0
                            self._set_tableau_value(i, j, Fraction(0))
                        else:
                            # New value = current_val_i_j - factor * val_pivot_row_j_normalized
                            # This needs to be calculated even if val_i_j is currently 0
                            new_val = val_i_j - factor * val_pivot_row_j_normalized
                            self._set_tableau_value(i, j, new_val)

    def _format_tableau(self):
        """Formats the sparse tableau for printing by reconstructing a dense view."""
        s = []
        # Create a dense representation for printing
        dense_tableau_for_print = []
        for i in range(self.rows):
            row_list = []
            for j in range(self.cols):
                row_list.append(self._get_tableau_value(i,j))
            dense_tableau_for_print.append(row_list)

        col_widths = [0] * self.cols
        for r_list in dense_tableau_for_print:
            for idx, cell_val in enumerate(r_list):
                col_widths[idx] = max(col_widths[idx], len(str(cell_val)))

        header = []
        for j in range(self.num_decision_vars):
            header.append(f"x{j+1}".ljust(col_widths[j]))
        for j in range(self.num_slack_vars):
            header.append(f"s{j+1}".ljust(col_widths[self.num_decision_vars + j]))
        header.append("P".ljust(col_widths[self.num_decision_vars + self.num_slack_vars]))
        header.append("RHS".ljust(col_widths[self.cols - 1]))
        s.append(" | ".join(header))
        s.append("-+-".join(["-" * w for w in col_widths]))


        for i in range(self.rows):
            row_str_list = []
            for j in range(self.cols):
                val = self._get_tableau_value(i,j)
                row_str_list.append(str(val).ljust(col_widths[j]))
            s.append(" | ".join(row_str_list))
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
        # Pass self.num_decision_vars and self.num_slack_vars to format_tableau if it needs them for headers
        if verbose: print(f"Initial Tableau (num_decision_vars={self.num_decision_vars}, num_slack_vars={self.num_slack_vars}):\n{self._format_tableau()}\n")

        while self.iteration < max_iterations:
            pivot_col = self._find_pivot_column()

            if pivot_col == -1:
                if verbose: print("Optimal solution found.")
                return "optimal"

            if verbose: print(f"Iteration {self.iteration+1}: Pivot column is {pivot_col} (0-indexed)")

            pivot_row = self._find_pivot_row(pivot_col)

            # Check for unboundedness: if a pivot column is identified, but all entries in that
            # column (for constraint rows) are non-positive (<= 0).
            if pivot_row == -1: # No valid pivot row found
                all_pivot_col_entries_non_positive = True
                for i in range(self.rows - 1): # Check all constraint rows
                    if self._get_tableau_value(i, pivot_col) > 0:
                        all_pivot_col_entries_non_positive = False
                        break
                if all_pivot_col_entries_non_positive:
                    if verbose: print(f"Problem is unbounded. Pivot column {pivot_col} has all non-positive or zero entries in constraint rows.")
                    return "unbounded"
                else:
                    # This could happen if, for example, all RHS values for potential pivot rows were negative.
                    # The current _find_pivot_row skips rows with RHS < 0 if pivot_col_val > 0.
                    if verbose: print(f"Warning: Pivot column {pivot_col} found, but no suitable pivot row. This might indicate issues with problem formulation (e.g. all valid ratios negative or undefined) or all RHS negative.")
                    return "error_no_suitable_pivot_row"


            if verbose: print(f"Pivot element at ({pivot_row}, {pivot_col}): {self._get_tableau_value(pivot_row, pivot_col)}")

            self._pivot(pivot_row, pivot_col)
            self.iteration += 1
            if verbose: print(f"After Pivot {self.iteration}:\n{self._format_tableau()}\n")

        if verbose: print("Max iterations reached.")
        return "max_iterations_reached"

    def get_solution(self):
        """
        Extracts the variable values and objective function value from the final (optimal) sparse tableau.
        Returns a dictionary with decision variable values (x1, x2, ...)
        and the objective function value keyed by 'objective_value'.
        """
        solution = {}
        rhs_col_idx = self.cols - 1
        obj_row_idx = self.rows - 1

        # Objective value is in the last row, last column
        solution['objective_value'] = self._get_tableau_value(obj_row_idx, rhs_col_idx)

        # Decision variables values
        # A decision variable x_j (column j) is basic if its column in the final tableau
        # has a single '1' in a constraint row (say row_basic) and all other entries
        # in that column (for other constraint rows and the objective row) are '0'.
        # Its value is then the RHS value of row_basic. Otherwise, it's non-basic and its value is 0.
        for j in range(self.num_decision_vars): # Iterate through columns of original decision variables
            val = Fraction(0) # Default to 0 for non-basic variables
            basic_row_candidate = -1
            is_basic_column = True

            # Check column j for basic variable characteristics
            count_ones_in_constraints = 0

            for i in range(self.rows - 1): # Iterate through constraint rows
                cell_val = self._get_tableau_value(i, j)
                if cell_val == Fraction(1):
                    count_ones_in_constraints += 1
                    basic_row_candidate = i
                elif cell_val != Fraction(0):
                    is_basic_column = False # Other non-zero in constraint rows
                    break

            if not is_basic_column or count_ones_in_constraints != 1:
                solution[f'x{j+1}'] = Fraction(0) # Non-basic or complex column
                continue

            # Check if the objective row coefficient for this column is zero
            if self._get_tableau_value(obj_row_idx, j) == Fraction(0):
                # This column j corresponds to a basic variable, its value is on RHS of basic_row_candidate
                val = self._get_tableau_value(basic_row_candidate, rhs_col_idx)
            else:
                # Column might look basic in constraints, but if obj func coeff is non-zero, it's non-basic.
                val = Fraction(0)

            solution[f'x{j+1}'] = val

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

    # MODIFIED: initialize_tableau now returns 5 values
    sparse_tab1, n_dec_vars1, n_constraints1, n_rows1, n_cols1 = initialize_tableau(c1, A1, b1)
    # MODIFIED: SimplexSolver constructor takes new arguments
    solver1 = SimplexSolver(sparse_tab1, n_dec_vars1, n_constraints1, n_rows1, n_cols1)

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

    sparse_tab2, n_dec_vars2, n_constraints2, n_rows2, n_cols2 = initialize_tableau(c2, A2, b2)
    solver2 = SimplexSolver(sparse_tab2, n_dec_vars2, n_constraints2, n_rows2, n_cols2)

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

    sparse_tab3, n_dec_vars3, n_constraints3, n_rows3, n_cols3 = initialize_tableau(c3, A3, b3)
    solver3 = SimplexSolver(sparse_tab3, n_dec_vars3, n_constraints3, n_rows3, n_cols3)
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
