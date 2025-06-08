from fractions import Fraction

def initialize_tableau(c, A_orig, b_orig):
    """
    Initializes the simplex tableau.
    Maximize P = c'x
    Subject to Ax <= b (or Ax >= b if b_i was negative)
    This function ensures all RHS values (b) are non-negative in the tableau.
    If b_i < 0 for a constraint A_i x <= b_i, it's converted to
    -A_i x >= -b_i. A surplus variable e_i is added: -A_i x - e_i = -b_i.

    Args:
        c: List of coefficients for the objective function (decision variables).
        A_orig: List of lists representing the constraint matrix.
        b_orig: List of RHS values for constraints.

    Returns:
        sparse_tableau: Dictionary {(row, col): Fraction_value}.
        num_decision_vars: Number of original decision variables.
        num_aux_vars: Number of auxiliary (slack/surplus) variables.
        num_tableau_rows: Total rows in the tableau.
        num_tableau_cols: Total columns in the tableau.
        constraint_types: List of strings ('slack' or 'surplus') for each constraint.
    """
    num_decision_vars = len(c)
    num_constraints = len(A_orig) # This is also num_aux_vars

    if num_constraints != len(b_orig):
        raise ValueError("Number of constraints in A must match number of RHS values in b.")
    for constraint_row_idx, row_coeffs in enumerate(A_orig):
        if len(row_coeffs) != num_decision_vars:
            raise ValueError(f"Constraint row {constraint_row_idx} in A must have num_decision_vars ({num_decision_vars}) elements.")

    A = [list(row) for row in A_orig] # Create copies to modify
    b = list(b_orig)
    constraint_types = []

    num_tableau_rows = num_constraints + 1
    # Columns: decision_vars + aux_vars (slack/surplus) + P_var + RHS_value
    num_tableau_cols = num_decision_vars + num_constraints + 1 + 1

    sparse_tableau = {}

    # Constraint rows
    for i in range(num_constraints):
        current_A_row = A[i]
        current_b_val = b[i]

        if current_b_val < 0:
            # Multiply constraint by -1: -A_i x >= -b_i (becomes positive)
            # Add surplus variable e_i: -A_i x - e_i = -b_i
            current_A_row = [-coeff for coeff in current_A_row]
            current_b_val = -current_b_val
            constraint_types.append('surplus')
            aux_var_coeff = Fraction(-1)
        else:
            # Original A_i x <= b_i
            # Add slack variable s_i: A_i x + s_i = b_i
            constraint_types.append('slack')
            aux_var_coeff = Fraction(1)

        # Decision variable coefficients
        for j, x_coeff in enumerate(current_A_row):
            if x_coeff != 0:
                sparse_tableau[(i, j)] = Fraction(x_coeff)

        # Auxiliary (slack or surplus) variable coefficient
        # For constraint row i, aux_var_i has its specific coefficient, 0 otherwise
        sparse_tableau[(i, num_decision_vars + i)] = aux_var_coeff

        # RHS
        if current_b_val != 0: # Should always be non-negative now
            sparse_tableau[(i, num_tableau_cols - 1)] = Fraction(current_b_val)

    # Objective function row (last row)
    obj_row_idx = num_constraints
    # Decision variable coefficients (negated)
    for j, c_coeff in enumerate(c):
        if c_coeff != 0: # Only store non-zero values
            sparse_tableau[(obj_row_idx, j)] = Fraction(-c_coeff)
    # Auxiliary variable coefficients in objective function are 0, so don't store
    # Coefficient for P is 1
    sparse_tableau[(obj_row_idx, num_decision_vars + num_constraints)] = Fraction(1) # This is always 1
    # RHS for objective function is 0, so don't store

    return sparse_tableau, num_decision_vars, num_constraints, num_tableau_rows, num_tableau_cols, constraint_types


def initialize_tableau_sparse(c_sparse, A_sparse, b_list):
    """
    Initializes the simplex tableau from sparse input formats.
    Maximize P = c'x
    Subject to Ax <= b (or Ax >= b if b_i was negative)
    This function ensures all RHS values (b) are non-negative in the tableau.

    Args:
        c_sparse: Dictionary of coefficients for the objective function
                  (keyed by integer variable number, 0-indexed).
        A_sparse: List of dictionaries for the constraint matrix. Each dictionary
                  represents a row, keyed by integer variable number (0-indexed).
        b_list: List of RHS values (fractions.Fraction) for constraints.

    Returns:
        sparse_tableau: Dictionary {(row, col): Fraction_value}.
        num_decision_vars: Number of original decision variables.
        num_aux_vars: Number of auxiliary (slack/surplus) variables (matches num_constraints).
        num_tableau_rows: Total rows in the tableau.
        num_tableau_cols: Total columns in the tableau.
        constraint_types: List of strings ('slack' or 'surplus') for each constraint.
    """
    max_var_idx = -1
    if c_sparse:
        max_var_idx = max(max_var_idx, max(c_sparse.keys()))
    for A_row_dict in A_sparse:
        if A_row_dict:
            max_var_idx = max(max_var_idx, max(A_row_dict.keys()))

    num_decision_vars = max_var_idx + 1 if max_var_idx > -1 else 0

    num_constraints = len(b_list) # This is also num_aux_vars (number of slack/surplus variables)

    if len(A_sparse) != num_constraints:
        raise ValueError("Number of constraint rows in A_sparse must match number of RHS values in b_list.")

    constraint_types = []

    num_tableau_rows = num_constraints + 1
    # Columns: decision_vars + aux_vars (slack/surplus) + P_var + RHS_value
    num_tableau_cols = num_decision_vars + num_constraints + 1 + 1

    sparse_tableau = {}

    # Constraint rows
    for i in range(num_constraints):
        current_A_row_dict = A_sparse[i]
        current_b_val = b_list[i]
        aux_var_coeff = Fraction(1) # Default for slack

        if current_b_val < 0:
            # Multiply constraint by -1: -A_i x >= -b_i (becomes positive)
            # Add surplus variable e_i: -A_i x - e_i = -b_i
            # So, coefficients in current_A_row_dict are negated.
            current_b_val = -current_b_val # b_i becomes positive
            constraint_types.append('surplus')
            aux_var_coeff = Fraction(-1) # Surplus variable has -1 coeff in its equation

            # Effective coefficients for the tableau row are negated from original A_sparse[i]
            for col_idx, coeff in current_A_row_dict.items():
                if coeff != 0:
                    sparse_tableau[(i, col_idx)] = Fraction(-coeff)
        else:
            # Original A_i x <= b_i
            # Add slack variable s_i: A_i x + s_i = b_i
            constraint_types.append('slack')
            # aux_var_coeff is already Fraction(1)
            for col_idx, coeff in current_A_row_dict.items():
                if coeff != 0:
                    sparse_tableau[(i, col_idx)] = Fraction(coeff)

        # Auxiliary (slack or surplus) variable coefficient
        # For constraint row i, aux_var_i has its specific coefficient
        sparse_tableau[(i, num_decision_vars + i)] = aux_var_coeff

        # RHS
        if current_b_val != 0: # Should always be non-negative now
            sparse_tableau[(i, num_tableau_cols - 1)] = Fraction(current_b_val)

    # Objective function row (last row)
    obj_row_idx = num_constraints
    # Decision variable coefficients (negated)
    for col_idx, coeff in c_sparse.items():
        if coeff != 0: # Only store non-zero values
            sparse_tableau[(obj_row_idx, col_idx)] = Fraction(-coeff)

    # Auxiliary variable coefficients in objective function are 0, so don't store
    # Coefficient for P is 1
    sparse_tableau[(obj_row_idx, num_decision_vars + num_constraints)] = Fraction(1)
    # RHS for objective function is 0, so don't store

    return sparse_tableau, num_decision_vars, num_constraints, num_tableau_rows, num_tableau_cols, constraint_types

# Helper function to set values in sparse tableau dictionary
def _set_sparse_val(target_dict, r, c, val):
    if val == Fraction(0):
        target_dict.pop((r, c), None)
    else:
        target_dict[(r, c)] = val

def _helper_canonicalize_objective_row(tableau_to_modify, objective_row_index,
                                     variable_columns_to_check_range, all_tableau_columns_range,
                                     constraint_rows_range, verbose=False):
    """
    Makes the specified objective row canonical with respect to basic variables.
    A variable is basic in a row if it has a coefficient of 1 in that row and 0 in all other constraint rows for that column.
    If a basic variable (from variable_columns_to_check_range) has a non-zero coefficient
    in the objective row, row operations are performed to make it zero.

    Args:
        tableau_to_modify: The sparse dict representing the tableau.
        objective_row_index: Index of the row to be made canonical.
        variable_columns_to_check_range: e.g., range(num_vars_total_non_obj_non_rhs)
        all_tableau_columns_range: e.g., range(total_cols_including_rhs)
        constraint_rows_range: e.g., range(objective_row_index)
        verbose: Boolean for logging.
    """
    if verbose: print("Making objective row canonical...")
    for j_col in variable_columns_to_check_range:
        coeff_in_obj_row = tableau_to_modify.get((objective_row_index, j_col), Fraction(0))

        if coeff_in_obj_row == Fraction(0):
            continue # Already canonical for this variable column in the objective row.

        # Check if j_col is basic in one of the constraint_rows_range
        basic_row_idx = -1
        num_ones_in_col = 0
        non_zero_other_than_one_in_constr_rows = False

        for r_chk in constraint_rows_range:
            val_in_constr = tableau_to_modify.get((r_chk, j_col), Fraction(0))
            if val_in_constr == Fraction(1):
                if num_ones_in_col == 0: # First '1' found
                    basic_row_idx = r_chk
                num_ones_in_col += 1
            elif val_in_constr != Fraction(0):
                non_zero_other_than_one_in_constr_rows = True
                break

        if num_ones_in_col == 1 and not non_zero_other_than_one_in_constr_rows:
            # j_col is basic in basic_row_idx and its coeff_in_obj_row is non-zero.
            # This is the variable/row we need to use for canonicalization.
            if verbose: print(f"  Var in col {j_col} is basic in row {basic_row_idx}. Obj coeff is {coeff_in_obj_row}. Pivoting obj row.")

            # ObjRow_new = ObjRow_old - coeff_in_obj_row * BasicRow(basic_row_idx)
            for op_col in all_tableau_columns_range:
                obj_val_at_op_col = tableau_to_modify.get((objective_row_index, op_col), Fraction(0))
                basic_row_val_at_op_col = tableau_to_modify.get((basic_row_idx, op_col), Fraction(0))

                new_obj_val = obj_val_at_op_col - coeff_in_obj_row * basic_row_val_at_op_col
                _set_sparse_val(tableau_to_modify, objective_row_index, op_col, new_obj_val)
        # else: j_col is not basic in a unique constraint row, or its obj_coeff was already zero.
        # If coeff_in_obj_row was non-zero but var j_col was not basic, that's fine, no op needed for this col.

def construct_phase1_tableau(initial_tableau_data):
    """
    Constructs the Phase 1 tableau for the simplex algorithm.
    Phase 1 is used when artificial variables are needed (e.g., for surplus variables
    resulting from >= constraints, or for equality constraints).
    The goal of Phase 1 is to find a basic feasible solution by minimizing the sum
    of artificial variables. If the minimum sum is > 0, the original problem is infeasible.

    Args:
        initial_tableau_data: A tuple containing the output from initialize_tableau:
            (sparse_tableau_in, num_decision_vars, num_aux_vars,
             num_tableau_rows_in, num_tableau_cols_in, constraint_types_in)

    Returns:
        A tuple: (phase1_tableau, phase1_dims, artificial_var_info,
                  saved_original_obj_coeffs, constraint_types_in)
        OR
        (None, "No Phase 1 needed", initial_tableau_data, saved_original_obj_coeffs)
        if no artificial variables are required.

        phase1_tableau: Sparse dictionary for the Phase 1 problem.
        phase1_dims: Tuple with dimensions and key column indices for Phase 1 tableau
                     (p1_num_decision_vars, p1_num_aux_vars, p1_num_artificial_vars,
                      p1_num_tableau_rows, p1_num_tableau_cols, p1_W_var_col_idx, p1_rhs_col_idx).
        artificial_var_info: List of dicts, [{'original_row': r_idx, 'art_var_col': col_idx}]
                             mapping original constraint rows to their artificial variable columns.
        saved_original_obj_coeffs: Dictionary storing the original objective function coefficients.
        constraint_types_in: Passed through from input.
    """
    sparse_tableau_in, num_decision_vars, num_aux_vars, \
    num_tableau_rows_in, num_tableau_cols_in, constraint_types_in = initial_tableau_data

    phase1_tableau = {}
    artificial_var_info = []
    num_artificial_vars = 0
    # Artificial variables are added after decision and aux variables
    art_var_col_start_idx = num_decision_vars + num_aux_vars
    current_art_var_col = art_var_col_start_idx

    # 1. Identify rows needing artificial variables and map columns
    # Artificial variables are needed for rows that had surplus variables.
    # (original_row refers to the row index in sparse_tableau_in)
    for i in range(len(constraint_types_in)):
        if constraint_types_in[i] == 'surplus':
            # This row 'i' in the original problem now needs an artificial variable.
            # The artificial variable will get its own column in the phase 1 tableau.
            artificial_var_info.append({'original_row': i, 'art_var_col': current_art_var_col})
            num_artificial_vars += 1
            current_art_var_col += 1

    # 2. Save Original Objective Row Coefficients
    # These are needed to reconstruct the tableau for Phase 2.
    saved_original_obj_coeffs = {}
    obj_row_idx_in = num_tableau_rows_in - 1
    # Columns in original tableau: decision_vars | aux_vars | P_var | RHS
    old_P_var_col_idx_in = num_decision_vars + num_aux_vars
    old_rhs_col_idx_in = num_tableau_cols_in - 1

    # Iterate through decision and auxiliary variable columns in the original objective row
    for j in range(old_P_var_col_idx_in): # Covers decision_vars and aux_vars
        val = sparse_tableau_in.get((obj_row_idx_in, j), Fraction(0))
        if val != Fraction(0):
            saved_original_obj_coeffs[j] = val

    # Save P coefficient and RHS from original objective row
    # The P var itself might not be explicitly in saved_original_obj_coeffs if we reconstruct carefully,
    # but saving its column index and value (should be 1) is good.
    # Let's save all parts of the original objective row, including P and RHS.
    # The key for saved_original_obj_coeffs will be the column index.
    val_P = sparse_tableau_in.get((obj_row_idx_in, old_P_var_col_idx_in), Fraction(0))
    if val_P != Fraction(0): # Should be 1 for P
         saved_original_obj_coeffs[old_P_var_col_idx_in] = val_P

    val_rhs = sparse_tableau_in.get((obj_row_idx_in, old_rhs_col_idx_in), Fraction(0))
    if val_rhs != Fraction(0): # Original objective value, usually 0
        saved_original_obj_coeffs[old_rhs_col_idx_in] = val_rhs

    # 3. Handle "No Phase 1 Needed" case
    if num_artificial_vars == 0:
        # No artificial variables means the initial solution (with slacks) is feasible.
        return (None, "No Phase 1 needed", initial_tableau_data, saved_original_obj_coeffs, constraint_types_in)

    # 4. Define Phase 1 Tableau Dimensions
    p1_num_decision_vars = num_decision_vars
    p1_num_aux_vars = num_aux_vars
    p1_num_artificial_vars = num_artificial_vars

    # Columns: decision_vars | aux_vars | artificial_vars | W_var | RHS
    p1_W_var_col_idx = p1_num_decision_vars + p1_num_aux_vars + p1_num_artificial_vars
    p1_rhs_col_idx = p1_W_var_col_idx + 1
    p1_num_tableau_cols = p1_rhs_col_idx + 1

    p1_num_tableau_rows = num_tableau_rows_in # Number of rows remains the same
    p1_obj_row_idx = p1_num_tableau_rows - 1

    # Helper for setting values in sparse tableau is now defined globally as _set_sparse_val

    # 5. Construct Phase 1 Tableau (Constraint Rows)
    # Copy from sparse_tableau_in to phase1_tableau
    # For each constraint row r from 0 to p1_obj_row_idx - 1:
    for r in range(p1_obj_row_idx):
        # Copy decision variable coefficients
        for c_dec in range(p1_num_decision_vars):
            val = sparse_tableau_in.get((r, c_dec), Fraction(0))
            _set_sparse_val(phase1_tableau, r, c_dec, val)

        # Copy auxiliary variable coefficients
        # Original aux var columns are from num_decision_vars to num_decision_vars + num_aux_vars - 1
        # New aux var columns are also from p1_num_decision_vars to p1_num_decision_vars + p1_num_aux_vars - 1
        for c_aux_offset in range(p1_num_aux_vars):
            orig_aux_col = num_decision_vars + c_aux_offset
            new_aux_col = p1_num_decision_vars + c_aux_offset
            val = sparse_tableau_in.get((r, orig_aux_col), Fraction(0))
            _set_sparse_val(phase1_tableau, r, new_aux_col, val)

        # Copy RHS
        val = sparse_tableau_in.get((r, old_rhs_col_idx_in), Fraction(0))
        _set_sparse_val(phase1_tableau, r, p1_rhs_col_idx, val)

    # Add artificial variable coefficients (column of 1s for each art_var in its row)
    for art_info in artificial_var_info:
        # art_info['original_row'] is the row index
        # art_info['art_var_col'] is the column index for this new artificial variable
        _set_sparse_val(phase1_tableau, art_info['original_row'], art_info['art_var_col'], Fraction(1))

    # 6. Create Initial Phase 1 Objective Row (W)
    # Objective: Minimize sum of artificial variables (e.g., a1 + a2 + ...)
    # This is equivalent to Maximize W = -(a1 + a2 + ...) => W + a1 + a2 + ... = 0
    # So, coefficients of artificial variables in W row are 1.
    # To match the tableau format (obj_coeff = -actual_coeff), we use -1 for art_vars.
    # However, standard Phase 1 setup for maximization tableau: W = Sum(art_vars), then make canonical.
    # Let's set up to Maximize W = - (sum of art_vars).
    # The objective row for "Maximize W = -a1 -a2 ..." is  W + a1 + a2 + ... = 0
    # This means obj coeffs for a_i are 1. Then pivot to make them 0.
    # Let's use the common approach: W = sum of artificial variables, then subtract rows.
    # Obj: Min (a_i). So Max (- sum a_i). Let W = -sum(a_i).
    # W + a_1 + a_2 + ... = 0.  Tableau row: [ ... 1 1 ... 1(W) 0(RHS) ]
    # where 1s are for a_i.

    # Initial W row: set coefficients for artificial variables to 1.
    for art_info in artificial_var_info:
        _set_sparse_val(phase1_tableau, p1_obj_row_idx, art_info['art_var_col'], Fraction(1)) # Coeff for art_var_i is 1

    _set_sparse_val(phase1_tableau, p1_obj_row_idx, p1_W_var_col_idx, Fraction(1)) # Coeff for W is 1
    _set_sparse_val(phase1_tableau, p1_obj_row_idx, p1_rhs_col_idx, Fraction(0))   # RHS for W is 0

    # 7. Make Phase 1 Objective Row Consistent (Canonical Form)
    # For each artificial variable a_k (which is basic in row i_k, with coefficient 1):
    # New W_row = Old W_row - Row_i_k (for each artificial var that was made basic)
    # This is now handled by a general canonicalization helper.
    # The W-row was set up with 1s for artificial variables. These are the ones that need to be zeroed out.
    # The variable_columns_to_check_range should cover all variables that might be basic,
    # but specifically, we are interested in the artificial variable columns that were just made basic.
    # The logic inside _helper_canonicalize_objective_row will find any basic var in its range
    # that has a non-zero coefficient in the objective row and fix it.
    # For Phase 1, the artificial variables are basic in their respective constraint rows (coeff 1),
    # and have a coeff of 1 in the W-row before this step.
    _helper_canonicalize_objective_row(
        phase1_tableau,
        p1_obj_row_idx,
        range(p1_W_var_col_idx), # Check all var cols: decision, aux, artificial
        range(p1_num_tableau_cols), # Operate on all cols: decision, aux, art, W, RHS
        range(p1_obj_row_idx), # Constraint rows range
        verbose=False # construct_phase1_tableau does not currently take verbose flag
    )

    # 8. Return Phase 1 Data
    phase1_dims = (p1_num_decision_vars, p1_num_aux_vars, p1_num_artificial_vars,
                   p1_num_tableau_rows, p1_num_tableau_cols,
                   p1_W_var_col_idx, p1_rhs_col_idx)

    return phase1_tableau, phase1_dims, artificial_var_info, saved_original_obj_coeffs, constraint_types_in


class SimplexSolver:
    def __init__(self, c_orig, A_orig, b_orig, sparse_input=False):
        """
        Initializes the SimplexSolver with the problem definition.
        It sets up for Phase 1 if needed, otherwise prepares for direct solution.
        Args:
            c_orig: Coefficients for the objective function.
                    List if sparse_input=False, Dict if sparse_input=True.
            A_orig: Constraint matrix.
                    List of lists if sparse_input=False, List of Dicts if sparse_input=True.
            b_orig: List of RHS values for constraints.
            sparse_input: Boolean, True if c_orig and A_orig are in sparse format.
        """
        if sparse_input:
            initial_tableau_data = initialize_tableau_sparse(c_orig, A_orig, b_orig)
        else:
            initial_tableau_data = initialize_tableau(c_orig, A_orig, b_orig)
        # sparse_tableau_in, num_decision_vars, num_aux_vars, num_rows_in, num_cols_in, constraint_types_in

        self.original_num_decision_vars = initial_tableau_data[1]
        self.constraint_types = initial_tableau_data[5] # constraint_types_in

        phase1_result = construct_phase1_tableau(initial_tableau_data)
        p1_tableau, p1_dims, p1_art_var_info, p1_saved_orig_obj_coeffs, _ = phase1_result

        self.saved_original_obj_coeffs = p1_saved_orig_obj_coeffs
        self.artificial_var_info = p1_art_var_info # List of dicts {'original_row': r, 'art_var_col': c}

        if p1_tableau is not None: # Phase 1 is needed
            self.is_phase1_needed = True
            self.current_phase = 1
            self.tableau = p1_tableau
            # Unpack p1_dims: (p1_num_decision_vars, p1_num_aux_vars, p1_num_artificial_vars,
            #                  p1_num_tableau_rows, p1_num_tableau_cols,
            #                  p1_W_var_col_idx, p1_rhs_col_idx)
            self.num_decision_vars = p1_dims[0] # These are original decision vars
            self.num_aux_vars = p1_dims[1]
            self.num_artificial_vars = p1_dims[2]
            self.rows = p1_dims[3]
            self.cols = p1_dims[4]
            self.objective_var_col_idx = p1_dims[5] # W_var_col_idx for Phase 1
            self.rhs_col_idx = p1_dims[6]
        else: # No Phase 1 needed
            self.is_phase1_needed = False
            self.current_phase = 2 # Proceed directly to Phase 2 logic
            self.tableau = initial_tableau_data[0] # sparse_tableau_in
            self.num_decision_vars = initial_tableau_data[1]
            self.num_aux_vars = initial_tableau_data[2]
            self.num_artificial_vars = 0
            self.rows = initial_tableau_data[3]
            self.cols = initial_tableau_data[4]
            # Objective var (P) is after decision and aux vars
            self.objective_var_col_idx = self.num_decision_vars + self.num_aux_vars
            self.rhs_col_idx = self.cols - 1

        self.iteration = 0

    def _get_tableau_value(self, row, col):
        """Helper to get value from sparse tableau, defaults to Fraction(0)."""
        return self.tableau.get((row, col), Fraction(0))

    def _set_tableau_value(self, row, col, value):
        """Helper to set value in sparse tableau. Removes entry if value is zero."""
        _set_sparse_val(self.tableau, row, col, value) # Use the global helper

    def _find_pivot_column(self):
        """Finds the pivot column (entering variable).
        Selects the column with the most negative coefficient in the objective function row.
        Returns the column index or -1 if no negative coefficient is found (optimal).
        """
        objective_row_idx = self.rows - 1
        min_val = Fraction(0)
        pivot_col = -1
        # Iterate through decision, auxiliary, and artificial (if phase 1) variable columns
        for j in range(self.objective_var_col_idx): # Columns up to P or W variable
            val = self._get_tableau_value(objective_row_idx, j)
            if val < min_val:
                min_val = val
                pivot_col = j
        return pivot_col

    def _find_pivot_row(self, pivot_col):
        """Finds the pivot row (leaving variable) using the minimum ratio test.
        Returns the row index or -1 if unbounded or other issue.
        Assumes RHS is non-negative (handled by initialize_tableau).
        """
        min_ratio = float('inf')
        pivot_row = -1
        # rhs_col_idx is now self.rhs_col_idx (set in __init__)

        for i in range(self.rows - 1): # Exclude objective function row
            pivot_col_val = self._get_tableau_value(i, pivot_col)

            if pivot_col_val > Fraction(0): # Consider only positive elements in pivot column
                rhs_val = self._get_tableau_value(i, self.rhs_col_idx) # Should be non-negative

                # Ratio test: RHS / pivot_col_val
                # If rhs_val is 0 and pivot_col_val > 0, ratio is 0. This is a valid and often preferred pivot.
                ratio = rhs_val / pivot_col_val

                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
                # Note: Degeneracy (multiple rows with same min_ratio) is not specially handled here.
                # Bland's rule could be implemented to prevent cycling in degenerate cases.

        # If all pivot_col_val <= 0 for rows with rhs_val >=0, then pivot_row remains -1 (unbounded)
        return pivot_row


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
        s = []
        dense_tableau_for_print = []
        for i in range(self.rows):
            row_list = []
            for j in range(self.cols):
                row_list.append(self._get_tableau_value(i,j)) # Use self._get_tableau_value
            dense_tableau_for_print.append(row_list)

        col_widths = [0] * self.cols
        # Step 1: Calculate initial widths based on cell values (data)
        for r_list in dense_tableau_for_print:
            for idx, cell_val in enumerate(r_list):
                col_widths[idx] = max(col_widths[idx], len(str(cell_val)))

        # Step 2: Generate final raw header strings
        raw_header_labels = [""] * self.cols # Initialize with placeholders matching number of columns

        # Populate raw_header_labels based on column type
        # Decision variables
        # self.num_decision_vars in Phase 1 might include artificial vars if not careful.
        # self.original_num_decision_vars is the true count of x_i variables.
        for j in range(self.original_num_decision_vars):
            if j < self.cols: raw_header_labels[j] = f"x{j+1}"

        current_col_idx = self.original_num_decision_vars
        # Auxiliary variables
        # self.num_aux_vars is the count of slack/surplus variables.
        # self.constraint_types length should match self.num_aux_vars (from original problem setup)
        for aux_var_num in range(self.num_aux_vars):
            if current_col_idx < self.cols:
                # Ensure constraint_types has this index, can happen if num_aux_vars changes unexpectedly
                # However, self.constraint_types should be static based on original problem.
                var_type_char = 's' if aux_var_num < len(self.constraint_types) and self.constraint_types[aux_var_num] == 'slack' else 'e'
                raw_header_labels[current_col_idx] = f"{var_type_char}{aux_var_num+1}"
            current_col_idx += 1

        # Artificial variables (if they are part of the current tableau structure)
        # self.num_artificial_vars is set in __init__ based on current phase
        if self.num_artificial_vars > 0:
            for art_var_num in range(self.num_artificial_vars):
                if current_col_idx < self.cols:
                     raw_header_labels[current_col_idx] = f"a{art_var_num+1}"
                current_col_idx += 1

        # Objective variable (P or W) - its column index is self.objective_var_col_idx
        if self.objective_var_col_idx < self.cols:
            active_obj_var_char = "W" if self.current_phase == 1 else "P"
            raw_header_labels[self.objective_var_col_idx] = active_obj_var_char

        # RHS - its column index is self.rhs_col_idx
        if self.rhs_col_idx < self.cols:
            raw_header_labels[self.rhs_col_idx] = "RHS"

        # Step 3: Update col_widths using the lengths of raw_header_labels
        for j in range(self.cols):
            col_widths[j] = max(col_widths[j], len(raw_header_labels[j]))

        # Previous DEBUG print lines for col_widths are now removed.

        # Step 4: Construct the actual display header string using ljust and final col_widths
        header_display_parts = []
        for j in range(self.cols):
            header_display_parts.append(raw_header_labels[j].ljust(col_widths[j]))

        s.append(" | ".join(header_display_parts))

        # Step 5: Separator line
        s.append("-+-".join(["-" * col_widths[j] for j in range(self.cols)]))

        # Step 6: Data rows
        for i in range(self.rows): # Iterate all rows including objective
            row_str_list = []
            for j in range(self.cols): # Iterate all actual columns in the tableau data
                val = self._get_tableau_value(i,j) # Use self._get_tableau_value
                row_str_list.append(str(val).ljust(col_widths[j]))
            s.append(" | ".join(row_str_list))

        return "\n".join(s)

    def _execute_simplex_iterations(self, verbose=False, max_iterations=100):
        """
        Core simplex algorithm loop. Operates on the current self.tableau.
        """
        # self.iteration should be managed by the caller if called multiple times (e.g. phase1 then phase2)
        # For now, let it be reset or continued if solve() calls this multiple times.
        # Let's assume it's reset for each call to solve -> _execute_simplex_iterations
        # self.iteration = 0 # Resetting here means iterations are per-phase.

        if verbose:
            phase_name = "Phase 1" if self.current_phase == 1 else "Phase 2"
            print(f"\n--- Starting {phase_name} ---")
            title = f"Initial {phase_name} Tableau"
            if self.current_phase == 1:
                 title += f" (Dec={self.num_decision_vars}, Aux={self.num_aux_vars}, Art={self.num_artificial_vars})"
            else: # Phase 2 (or direct solve)
                 title += f" (Dec={self.original_num_decision_vars}, Aux={self.num_aux_vars})"
            print(f"{title}:\n{self._format_tableau()}\n")


        while self.iteration < max_iterations:
            pivot_col = self._find_pivot_column()

            if pivot_col == -1:
                if verbose: print("Optimal solution found for current phase.")
                return "optimal"

            if verbose: print(f"Iteration {self.iteration+1}: Pivot column is {pivot_col} (0-indexed)")

            pivot_row = self._find_pivot_row(pivot_col)

            if pivot_row == -1: # No valid pivot row found
                all_pivot_col_entries_non_positive_for_positive_rhs = True
                # rhs_col_idx is self.rhs_col_idx
                for i in range(self.rows - 1): # Check all constraint rows
                    if self._get_tableau_value(i, self.rhs_col_idx) >= 0 and self._get_tableau_value(i, pivot_col) > 0:
                        all_pivot_col_entries_non_positive_for_positive_rhs = False
                        break

                if all_pivot_col_entries_non_positive_for_positive_rhs:
                    if verbose: print(f"Problem is unbounded. Pivot column {pivot_col} has all non-positive entries in constraint rows with non-negative RHS.")
                    return "unbounded"
                else:
                    if verbose: print(f"Warning: Pivot column {pivot_col} found, but no suitable pivot row.")
                    return "error_no_suitable_pivot_row"

            if verbose: print(f"Pivot element at ({pivot_row}, {pivot_col}): {self._get_tableau_value(pivot_row, pivot_col)}")

            self._pivot(pivot_row, pivot_col)
            self.iteration += 1
            if verbose: print(f"After Pivot {self.iteration}:\n{self._format_tableau()}\n")

        if verbose: print("Max iterations reached.")
        return "max_iterations_reached"

    def solve(self, verbose=False, max_iterations=100):
        """
        Main solve method for the simplex algorithm. Handles Phase 1 if necessary.
        """
        self.iteration = 0 # Reset iteration count for the solve process

        if self.current_phase == 1:
            if verbose: print("=== Starting Phase 1 ===")
            phase1_status = self._execute_simplex_iterations(verbose=verbose, max_iterations=max_iterations)

            if phase1_status == "optimal":
                # Check W value (Phase 1 objective value)
                # Objective row is self.rows - 1, RHS column is self.rhs_col_idx
                W_value = self._get_tableau_value(self.rows - 1, self.rhs_col_idx)

                # Using a small tolerance for comparing W_value to zero with Fractions might not be necessary
                # as Fraction arithmetic is exact. W_value should be exactly 0 if feasible.
                if W_value < Fraction(0): # Max W = -sum(art_vars). If W_opt < 0 => sum(art_vars) > 0
                    if verbose: print(f"Phase 1 optimal, but W = {W_value} < 0. Original problem is infeasible.")
                    return "infeasible"
                else: # W_value == 0
                    if verbose: print(f"Phase 1 optimal and W = {W_value}. Proceeding to Phase 2.")
                    self._prepare_for_phase2(verbose)
                    # self.current_phase is set to 2 inside _prepare_for_phase2
                    if verbose: print("=== Starting Phase 2 (after successful Phase 1) ===")
                    # Iteration count for Phase 2 continues from Phase 1, or could be reset in _prepare_for_phase2 if desired
                    return self._execute_simplex_iterations(verbose=verbose, max_iterations=max_iterations)
            else: # Phase 1 not optimal (e.g., unbounded, max_iterations)
                if verbose: print(f"Phase 1 resulted in status: {phase1_status}. Original problem likely infeasible.")
                return "infeasible" # Or map phase1_status more directly

        elif self.current_phase == 2: # Directly to Phase 2 (no Phase 1 was needed)
            if verbose:
                 print("=== Starting Simplex Method (Direct Solve) ===")
            # Iteration count for Phase 2 should ideally start from 0 or be managed.
            # self.iteration was reset at the start of solve().
            return self._execute_simplex_iterations(verbose=verbose, max_iterations=max_iterations)
        else:
            raise RuntimeError(f"Unknown current_phase: {self.current_phase}")

    def _prepare_for_phase2(self, verbose=False):
        """
        Prepares the tableau for Phase 2 after a successful Phase 1.
        This involves removing artificial variables, restoring the original objective function,
        and ensuring the tableau is in canonical form for the original objective.
        """
        if verbose:
            print("\n--- Preparing for Phase 2 ---")
            # print("Final Phase 1 Tableau:")
            # print(self._format_tableau()) # Current tableau is still Phase 1 here

        # Define new column indices (Post-Artificial Variable Removal)
        phase2_num_decision_vars = self.original_num_decision_vars
        phase2_num_aux_vars = self.num_aux_vars # Aux vars from Phase 1 (slacks/surplus)

        phase2_P_var_col_idx = phase2_num_decision_vars + phase2_num_aux_vars
        phase2_rhs_col_idx = phase2_P_var_col_idx + 1
        phase2_cols = phase2_rhs_col_idx + 1
        # Number of rows (constraints + 1 objective) remains the same: self.rows

        phase2_tableau = {}
        obj_row_idx = self.rows - 1 # Objective row index is the last row

        # Copy Constraint Rows (excluding artificial variable columns)
        for r in range(obj_row_idx): # Iterate through constraint rows
            # Copy decision variable coefficients
            for c_dec in range(self.original_num_decision_vars):
                val = self._get_tableau_value(r, c_dec) # Read from final Phase 1 tableau
                _set_sparse_val(phase2_tableau, r, c_dec, val)

            # Copy auxiliary variable coefficients
            # Original aux vars in Phase 1 tableau are at cols: self.original_num_decision_vars to self.original_num_decision_vars + self.num_aux_vars - 1
            # Their new positions in Phase 2 tableau are the same relative indices.
            for c_aux_offset in range(self.num_aux_vars):
                # Column in Phase 1 tableau:
                p1_aux_col = self.original_num_decision_vars + c_aux_offset
                # Column in Phase 2 tableau: (same)
                p2_aux_col = phase2_num_decision_vars + c_aux_offset
                val = self._get_tableau_value(r, p1_aux_col)
                _set_sparse_val(phase2_tableau, r, p2_aux_col, val)

            # Copy RHS
            val = self._get_tableau_value(r, self.rhs_col_idx) # self.rhs_col_idx is from Phase 1
            _set_sparse_val(phase2_tableau, r, phase2_rhs_col_idx, val)

        # Restore Original Objective Function into phase2_tableau's objective row
        orig_P_col_in_saved_coeffs = self.original_num_decision_vars + self.num_aux_vars
        orig_RHS_col_in_saved_coeffs = orig_P_col_in_saved_coeffs + 1

        for col_key, val in self.saved_original_obj_coeffs.items():
            if col_key < self.original_num_decision_vars: # Decision variable
                _set_sparse_val(phase2_tableau, obj_row_idx, col_key, val)
            elif col_key < orig_P_col_in_saved_coeffs: # Auxiliary variable
                # col_key in saved_coeffs is the original index (dec_vars + aux_offset)
                # This maps directly to the same column index in phase2_tableau's new structure
                # because aux vars follow decision vars.
                _set_sparse_val(phase2_tableau, obj_row_idx, col_key, val)
            elif col_key == orig_P_col_in_saved_coeffs: # P-variable coefficient
                _set_sparse_val(phase2_tableau, obj_row_idx, phase2_P_var_col_idx, val)
            elif col_key == orig_RHS_col_in_saved_coeffs: # RHS value
                _set_sparse_val(phase2_tableau, obj_row_idx, phase2_rhs_col_idx, val)
            # else: Malformed saved_original_obj_coeffs key, or not handled.

        # Make Restored Objective Row Canonical
        # For each column j that is basic in a constraint row r (in phase2_tableau),
        # the coefficient in the objective row for column j must be zero.
        _helper_canonicalize_objective_row(
            phase2_tableau,
            obj_row_idx,
            range(phase2_P_var_col_idx), # Check decision and aux variable columns
            range(phase2_cols),          # Operate on all columns including P and RHS
            range(obj_row_idx),          # Constraint rows in phase2_tableau
            verbose=verbose
        )

        # Update Solver State for Phase 2
        self.tableau = phase2_tableau
        # self.num_decision_vars remains self.original_num_decision_vars for Phase 2 logic
        # self.num_aux_vars remains self.num_aux_vars (from Phase 1)
        self.num_artificial_vars = 0 # Critical change
        self.cols = phase2_cols
        self.objective_var_col_idx = phase2_P_var_col_idx
        self.rhs_col_idx = phase2_rhs_col_idx
        # self.rows remains the same
        self.current_phase = 2 # Officially in Phase 2
        self.artificial_var_info = [] # Clear this

        if verbose:
            print("Phase 2 Tableau ready.")
            # print(self._format_tableau()) # Now self.tableau is phase2_tableau

    def get_solution(self):
        """
        Extracts variable values and objective function value from the final tableau.
        Reports original decision variables.
        """
        solution = {}
        # rhs_col_idx and obj_row_idx are set based on current phase tableau
        obj_row_idx = self.rows - 1

        # Objective value
        # If current_phase is 1, this is W. If 2, this is P.
        # The value retrieved is from the current tableau's objective row.
        active_obj_char = "W" if self.current_phase == 1 else "P"
        solution[f'{active_obj_char}_objective_value'] = self._get_tableau_value(obj_row_idx, self.rhs_col_idx)

        row_has_sourced_basic_var = [False] * (self.rows - 1)

        # Decision variables values (use self.original_num_decision_vars)
        for j in range(self.original_num_decision_vars):
            val = Fraction(0)
            basic_row_candidate = -1
            is_basic_column = True
            count_ones_in_constraints = 0
            for i in range(self.rows - 1): # Constraint rows
                cell_val = self._get_tableau_value(i, j)
                if cell_val == Fraction(1):
                    count_ones_in_constraints +=1
                    basic_row_candidate = i
                elif cell_val != Fraction(0):
                    is_basic_column = False
                    break

            if is_basic_column and count_ones_in_constraints == 1 and \
               self._get_tableau_value(obj_row_idx, j) == Fraction(0):
                if not row_has_sourced_basic_var[basic_row_candidate]:
                    val = self._get_tableau_value(basic_row_candidate, self.rhs_col_idx)
                    row_has_sourced_basic_var[basic_row_candidate] = True
                # else: val remains Fraction(0) as row already sourced a basic var
            # else: val remains Fraction(0) as variable j is not basic
            solution[f'x{j+1}'] = val

        # Auxiliary (slack/surplus) variables values
        # These are in columns from self.original_num_decision_vars to self.original_num_decision_vars + self.num_aux_vars -1
        if self.num_aux_vars > 0: # If there are any auxiliary variables
            for j_aux_offset in range(self.num_aux_vars):
                aux_var_col_idx = self.original_num_decision_vars + j_aux_offset
                val = Fraction(0)
                basic_row_candidate = -1
                is_basic_this_aux_var = True
                num_ones_in_col = 0

                for i in range(self.rows - 1): # Constraint rows
                    cell_val = self._get_tableau_value(i, aux_var_col_idx)
                    if cell_val == Fraction(1): # Basic aux vars should have a 1 after pivots
                        num_ones_in_col +=1
                        basic_row_candidate = i
                    elif cell_val != Fraction(0):
                        is_basic_this_aux_var = False
                        break

                if is_basic_this_aux_var and num_ones_in_col == 1 and \
                   self._get_tableau_value(obj_row_idx, aux_var_col_idx) == Fraction(0):
                    if not row_has_sourced_basic_var[basic_row_candidate]:
                        val = self._get_tableau_value(basic_row_candidate, self.rhs_col_idx)
                        row_has_sourced_basic_var[basic_row_candidate] = True
                    # else: val remains Fraction(0)
                # else: val remains Fraction(0)

                # self.constraint_types refers to the original problem's constraints
                var_type_char = 's' if self.constraint_types[j_aux_offset] == 'slack' else 'e'
                solution[f'{var_type_char}{j_aux_offset+1}'] = val

        # Optionally, report artificial variable values if in Phase 1 and they are basic
        # This part should also ideally use row_has_sourced_basic_var if artificial vars can be basic in phase 1 final tableau
        # For now, assuming Phase 1 ensures artificial vars are non-basic if W=0, or problem is infeasible.
        # If an artificial var *is* still basic and W=0 (degenerate case), the current logic might misreport.
        # However, the primary goal of Phase 1 is to drive them to 0.
        if self.current_phase == 1 and self.num_artificial_vars > 0:
            art_var_start_col = self.original_num_decision_vars + self.num_aux_vars
            for k in range(self.num_artificial_vars):
                art_var_col_idx = art_var_start_col + k
                val = Fraction(0)
                basic_row_candidate = -1
                is_basic_art_var = True
                num_ones_in_col = 0
                for i in range(self.rows -1):
                    cell_val = self._get_tableau_value(i, art_var_col_idx)
                    if cell_val == Fraction(1):
                        num_ones_in_col +=1
                        basic_row_candidate = i
                    elif cell_val != Fraction(0):
                        is_basic_art_var = False; break
                if is_basic_art_var and num_ones_in_col == 1 and \
                   self._get_tableau_value(obj_row_idx, art_var_col_idx) == Fraction(0): # Should be 0 for basic art vars in W row
                    val = self._get_tableau_value(basic_row_candidate, self.rhs_col_idx)
                solution[f'a{k+1}'] = val

        return solution

if __name__ == '__main__':
    # Example 1: Maximize P = 3x1 + 2x2 (No Phase 1 needed)
    # Subject to: x1 + x2 <= 10, 2x1 + x2 <= 15
    print("--- Example 1 (Optimal - Direct Solve) ---")
    c1 = [Fraction(3), Fraction(2)]
    A1 = [[Fraction(1), Fraction(1)], [Fraction(2), Fraction(1)]]
    b1 = [Fraction(10), Fraction(15)]
    solver1 = SimplexSolver(c1, A1, b1)
    status1 = solver1.solve(verbose=True)
    print(f"\nSolver status (Example 1): {status1}")
    if status1 == 'optimal':
        print("Final Tableau (Example 1):")
        print(solver1._format_tableau())
        solution1 = solver1.get_solution()
        print("\nSolution (Example 1):")
        for var, val in solution1.items():
            print(f"{var}: {val}")

    # Example 2: Unbounded (No Phase 1 needed)
    # Max P = x1 + x2
    # s.t. -x1 + x2 <= 1, x1 - 2x2 <= 2
    print("\n--- Example 2 (Unbounded - Direct Solve) ---")
    c2 = [Fraction(1), Fraction(1)]
    A2 = [[Fraction(-1), Fraction(1)], [Fraction(1), Fraction(-2)]]
    b2 = [Fraction(1), Fraction(2)]
    solver2 = SimplexSolver(c2, A2, b2)
    status2 = solver2.solve(verbose=True)
    print(f"\nSolver status (Example 2): {status2}")
    if status2 == 'unbounded':
        print("Final Tableau (Example 2 - Unbounded):")
        print(solver2._format_tableau())
        print("Problem correctly identified as unbounded.")

    # Example 3: Wikipedia (No Phase 1 needed)
    print("\n--- Example 3 (Wikipedia - Direct Solve) ---")
    c3 = [Fraction(2), Fraction(3), Fraction(4)]
    A3 = [[Fraction(3), Fraction(2), Fraction(1)], [Fraction(2), Fraction(5), Fraction(3)]]
    b3 = [Fraction(10), Fraction(15)]
    solver3 = SimplexSolver(c3, A3, b3)
    status3 = solver3.solve(verbose=True)
    print(f"\nSolver status (Example 3): {status3}")
    if status3 == 'optimal':
        print("Final Tableau (Example 3):")
        print(solver3._format_tableau())
        solution3 = solver3.get_solution()
        print("\nSolution (Example 3):")
        for var, val in solution3.items():
            print(f"{var}: {val}")

    # Example 4: Constraint b_i < 0 (Phase 1 needed, should be INFEASIBLE)
    # Max P = x1 + x2
    # s.t. x1 + x2 <= -1
    print("\n--- Example 4 (Infeasible - Phase 1 Triggered) ---")
    c4 = [Fraction(1), Fraction(1)]
    A4 = [[Fraction(1), Fraction(1)]]
    b4 = [Fraction(-1)] # Requires Phase 1 because of negative b
    solver4 = SimplexSolver(c4, A4, b4)
    status4 = solver4.solve(verbose=True)
    print(f"\nSolver status (Example 4): {status4}")
    if status4 == 'infeasible':
        print("Problem correctly identified as infeasible by Phase 1.")
        # Optionally print Phase 1 final tableau
        print("Final Tableau (Phase 1 - Example 4):")
        print(solver4._format_tableau())
        # Solution from get_solution will show W's value
        solution4 = solver4.get_solution()
        print("\nSolution data from Phase 1 (Example 4):")
        for var, val in solution4.items():
            print(f"{var}: {val}")
    elif status4 == "phase1_optimal_proceed_to_phase2": # Should not happen for this problem
        print("Error: Example 4 should be infeasible, not proceed to Phase 2.")
        print("Final Tableau (Phase 1 - Example 4):")
        print(solver4._format_tableau())


    # Example 5: A problem that is feasible after Phase 1
    # Max P = 3x1 + 5x2
    # s.t. x1 <= 4  (x1 + s1 = 4)
    #      2x2 <= 12 (2x2 + s2 = 12)
    #      3x1 + 2x2 >= 18 (3x1 + 2x2 - e3 + a1 = 18) -> Phase 1 needed
    print("\n--- Example 5 (Feasible - Phase 1 to Phase 2) ---")
    c5 = [Fraction(3), Fraction(5)]
    A5 = [
        [Fraction(1), Fraction(0)],
        [Fraction(0), Fraction(2)],
        [Fraction(-3), Fraction(-2)] # Changed to model 3x1+2x2 >= 18 correctly
    ]
    # To make the third constraint >=, we model it as -Ax <= -b.
    # So, for 3x1 + 2x2 >= 18, input A_row = [-3, -2] and b = -18.
    # initialize_tableau will flip it to current_A_row = [3,2], current_b = 18, type='surplus'.
    # or handle >= directly in initialize_tableau.
    # Our current initialize_tableau: if b_i < 0 for A_i x <= b_i, it flips.
    # So, to model 3x1 + 2x2 >= 18, we can write it as -3x1 - 2x2 <= -18.
    b5 = [Fraction(4), Fraction(12), Fraction(-18)] # Last one will be flipped by initialize_tableau
                                                 # -3x1 -2x2 <= -18  => 3x1+2x2 >= 18 => 3x1+2x2-e3=18
                                                 # constraint_types should be ['slack', 'slack', 'surplus']
    solver5 = SimplexSolver(c5, A5, b5)
    status5 = solver5.solve(verbose=True)
    print(f"\nSolver status (Example 5): {status5}")

    if status5 == 'optimal': # This will be from Phase 2 if fully implemented
        print("Final Tableau (Phase 2 - Example 5):")
        print(solver5._format_tableau())
        solution5 = solver5.get_solution()
        print("\nSolution (Example 5):")
        for var, val in solution5.items():
            print(f"{var}: {val}")
    elif status5 == "phase1_optimal_proceed_to_phase2":
        print("Phase 1 was optimal, ready for Phase 2.")
        print("Current (end of Phase 1) Tableau (Example 5):")
        # When printing the tableau at the end of successful phase 1,
        # self.current_phase might have already been set to 2 by the solve() method
        # if we were to fully implement _prepare_for_phase2.
        # For now, solve() returns "phase1_optimal_proceed_to_phase2" and current_phase is still 1.
        print(solver5._format_tableau()) # This should use 'W' due to current_phase=1
        solution5_phase1 = solver5.get_solution() # This should also report W_objective_value
        print("\nSolution data from end of Phase 1 (Example 5):")
        for var, val in solution5_phase1.items():
            print(f"{var}: {val}")
    elif status5 == 'infeasible':
         print("Problem (Example 5) found infeasible.")

    # Remove direct tests of construct_phase1_tableau as it's internal now
    # (The test code for construct_phase1_tableau from previous step is implicitly removed by this overwrite)
