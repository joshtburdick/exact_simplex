import unittest
from fractions import Fraction
from simplex_algorithm.simplex import initialize_tableau, SimplexSolver

class TestEqualityConstraints(unittest.TestCase):
    def test_optimal_solution_example1(self):
        # Max P = 3x1 + 2x2
        # s.t. x1 + x2 = 10, 2x1 + x2 <= 15
        #   (note that first constraint is equality)
        c = [Fraction(3), Fraction(2)]
        A = [[Fraction(1), Fraction(1)],
             [Fraction(-1), Fraction(-1)],
             [Fraction(2), Fraction(1)]]
        b = [Fraction(10), Fraction(-10), Fraction(15)]
        # Updated constructor
        solver = SimplexSolver(c, A, b)
        status = solver.solve()

        self.assertEqual(status, "optimal")
        solution = solver.get_solution()
        # Solution key is P_objective_value now
        self.assertEqual(solution['P_objective_value'], Fraction(25))
        self.assertEqual(solution['x1'], Fraction(5))
        self.assertEqual(solution['x2'], Fraction(5))

